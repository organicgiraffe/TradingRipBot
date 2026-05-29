"""
webhook_server.py -- TradingView webhook-driven trading bot for IBKR.

Architecture:
  TradingView chart (real-time Ripster signals)
    -> webhook POST
      -> Flask server (this file, port 5000)
        -> IBKR TWS (entry order + position management)

Run:
  1. Start TWS / IB Gateway (paper: port 7497, live: 7496)
  2. python webhook_server.py
  3. ngrok http 5000          <- in a separate terminal
  4. Copy the https://xxx.ngrok-free.app URL
  5. Paste it into TradingView alert > Notifications > Webhook URL

Alert JSON format (set this as the alert message in TradingView):
  {"symbol":"{{ticker}}","action":"long","price":{{close}},"stop":{{plot_0}},"signal":"cloud_flip"}

Endpoints:
  POST /webhook        <- TradingView sends alerts here
  GET  /status         <- see positions + queue in browser
  POST /close/SYMBOL   <- emergency manual close
"""

import json
import logging
import os
import pathlib
import queue
import threading
from datetime import datetime, date
from typing import Optional

from flask import Flask, request, jsonify
from ib_insync import IB, Stock, Order

from config import (
    TWS_HOST, TWS_PORT,
    FIXED_SHARES, FIXED_SHARES_HIGH, HIGH_PRICE_THRESHOLD,
    STARTER_RATIO, ADD_TRIGGER_PROFIT,
    MAX_RISK_DOLLARS, MAX_RISK_DOLLARS_HIGH,
    MAX_TRADES_PER_DAY, MAX_SIMULTANEOUS_POSITIONS,
    LEVEL_PROX_LONG, LEVEL_PROX_SHORT,
    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
    FIRST_ENTRY_MINUTE, LAST_ENTRY_HOUR, LAST_ENTRY_MINUTE,
    BAR_SIZE_3M, BAR_SIZE_10M,
    PROFIT_TARGET_SHARE,
    DEBUG_SIGNALS,
)
from ema_engine import compute_emas, compute_trailing_stop
from position import Position

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _setup_trade_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"trades_{date.today():%Y-%m-%d}_tv.log")
    tlog = logging.getLogger("trade_log_tv")
    tlog.setLevel(logging.DEBUG)
    if not tlog.handlers:
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        tlog.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        tlog.addHandler(ch)
    return tlog


tlog = _setup_trade_logger()

# ------------------------------------------------------------------ #
# Flask app + thread-safe alert queue
# ------------------------------------------------------------------ #

app = Flask(__name__)

# Webhook thread puts raw alert dicts here; IB main loop pops them.
# This keeps all IB calls on the main thread -- no threading headaches.
_alert_queue: queue.Queue = queue.Queue()


# ------------------------------------------------------------------ #
# The bot
# ------------------------------------------------------------------ #

class WebhookBot:
    """
    Receives entry signals from TradingView via HTTP webhook.
    Manages all position lifecycle (entry, ratchet, half-exit, stop, EOD)
    via IBKR TWS.

    Uses client ID 3 so it can run alongside the bar-scanning bot (ID 1).
    """

    WEBHOOK_CLIENT_ID = 3

    def __init__(self, plan: dict = None):
        """
        plan: {symbol: {'support': float|None, 'resistance': float|None}}
              loaded from plan.json at startup, updated via hot-reload if
              plan.json is modified while running.
        """
        self.plan = plan or {}
        self.ib   = IB()

        self.positions:       dict[str, Position] = {}
        self.tickers:         dict = {}   # symbol -> Ticker (real-time price)
        self.bars_3m:         dict = {}   # symbol -> BarDataList (ratchet EMAs)
        self.bars_10m:        dict = {}   # symbol -> BarDataList (exit trend)
        self.trade_log:       list = []
        self._trades_today:   dict = {}
        self._last_trade_date = None
        self._twss_stops:     dict = {}   # symbol -> crash-stop Order
        self._pending:        dict = {}   # symbol -> pending entry metadata

    # ----------------------------------------------------------------
    # Connection
    # ----------------------------------------------------------------

    def connect(self):
        self.ib.connect(TWS_HOST, TWS_PORT, clientId=self.WEBHOOK_CLIENT_ID)
        log.info("Webhook bot connected to IBKR TWS  port=%s  clientId=%s",
                 TWS_PORT, self.WEBHOOK_CLIENT_ID)

    # ----------------------------------------------------------------
    # Process one incoming TradingView alert
    # (called from the main IB loop -- safe to make IB calls here)
    # ----------------------------------------------------------------

    def process_alert(self, data: dict):
        now   = datetime.now()
        today = now.date()

        # Reset daily counters on new trading day
        if today != self._last_trade_date:
            self._trades_today    = {}
            self._last_trade_date = today
            tlog.info("=" * 55 + f"  {today}  [TV webhook bot]")

        # -- Parse payload ------------------------------------------------
        # TradingView sometimes prefixes ticker with exchange: "NASDAQ:MU" -> "MU"
        raw_sym = data.get("symbol", "").upper()
        symbol  = raw_sym.split(":")[-1].strip()

        action  = data.get("action", "").lower()       # "long" or "short"
        price   = float(data.get("price", 0))
        stop    = float(data.get("stop",  0))
        signal  = data.get("signal", "tv_alert")

        # Levels: alert can override plan (useful for intraday level updates)
        plan_entry = self.plan.get(symbol, {})
        sup = data.get("support")    or plan_entry.get("support")
        res = data.get("resistance") or plan_entry.get("resistance")
        if sup is not None: sup = float(sup)
        if res is not None: res = float(res)

        if not symbol or action not in ("long", "short") or price <= 0 or stop <= 0:
            log.warning("TV ALERT: invalid payload -- %s", data)
            return

        # -- Session time gates ------------------------------------------
        is_gap = (signal.startswith("open_cloud_break") or
                  signal.startswith("gap_"))

        # Outside session
        if (now.hour < MARKET_OPEN_HOUR or
                now.hour > MARKET_CLOSE_HOUR or
                (now.hour == MARKET_CLOSE_HOUR and now.minute >= MARKET_CLOSE_MINUTE)):
            log.info("SKIP  %s  outside session (%s)", symbol, now.strftime("%H:%M"))
            return

        # Pre-first-bar (before 09:33)
        if (now.hour == MARKET_OPEN_HOUR and
                now.minute < MARKET_OPEN_MINUTE):
            log.info("SKIP  %s  pre-open (%s)", symbol, now.strftime("%H:%M"))
            return

        # Early gate: normal entries wait until FIRST_ENTRY_MINUTE
        if not is_gap:
            if (now.hour == MARKET_OPEN_HOUR and
                    now.minute < FIRST_ENTRY_MINUTE):
                log.info("SKIP  %s  before 09:%02d gate", symbol, FIRST_ENTRY_MINUTE)
                return

        # Last entry
        if (now.hour > LAST_ENTRY_HOUR or
                (now.hour == LAST_ENTRY_HOUR and now.minute >= LAST_ENTRY_MINUTE)):
            log.info("SKIP  %s  after last-entry 15:%02d", symbol, LAST_ENTRY_MINUTE)
            return

        # -- Slot + daily limits -----------------------------------------
        active_slots = len(self.positions) + len(self._pending)
        if active_slots >= MAX_SIMULTANEOUS_POSITIONS:
            tlog.info("SKIP  %s  max positions (%d)", symbol, MAX_SIMULTANEOUS_POSITIONS)
            return

        if self._trades_today.get(symbol, 0) >= MAX_TRADES_PER_DAY:
            tlog.info("SKIP  %s  max trades/day (%d)", symbol, MAX_TRADES_PER_DAY)
            return

        if symbol in self.positions or symbol in self._pending:
            tlog.info("SKIP  %s  already open/pending", symbol)
            return

        # -- Share sizing + risk cap ------------------------------------
        n_full    = FIXED_SHARES_HIGH if price >= HIGH_PRICE_THRESHOLD else FIXED_SHARES
        n_starter = max(1, int(n_full * STARTER_RATIO))
        n_add     = n_full - n_starter
        stop_dist = abs(price - stop)
        risk      = stop_dist * n_starter
        risk_cap  = (MAX_RISK_DOLLARS_HIGH if price >= HIGH_PRICE_THRESHOLD
                     else MAX_RISK_DOLLARS)

        if risk > risk_cap:
            tlog.info("SKIP  %s  %s  risk=$%.0f > cap $%d  [%s]",
                      symbol, action.upper(), risk, risk_cap, signal)
            return

        # -- Level proximity gate (same as ibkr_client) -----------------
        is_cont = signal in ("cloud_cont", "cloud_cont_crash")
        if not is_gap and not is_cont:
            if action == "long" and res is not None:
                if price > res * (1 + LEVEL_PROX_LONG):
                    tlog.info("SKIP  %s  LONG chasing +%.1f%% above res $%.2f",
                              symbol, (price/res - 1)*100, res)
                    return
            if action == "short" and sup is not None:
                if price < sup * (1 - LEVEL_PROX_SHORT):
                    tlog.info("SKIP  %s  SHORT chasing -%.1f%% below sup $%.2f",
                              symbol, (1 - price/sup)*100, sup)
                    return

        # -- All checks passed -- place entry ----------------------------
        tlog.info(
            "TV ENTRY  %s  x%dsh(+%d add)  $%.2f  stop=$%.2f  "
            "dist=$%.2f  risk=$%.0f  [%s]  sup=%s  res=%s",
            f"{action.upper():<5}  {symbol}",
            n_starter, n_add, price, stop,
            stop_dist, risk, signal, sup, res,
        )
        self._open_position(
            symbol, action, price, stop, n_starter, now,
            shares_full=n_full, shares_add=n_add,
            entry_signal=signal, level_res=res, level_sup=sup,
        )

    # ----------------------------------------------------------------
    # Order helpers
    # ----------------------------------------------------------------

    def _entry_order(self, action: str, qty: int) -> Order:
        o = Order()
        o.action        = action
        o.totalQuantity = qty
        o.tif           = "DAY"
        # Midprice on live (saves spread), plain MKT on paper
        o.orderType = "MIDPRICE" if TWS_PORT == 7496 else "MKT"
        return o

    def _open_position(self, symbol, direction, entry_price, stop_price,
                       shares, time, shares_full=0, shares_add=0,
                       entry_signal="", level_res=None, level_sup=None):
        action   = "BUY" if direction == "long" else "SELL"
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        # Real-time ticker -- activate before placing order
        if symbol not in self.tickers:
            ticker = self.ib.reqMktData(contract, "", False, False)
            self.tickers[symbol] = ticker
            self.ib.sleep(1.0)   # let subscription register

        trade = self.ib.placeOrder(contract, self._entry_order(action, shares))
        log.info("  ORDER: %s %dsh starter  %s  ~$%.2f  (order #%d)",
                 action, shares, symbol, entry_price, trade.order.orderId)

        self._pending[symbol] = {
            "direction":    direction,
            "shares":       shares,
            "shares_full":  shares_full or shares,
            "shares_add":   shares_add,
            "entry_price":  entry_price,
            "stop_price":   stop_price,
            "time":         time,
            "entry_signal": entry_signal,
            "level_res":    level_res,
            "level_sup":    level_sup,
            "contract":     contract,
        }

        def _on_fill(t, fill):
            if symbol not in self._pending:
                return
            info      = self._pending.pop(symbol)
            actual_px = fill.execution.avgPrice
            pos = Position(
                symbol=symbol,
                direction=info["direction"],
                shares=info["shares"],
                entry_price=actual_px,
                entry_time=info["time"],
                stop_price=info["stop_price"],
                ibkr_order_id=t.order.orderId,
                entry_signal=info["entry_signal"],
                level_res=info["level_res"],
                level_sup=info["level_sup"],
                shares_full=info["shares_full"],
                shares_add=info["shares_add"],
            )
            self.positions[symbol] = pos
            tlog.info("FILLED  %s %dsh starter  %s  @$%.2f  (add %dsh when +$%.0f/sh)",
                      action, info["shares"], symbol, actual_px,
                      info["shares_add"], ADD_TRIGGER_PROFIT)
            # Load bars so ratchet + exits can run
            self._load_bars(symbol, info["contract"])
            self._place_crash_stop(symbol, info["contract"],
                                   info["direction"], info["shares"], info["stop_price"])

        def _on_cancelled(t):
            removed = self._pending.pop(symbol, None)
            if removed:
                log.warning("  %s: entry order cancelled (order #%d)",
                            symbol, t.order.orderId)

        trade.fillEvent      += _on_fill
        trade.cancelledEvent += _on_cancelled

    def _load_bars(self, symbol: str, contract):
        """Load live-streaming bars for a newly entered position.
        Used only for exit signal computation (ratchet, 10m trend exit).
        """
        try:
            b10 = self.ib.reqHistoricalData(
                contract, endDateTime="", durationStr="5 D",
                barSizeSetting=BAR_SIZE_10M, whatToShow="TRADES",
                useRTH=True, keepUpToDate=True,
            )
            b3 = self.ib.reqHistoricalData(
                contract, endDateTime="", durationStr="1 D",
                barSizeSetting=BAR_SIZE_3M, whatToShow="TRADES",
                useRTH=False, keepUpToDate=True,
            )
            self.bars_10m[symbol] = b10
            self.bars_3m[symbol]  = b3
            log.info("  %s: bars loaded (%d x 10m, %d x 3m)", symbol, len(b10), len(b3))
        except Exception as e:
            log.warning("  %s: bar load failed -- %s", symbol, e)

    def _place_crash_stop(self, symbol, contract, direction, shares, stop_price):
        """TWS crash-backstop STP order -- cancelled on any normal exit."""
        crash_action = "SELL" if direction == "long" else "BUY"
        o = Order()
        o.action = crash_action
        o.totalQuantity = shares
        o.orderType = "STP"
        o.auxPrice  = round(stop_price, 2)
        o.tif       = "DAY"
        try:
            ct = self.ib.placeOrder(contract, o)
            self._twss_stops[symbol] = ct.order
            log.info("  CRASH STOP: %s %dsh %s  STP@$%.2f  (order #%d)",
                     crash_action, shares, symbol, stop_price, ct.order.orderId)
        except Exception as e:
            log.warning("  %s: crash stop failed -- %s", symbol, e)

    def _rt_price(self, symbol: str):
        """Best available real-time last price."""
        t = self.tickers.get(symbol)
        if t is None:
            return None
        px = t.last
        if px and px == px:
            return px
        mid = t.midpoint()
        if mid and mid == mid:
            return mid
        return None

    @staticmethod
    def _closed_bars(bars) -> list:
        """Exclude the live (incomplete) bar from a keepUpToDate=True list."""
        lst = list(bars)
        return lst[:-1] if len(lst) > 1 else lst

    def _close_partial(self, symbol, shares, price, now, reason="half@level"):
        pos    = self.positions[symbol]
        action = "SELL" if pos.direction == "long" else "BUY"
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        self.ib.placeOrder(contract, self._entry_order(action, shares))
        pnl  = ((price - pos.entry_price) * shares if pos.direction == "long"
                else (pos.entry_price - price) * shares)
        sign = "+" if pnl >= 0 else ""
        tlog.info("HALF  %s  x%dsh  $%.2f->$%.2f  pnl=%s$%.0f  [%s]  remaining=%dsh",
                  f"{pos.direction.upper():<5}  {symbol}",
                  shares, pos.entry_price, price, sign, pnl, reason,
                  pos.shares - shares)
        self.trade_log.append({
            "event": reason, "symbol": symbol, "direction": pos.direction,
            "entry": pos.entry_price, "exit": price,
            "shares": shares, "pnl": pnl, "entry_signal": pos.entry_signal,
        })
        # Adjust crash stop quantity to match remaining shares
        remaining   = pos.shares - shares
        crash_stop  = self._twss_stops.get(symbol)
        if crash_stop is not None and remaining > 0:
            crash_stop.totalQuantity = remaining
            try:
                self.ib.placeOrder(contract, crash_stop)
            except Exception as e:
                log.warning("  %s: crash stop qty adjust failed -- %s", symbol, e)

    def _close_position(self, symbol, price, now, reason=""):
        pos = self.positions.get(symbol)
        if not pos:
            return
        action   = "SELL" if pos.direction == "long" else "BUY"
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        self.ib.placeOrder(contract, self._entry_order(action, pos.shares))
        pos.close(price, now, reason)
        pnl    = pos.pnl or 0.0
        sign   = "+" if pnl >= 0 else ""
        result = "WIN " if pnl > 0 else ("EVEN" if pnl == 0 else "LOSS")
        dur    = int((now - pos.entry_time).total_seconds() // 60)
        tlog.info("EXIT  %s  %s  x%dsh  $%.2f->$%.2f  pnl=%s$%.0f  [%s]  held=%dmin  signal=%s",
                  result, f"{pos.direction.upper():<5}  {symbol}",
                  pos.shares, pos.entry_price, price, sign, pnl, reason, dur, pos.entry_signal)
        self._trades_today[symbol] = self._trades_today.get(symbol, 0) + 1
        self.trade_log.append({
            "event": "exit", "symbol": symbol, "direction": pos.direction,
            "entry": pos.entry_price, "exit": price, "shares": pos.shares,
            "pnl": pnl, "reason": reason, "held_min": dur,
            "entry_signal": pos.entry_signal,
        })
        stp = self._twss_stops.pop(symbol, None)
        if stp:
            try:
                self.ib.cancelOrder(stp)
                log.info("  CRASH STOP CANCELLED: %s", symbol)
            except Exception as e:
                log.warning("  %s: crash stop cancel failed -- %s", symbol, e)
        del self.positions[symbol]
        self.bars_3m.pop(symbol, None)
        self.bars_10m.pop(symbol, None)

    # ----------------------------------------------------------------
    # Position management (called every second from main loop)
    # ----------------------------------------------------------------

    def manage_positions(self):
        now = datetime.now()

        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if not pos or not pos.is_open:
                continue

            # -- EOD forced close ----------------------------------------
            if (now.hour > MARKET_CLOSE_HOUR or
                    (now.hour == MARKET_CLOSE_HOUR
                     and now.minute >= MARKET_CLOSE_MINUTE)):
                rt = self._rt_price(symbol) or pos.entry_price
                tlog.warning("EOD  forced close  %s  %s  px=$%.2f",
                             pos.direction.upper(), symbol, rt)
                self._close_position(symbol, rt, now, "eod_close")
                continue

            rt = self._rt_price(symbol)
            if rt is None:
                continue

            # -- Ratchet stop update (from 3m bars) ----------------------
            bars_3m = self.bars_3m.get(symbol)
            if bars_3m and len(bars_3m) > 2:
                df_3m    = compute_emas(self._closed_bars(bars_3m))
                new_stop = compute_trailing_stop(
                    df_3m, pos.direction, pos.stop_price, pos.entry_price,
                    best_unrealised=pos.best_unrealised)
                pos.update_stop(new_stop)

            # -- Add-in trigger ------------------------------------------
            if not pos.add_triggered and pos.shares_add > 0:
                starter_profit = (pos.entry_price - rt if pos.direction == "short"
                                  else rt - pos.entry_price)
                if starter_profit >= ADD_TRIGGER_PROFIT:
                    pos.add_triggered = True
                    add_action   = "SELL" if pos.direction == "short" else "BUY"
                    add_contract = Stock(symbol, "SMART", "USD")
                    try:
                        self.ib.qualifyContracts(add_contract)
                        add_trade = self.ib.placeOrder(
                            add_contract,
                            self._entry_order(add_action, pos.shares_add))
                        tlog.info("ADD  %s  %s  x%dsh  trigger +$%.2f/sh  (order #%d)",
                                  symbol, add_action, pos.shares_add,
                                  starter_profit, add_trade.order.orderId)

                        def _on_add_fill(t, fill, _p=pos, _s=symbol, _ash=pos.shares_add):
                            add_px   = fill.execution.avgPrice
                            total_sh = _p.shares + _ash
                            avg_px   = (_p.entry_price * _p.shares + add_px * _ash) / total_sh
                            _p.add_entry_price = add_px
                            _p.entry_price     = avg_px
                            _p.shares          = total_sh
                            tlog.info("ADD_FILL  %s  @$%.2f  avg=$%.2f  total=%dsh",
                                      _s, add_px, avg_px, total_sh)

                        add_trade.fillEvent += _on_add_fill
                    except Exception as e:
                        log.warning("  %s: add-in order failed -- %s", symbol, e)
                        pos.add_triggered = False   # allow retry next second

            # -- Half-exit at Rip's level or profit target ----------------
            if not pos.half_exited:
                half_px  = None
                half_rsn = "half@level_rt"

                if pos.direction == "long" and pos.level_res and rt >= pos.level_res:
                    half_px = pos.level_res
                elif pos.direction == "short" and pos.level_sup and rt <= pos.level_sup:
                    half_px = pos.level_sup

                # Profit target fallback when no usable level is ahead of entry
                if half_px is None:
                    level_ahead = ((pos.level_res is not None
                                    and pos.level_res > pos.entry_price)
                                   if pos.direction == "long"
                                   else (pos.level_sup is not None
                                         and pos.level_sup < pos.entry_price))
                    unr = (rt - pos.entry_price if pos.direction == "long"
                           else pos.entry_price - rt)
                    if not level_ahead and unr >= PROFIT_TARGET_SHARE:
                        half_px  = rt
                        half_rsn = "half@target_rt"

                if half_px is not None:
                    half_sh  = pos.shares // 2
                    half_pnl = ((half_px - pos.entry_price) * half_sh
                                if pos.direction == "long"
                                else (pos.entry_price - half_px) * half_sh)
                    if half_pnl > 0 and half_sh > 0:
                        self._close_partial(symbol, half_sh, half_px, now, half_rsn)
                        pos.shares -= half_sh
                    pos.half_exited = True

            # -- HWM update (AFTER half-exit check) ----------------------
            unr = (rt - pos.entry_price if pos.direction == "long"
                   else pos.entry_price - rt)
            pos.best_unrealised = max(pos.best_unrealised, unr)

            # -- Hard stop (real-time price) -----------------------------
            if pos.direction == "long" and rt <= pos.stop_price:
                tlog.info("RT STOP  LONG   %s  last=$%.2f  stop=$%.2f  %s",
                          symbol, rt, pos.stop_price, now.strftime("%H:%M:%S"))
                self._close_position(symbol, pos.stop_price, now, "stop_rt")
            elif pos.direction == "short" and rt >= pos.stop_price:
                tlog.info("RT STOP  SHORT  %s  last=$%.2f  stop=$%.2f  %s",
                          symbol, rt, pos.stop_price, now.strftime("%H:%M:%S"))
                self._close_position(symbol, pos.stop_price, now, "stop_rt")

    # ----------------------------------------------------------------
    # Status display
    # ----------------------------------------------------------------

    def print_status(self):
        now = datetime.now()
        if not self.positions:
            return
        for symbol, pos in self.positions.items():
            rt   = self._rt_price(symbol)
            if rt is None:
                continue
            unr  = (rt - pos.entry_price if pos.direction == "long"
                    else pos.entry_price - rt) * pos.shares
            sign = "+" if unr >= 0 else ""
            away = (rt - pos.stop_price if pos.direction == "long"
                    else pos.stop_price - rt)
            locked = " [LOCKED]" if (
                (pos.direction == "long"  and pos.stop_price > pos.entry_price) or
                (pos.direction == "short" and pos.stop_price < pos.entry_price)
            ) else ""
            print(f"  {now.strftime('%H:%M:%S')}  {pos.direction.upper()} {symbol}"
                  f"  x{pos.shares}sh  entry=${pos.entry_price:.2f}  now=${rt:.2f}"
                  f"  PnL={sign}${unr:.0f}  stop=${pos.stop_price:.2f}"
                  f" ({away:.2f} away){locked}")

    # ----------------------------------------------------------------
    # Main event loop
    # ----------------------------------------------------------------

    def run(self):
        self.connect()
        tlog.info("WEBHOOK BOT LIVE  (listening for TradingView alerts on port 5000)")
        tlog.info("  max_pos=%d  risk_cap=$%d (<$%.0f) / $%d (>=$%.0f)",
                  MAX_SIMULTANEOUS_POSITIONS,
                  MAX_RISK_DOLLARS, HIGH_PRICE_THRESHOLD,
                  MAX_RISK_DOLLARS_HIGH, HIGH_PRICE_THRESHOLD)

        _status_timer = 0

        while True:
            self.ib.sleep(1)
            _status_timer += 1

            if not self.ib.isConnected():
                log.warning("TWS disconnected -- stopping.")
                break

            # -- Process incoming TradingView alerts ---------------------
            while not _alert_queue.empty():
                try:
                    alert = _alert_queue.get_nowait()
                    self.process_alert(alert)
                except queue.Empty:
                    break
                except Exception as e:
                    log.error("Alert processing error: %s", e, exc_info=True)

            # -- Position management -------------------------------------
            try:
                self.manage_positions()
            except Exception as e:
                log.error("Position management error: %s", e, exc_info=True)

            # -- Status print every 60 seconds when in a trade -----------
            if _status_timer >= 60:
                _status_timer = 0
                self.print_status()

        tlog.info("SESSION END")
        for t in self.trade_log:
            sign = "+" if t["pnl"] >= 0 else ""
            tag  = "H" if t["event"] != "exit" else ("W" if t["pnl"] > 0 else "L")
            print(f"  {tag}  {t['direction'].upper():<5} {t['symbol']:6s}  "
                  f"${t['entry']:.2f}->${t['exit']:.2f}  x{t['shares']}sh  "
                  f"{sign}${t['pnl']:.0f}  [{t['entry_signal']}]")


# ------------------------------------------------------------------ #
# Flask endpoints
# ------------------------------------------------------------------ #

@app.route("/webhook", methods=["POST"])
def webhook():
    """TradingView posts alerts here."""
    try:
        data = request.get_json(force=True) or {}
        if DEBUG_SIGNALS:
            log.info("TV ALERT IN: %s", json.dumps(data))
        _alert_queue.put(data)
        return jsonify({"status": "queued", "queue_size": _alert_queue.qsize()}), 200
    except Exception as e:
        log.error("Webhook parse error: %s", e)
        return jsonify({"status": "error", "msg": str(e)}), 400


@app.route("/status", methods=["GET"])
def status():
    """Browser health check -- visit https://xxx.ngrok.io/status"""
    b = bot
    return jsonify({
        "connected":    b.ib.isConnected() if b else False,
        "positions":    {
            s: {
                "direction":  p.direction,
                "entry":      p.entry_price,
                "stop":       p.stop_price,
                "shares":     p.shares,
                "signal":     p.entry_signal,
                "half_exited": p.half_exited,
                "add_triggered": p.add_triggered,
            }
            for s, p in b.positions.items()
        } if b else {},
        "pending":      list(b._pending.keys()) if b else [],
        "trades_today": b._trades_today if b else {},
        "queue_size":   _alert_queue.qsize(),
        "time":         datetime.now().strftime("%H:%M:%S"),
        "mode":         "LIVE" if TWS_PORT == 7496 else "PAPER",
    })


@app.route("/close/<symbol>", methods=["POST"])
def close_symbol(symbol):
    """Emergency manual close: POST /close/MU  (or click in browser with Postman)"""
    symbol = symbol.upper()
    if bot and symbol in bot.positions:
        rt = bot._rt_price(symbol) or bot.positions[symbol].entry_price
        bot._close_position(symbol, rt, datetime.now(), "manual_close")
        return jsonify({"status": "closed", "symbol": symbol, "price": rt})
    return jsonify({"status": "not_found", "symbol": symbol}), 404


@app.route("/plan", methods=["POST"])
def update_plan():
    """Hot-update levels mid-session:
       curl -X POST http://localhost:5000/plan -d '{"MU":{"support":950,"resistance":975}}'
    """
    data = request.get_json(force=True) or {}
    if bot:
        bot.plan.update(data)
        log.info("PLAN updated: %s", data)
        return jsonify({"status": "ok", "plan": bot.plan})
    return jsonify({"status": "error"}), 500


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

bot: Optional[WebhookBot] = None


def _run_flask():
    """Flask runs in a daemon thread; IB event loop stays on main thread."""
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    # Load today's plan from plan.json if it exists
    # Format: {"MU": {"support": 950, "resistance": 975}, "AVGO": {...}}
    plan = {}
    plan_path = pathlib.Path("plan.json")
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            log.info("Loaded plan: %s", plan)
        except Exception as e:
            log.warning("plan.json parse error: %s", e)

    bot = WebhookBot(plan=plan)

    # Start Flask in background thread
    flask_thread = threading.Thread(target=_run_flask, daemon=True)
    flask_thread.start()

    print()
    print("=" * 62)
    print("  WEBHOOK BOT  --  waiting for TradingView alerts")
    print("=" * 62)
    print("  Flask server running on  http://localhost:5000")
    print()
    print("  Next steps:")
    print("  1. Open a NEW terminal and run:  ngrok http 5000")
    print("  2. Copy the  https://xxx.ngrok-free.app  URL")
    print("  3. Paste it into TradingView alert > Webhook URL field")
    print("  4. Check health:  https://xxx.ngrok-free.app/status")
    print("=" * 62)
    print()

    try:
        bot.run()
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        if bot and bot.ib.isConnected():
            bot.ib.disconnect()
