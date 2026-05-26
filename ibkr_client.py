"""
ibkr_client.py — live trading bot connected to Interactive Brokers TWS.

All entry/exit logic mirrors week_backtest.py exactly:
  - Dynamic share sizing: 100 shares < $500, 50 shares >= $500
  - $700 max dollar risk cap per trade
  - 9:40 ET minimum entry time (via FIRST_ENTRY_MINUTE in config)
  - Rip's plan levels: support/resistance fed into entry signal
  - DTR/ATR exhaustion filter (skip when daily range >= 75% of ATR)
  - Level proximity filter (don't chase mid-range)
  - Ratchet trailing stop with intrabar-safe HWM
  - Half-exit at Rip's level (50% shares at target, rest runs with ratchet)
  - RVOL exit when momentum dries up (suppressed once ratchet locks profit)
  - 10-min fast cloud exit
  - 1 trade per symbol per direction per day after a loss
"""
import asyncio
import logging
import os
from datetime import datetime, date
from typing import Optional

import pandas as pd
from ib_insync import IB, Stock, MarketOrder

from config import (TWS_HOST, TWS_PORT, TWS_CLIENT_ID,
                    BAR_SIZE_10M, BAR_SIZE_3M,
                    MAX_TRADES_PER_DAY, MAX_SIMULTANEOUS_POSITIONS,
                    FIXED_SHARES, FIXED_SHARES_HIGH, HIGH_PRICE_THRESHOLD,
                    MAX_RISK_DOLLARS, LEVEL_PROX_LONG, LEVEL_PROX_SHORT,
                    DTR_MAX_PCT)
from ema_engine import (compute_emas, get_trend_10m,
                        get_entry_signal_3m, should_exit_10m,
                        should_exit_rvol, compute_trailing_stop,
                        compute_dtr_atr_ratio)
from position import Position

log = logging.getLogger(__name__)


# ── Trade event logger ─────────────────────────────────────────────────────
# Writes every ENTRY / SKIP / HALF / EXIT event to:
#   logs/trades_YYYY-MM-DD.log   (file, full detail)
#   console                      (same lines at INFO level)

def _setup_trade_logger(log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"trades_{date.today().strftime('%Y-%m-%d')}.log")
    tlog = logging.getLogger("trade_log")
    tlog.setLevel(logging.DEBUG)
    if not tlog.handlers:
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s",
                                          datefmt="%H:%M:%S"))
        tlog.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s  %(message)s",
                                          datefmt="%H:%M:%S"))
        tlog.addHandler(ch)
    return tlog


tlog = _setup_trade_logger()


class TradingBot:
    def __init__(self, symbols: list[str], plan: dict):
        """
        symbols : list of tickers, e.g. ['TSLA', 'NVDA', 'AMD']
        plan    : {symbol: {'support': float|None, 'resistance': float|None}}
                  Missing symbols default to rules-only (no level filter).
        """
        self.symbols = [s.upper() for s in symbols]
        self.plan    = plan          # Rip's levels for today's session

        self.ib        = IB()
        self.bars_10m: dict = {}     # symbol -> BarDataList  (trend direction)
        self.bars_3m:  dict = {}     # symbol -> BarDataList  (entry + management)
        self.trend:    dict = {}     # symbol -> 'bullish' | 'bearish' | 'none'

        self.positions: dict[str, Position] = {}
        self.trade_log: list[dict]          = []   # all completed trade events

        self._trades_today:   dict = {s: 0    for s in self.symbols}
        self._lost_dir_today: dict = {s: None for s in self.symbols}
        self._last_trade_date      = None

        # Pre-market high / low per symbol — used for PMH/PML breakout signals
        self.pmh: dict = {s: None for s in self.symbols}
        self.pml: dict = {s: None for s in self.symbols}

    # ──────────────────────────────────────────────────────────────────────
    # Connection
    # ──────────────────────────────────────────────────────────────────────

    async def connect(self):
        await self.ib.connectAsync(TWS_HOST, TWS_PORT, clientId=TWS_CLIENT_ID)
        log.info("Connected to IBKR TWS  (port %s)", TWS_PORT)

    def disconnect(self):
        self.ib.disconnect()
        log.info("Disconnected from IBKR TWS")

    # ──────────────────────────────────────────────────────────────────────
    # Bar subscriptions
    # ──────────────────────────────────────────────────────────────────────

    async def subscribe_bars(self):
        for symbol in self.symbols:
            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)
            self.trend[symbol] = "none"

            b10 = self.ib.reqHistoricalData(
                contract, endDateTime="", durationStr="3 D",
                barSizeSetting=BAR_SIZE_10M, whatToShow="TRADES",
                useRTH=True, keepUpToDate=True,
            )
            self.bars_10m[symbol] = b10
            b10.updateEvent += self._make_handler(symbol, "10m")

            b3 = self.ib.reqHistoricalData(
                contract, endDateTime="", durationStr="3 D",
                barSizeSetting=BAR_SIZE_3M, whatToShow="TRADES",
                useRTH=True, keepUpToDate=True,
            )
            self.bars_3m[symbol] = b3
            b3.updateEvent += self._make_handler(symbol, "3m")

            log.info(f"  {symbol}: {len(b10)} x 10-min  |  {len(b3)} x 3-min bars loaded")

    def _make_handler(self, symbol: str, tf: str):
        def on_bar(bars, has_new_bar):
            if has_new_bar:
                if tf == "10m":
                    self._on_new_bar_10m(symbol, bars)
                else:
                    self._on_new_bar_3m(symbol, bars)
        return on_bar

    # ──────────────────────────────────────────────────────────────────────
    # Pre-market levels  (call once before 09:30 ET each morning)
    # ──────────────────────────────────────────────────────────────────────

    def setup_premarket_levels(self):
        """Pull 1-min pre-market bars from IBKR and compute PMH/PML per symbol."""
        for symbol in self.symbols:
            try:
                contract = Stock(symbol, "SMART", "USD")
                self.ib.qualifyContracts(contract)
                bars = self.ib.reqHistoricalData(
                    contract, endDateTime="", durationStr="1 D",
                    barSizeSetting="1 min", whatToShow="TRADES",
                    useRTH=False, keepUpToDate=False,
                )
                if not bars:
                    log.warning(f"{symbol}: no pre-market bars returned")
                    continue

                df = pd.DataFrame({"time": [b.date for b in bars],
                                   "high": [b.high for b in bars],
                                   "low":  [b.low  for b in bars]})
                df["time"] = pd.to_datetime(df["time"])
                if df["time"].dt.tz is None:
                    df["time"] = df["time"].dt.tz_localize("US/Eastern")
                else:
                    df["time"] = df["time"].dt.tz_convert("US/Eastern")
                df = df.set_index("time")
                pre = df.between_time("04:00", "09:29")
                if pre.empty:
                    log.warning(f"{symbol}: empty pre-market slice")
                    continue

                self.pmh[symbol] = float(pre["high"].max())
                self.pml[symbol] = float(pre["low"].min())
                log.info(f"  {symbol}  PM high=${self.pmh[symbol]:.2f}"
                         f"  PM low=${self.pml[symbol]:.2f}"
                         f"  range=${self.pmh[symbol] - self.pml[symbol]:.2f}")
            except Exception as e:
                log.warning(f"{symbol}: pre-market error — {e}")

    # ──────────────────────────────────────────────────────────────────────
    # 10-min bar handler — trend direction only
    # ──────────────────────────────────────────────────────────────────────

    def _on_new_bar_10m(self, symbol: str, bars):
        df   = compute_emas(list(bars))
        prev = self.trend[symbol]
        self.trend[symbol] = get_trend_10m(df)
        if self.trend[symbol] != prev:
            log.info(f"  {symbol}  10m trend: {prev} -> {self.trend[symbol]}")

    # ──────────────────────────────────────────────────────────────────────
    # 3-min bar handler — entry + full position management
    # ──────────────────────────────────────────────────────────────────────

    def _on_new_bar_3m(self, symbol: str, bars):
        df_3m  = compute_emas(list(bars))
        df_10m = compute_emas(list(self.bars_10m[symbol]))
        now    = datetime.now()
        cur    = df_3m.iloc[-1]
        today  = now.date()

        # Reset daily counters on new trading day
        if today != self._last_trade_date:
            self._trades_today   = {s: 0    for s in self.symbols}
            self._lost_dir_today = {s: None for s in self.symbols}
            self._last_trade_date = today
            tlog.info("=" * 55 + f"  {today}")

        # ─────────────────────────────────────────────────────────────────
        # MANAGE OPEN POSITION
        # ─────────────────────────────────────────────────────────────────
        pos = self.positions.get(symbol)
        if pos and pos.is_open:

            # Compute unrealised P&L BEFORE updating the ratchet stop.
            # HWM is updated AFTER the stop check — prevents the intrabar phantom:
            # the same bar that raises the ratchet floor cannot immediately hit it.
            unrealised = (cur.close - pos.entry_price if pos.direction == "long"
                          else pos.entry_price - cur.close)

            new_stop = compute_trailing_stop(
                df_3m, pos.direction, pos.stop_price, pos.entry_price,
                best_unrealised=pos.best_unrealised)
            pos.update_stop(new_stop)

            # Live P&L printed every bar while in the trade
            self._print_live_pnl(symbol, cur, now)

            # ── Half-exit at Rip's level ──────────────────────────────────
            # Exit 50% of shares when price reaches the target level.
            # Locks partial profit; remaining shares run with the ratchet stop.
            # Fires at most once per position (half_exited flag).
            if not pos.half_exited:
                half_sh = pos.shares // 2
                half_px = None
                if (pos.direction == "long" and pos.level_res is not None
                        and cur.high >= pos.level_res):
                    half_px = pos.level_res
                elif (pos.direction == "short" and pos.level_sup is not None
                        and cur.low <= pos.level_sup):
                    half_px = pos.level_sup

                if half_px is not None and half_sh > 0:
                    half_pnl = ((half_px - pos.entry_price) * half_sh
                                if pos.direction == "long"
                                else (pos.entry_price - half_px) * half_sh)
                    if half_pnl > 0:
                        self._close_partial(symbol, half_sh, half_px, now)
                        pos.shares -= half_sh
                pos.half_exited = True   # prevent re-firing even if pnl was 0

            # Update HWM after stop check (intrabar-safe)
            pos.best_unrealised = max(pos.best_unrealised, unrealised)

            # ── Hard stop hit ─────────────────────────────────────────────
            if pos.direction == "long" and cur.low <= pos.stop_price:
                self._close_position(symbol, pos.stop_price, now, "stop")
                return
            if pos.direction == "short" and cur.high >= pos.stop_price:
                self._close_position(symbol, pos.stop_price, now, "stop")
                return

            # ── 10-min fast cloud exit ────────────────────────────────────
            # Higher-timeframe momentum reversal — more reliable than 3m exit
            if should_exit_10m(df_10m, pos.direction):
                self._close_position(symbol, cur.close, now, "10m exit")
                return

            # ── RVOL exit — momentum dried up ────────────────────────────
            # Suppress once the ratchet has locked in profit (let stop manage it)
            stop_locked = (pos.stop_price > pos.entry_price if pos.direction == "long"
                           else pos.stop_price < pos.entry_price)
            if should_exit_rvol(df_3m) and not stop_locked:
                self._close_position(symbol, cur.close, now, "low rvol")
            return

        # ─────────────────────────────────────────────────────────────────
        # ENTRY CHECKS
        # ─────────────────────────────────────────────────────────────────
        if self._trades_today[symbol] >= MAX_TRADES_PER_DAY:
            return
        if len(self.positions) >= MAX_SIMULTANEOUS_POSITIONS:
            return

        # DTR/ATR exhaustion gate — skip when daily range is >= 75% of ATR
        dtr_ratio = compute_dtr_atr_ratio(df_10m, today, bar_time=now)
        if dtr_ratio > DTR_MAX_PCT:
            return

        # Rip's levels for this symbol (None = rules-only, no filter applied)
        plan_entry = self.plan.get(symbol, {})
        sup = plan_entry.get("support")
        res = plan_entry.get("resistance")

        trend = self.trend.get(symbol, "none")
        signal, stop_price, entry_reason = get_entry_signal_3m(
            df_3m, trend, bar_time=now,
            pmh=self.pmh.get(symbol), pml=self.pml.get(symbol),
            support=sup, resistance=res)

        if signal == "none":
            return
        if signal == self._lost_dir_today.get(symbol):
            return   # blocked after same-direction loss today

        entry_price = cur.close
        stop_dist   = abs(entry_price - stop_price)

        # Share sizing: 100 shares below $500, 50 shares at $500+
        n    = FIXED_SHARES_HIGH if entry_price >= HIGH_PRICE_THRESHOLD else FIXED_SHARES
        risk = stop_dist * n

        # Hard dollar risk cap — skip if stop is too wide
        if risk > MAX_RISK_DOLLARS:
            tlog.info(
                f"SKIP  {symbol}  {signal.upper():<5}  ${entry_price:.2f}  "
                f"stop=${stop_price:.2f}  risk=${risk:.0f} > cap ${MAX_RISK_DOLLARS}  "
                f"[{entry_reason}]"
            )
            return

        # Level proximity gate — only enter AT Rip's level, not mid-range
        if signal == "long" and res is not None:
            if entry_price > res * (1 + LEVEL_PROX_LONG):
                tlog.info(
                    f"SKIP  {symbol}  LONG   ${entry_price:.2f}  "
                    f"chasing +{(entry_price/res - 1)*100:.1f}% above res ${res:.2f}"
                )
                return
        if signal == "short" and sup is not None:
            if entry_price < sup * (1 - LEVEL_PROX_SHORT):
                tlog.info(
                    f"SKIP  {symbol}  SHORT  ${entry_price:.2f}  "
                    f"chasing -{(1 - entry_price/sup)*100:.1f}% below sup ${sup:.2f}"
                )
                return

        # ── All checks passed — enter the trade ──────────────────────────
        slot = len(self.positions) + 1
        tlog.info(
            f"ENTRY  [{slot}/{MAX_SIMULTANEOUS_POSITIONS}]  "
            f"{signal.upper():<5}  {symbol}  x{n}sh  "
            f"${entry_price:.2f}  stop=${stop_price:.2f}  "
            f"dist=${stop_dist:.2f} ({stop_dist/entry_price*100:.2f}%)  "
            f"risk=${risk:.0f}  [{entry_reason}]  "
            f"trend={trend}  sup={sup}  res={res}"
        )
        self._open_position(symbol, signal, entry_price, stop_price, n, now,
                            entry_reason=entry_reason, level_res=res, level_sup=sup)

    # ──────────────────────────────────────────────────────────────────────
    # Order + position helpers
    # ──────────────────────────────────────────────────────────────────────

    def _print_live_pnl(self, symbol: str, cur, now: datetime):
        """Compact one-line P&L status printed every 3-min bar."""
        p          = self.positions[symbol]
        price      = cur.close
        unrealised = ((price - p.entry_price) if p.direction == "long"
                      else (p.entry_price - price)) * p.shares
        stop_away  = ((price - p.stop_price) if p.direction == "long"
                      else (p.stop_price - price))
        locked     = (" [LOCKED]"
                      if (p.direction == "long"  and p.stop_price > p.entry_price) or
                         (p.direction == "short" and p.stop_price < p.entry_price)
                      else "")
        sign = "+" if unrealised >= 0 else ""
        print(
            f"  {now.strftime('%H:%M')}  {p.direction.upper()} {p.symbol}"
            f"  x{p.shares}sh  entry=${p.entry_price:.2f}  now=${price:.2f}"
            f"  PnL={sign}${unrealised:.0f}"
            f"  stop=${p.stop_price:.2f} ({stop_away:.2f} away)"
            f"{locked}"
        )

    def _open_position(self, symbol: str, direction: str,
                       entry_price: float, stop_price: float,
                       shares: int, time: datetime,
                       entry_reason: str = "",
                       level_res: Optional[float] = None,
                       level_sup: Optional[float] = None):
        action   = "BUY" if direction == "long" else "SELL"
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        order = self.ib.placeOrder(contract, MarketOrder(action, shares))

        pos = Position(
            symbol=symbol, direction=direction, shares=shares,
            entry_price=entry_price, entry_time=time, stop_price=stop_price,
            ibkr_order_id=order.order.orderId,
            entry_signal=entry_reason,
            level_res=level_res, level_sup=level_sup,
        )
        self.positions[symbol] = pos
        log.info(f"  ORDER: {action} {shares}x {symbol}  "
                 f"(order #{order.order.orderId}  ~${entry_price:.2f})")

    def _close_partial(self, symbol: str, shares: int,
                       price: float, time: datetime):
        """Place a partial exit order (half-exit at Rip's level)."""
        pos    = self.positions[symbol]
        action = "SELL" if pos.direction == "long" else "BUY"
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        self.ib.placeOrder(contract, MarketOrder(action, shares))

        half_pnl = ((price - pos.entry_price) * shares if pos.direction == "long"
                    else (pos.entry_price - price) * shares)
        sign = "+" if half_pnl >= 0 else ""
        tlog.info(
            f"HALF   {pos.direction.upper():<5}  {symbol}  x{shares}sh  "
            f"${pos.entry_price:.2f}->${price:.2f}  pnl={sign}${half_pnl:.0f}  "
            f"[half@level]  remaining={pos.shares - shares}sh"
        )
        self.trade_log.append({
            "time":         time.strftime("%H:%M:%S"),
            "event":        "half@level",
            "symbol":       symbol,
            "direction":    pos.direction,
            "entry":        pos.entry_price,
            "exit":         price,
            "shares":       shares,
            "pnl":          half_pnl,
            "reason":       "half@level",
            "entry_signal": pos.entry_signal,
        })

    def _close_position(self, symbol: str, price: float,
                        time: datetime, reason: str = ""):
        pos = self.positions.get(symbol)
        if not pos:
            return

        action   = "SELL" if pos.direction == "long" else "BUY"
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        self.ib.placeOrder(contract, MarketOrder(action, pos.shares))

        pos.close(price, time, reason)
        pnl    = pos.pnl or 0.0
        sign   = "+" if pnl >= 0 else ""
        result = "WIN " if pnl > 0 else ("EVEN" if pnl == 0 else "LOSS")
        dur    = int((time - pos.entry_time).total_seconds() // 60)

        tlog.info(
            f"EXIT   {result}  {pos.direction.upper():<5}  {symbol}  x{pos.shares}sh  "
            f"${pos.entry_price:.2f}->${price:.2f}  pnl={sign}${pnl:.0f}  "
            f"[{reason}]  held={dur}min  signal={pos.entry_signal}"
        )

        self._trades_today[symbol] += 1
        if pnl < 0:
            self._lost_dir_today[symbol] = pos.direction
            tlog.info(
                f"  BLOCK  {symbol}  {pos.direction.upper()} re-entries today (loss)"
            )

        self.trade_log.append({
            "time":         time.strftime("%H:%M:%S"),
            "event":        "exit",
            "symbol":       symbol,
            "direction":    pos.direction,
            "entry":        pos.entry_price,
            "exit":         price,
            "shares":       pos.shares,
            "pnl":          pnl,
            "reason":       reason,
            "entry_signal": pos.entry_signal,
            "held_min":     dur,
        })
        del self.positions[symbol]

    # ──────────────────────────────────────────────────────────────────────
    # Run
    # ──────────────────────────────────────────────────────────────────────

    async def run(self):
        await self.connect()
        await self.subscribe_bars()

        tlog.info("BOT LIVE  symbols=" + ", ".join(self.symbols))
        for sym in self.symbols:
            p = self.plan.get(sym, {})
            tlog.info(f"  {sym:6s}  sup={p.get('support')}  res={p.get('resistance')}")

        await asyncio.sleep(float("inf"))

    # ──────────────────────────────────────────────────────────────────────
    # Session summary
    # ──────────────────────────────────────────────────────────────────────

    def print_session_summary(self):
        exits    = [t for t in self.trade_log if t["event"] == "exit"]
        partials = [t for t in self.trade_log if t["event"] == "half@level"]
        total    = sum(t["pnl"] for t in self.trade_log)
        wins     = sum(1 for t in exits if t["pnl"] > 0)
        losses   = sum(1 for t in exits if t["pnl"] <= 0)

        print("\n" + "=" * 58)
        print("  SESSION SUMMARY")
        print("=" * 58)
        if not self.trade_log:
            print("  No trades today.")
        for t in self.trade_log:
            sign = "+" if t["pnl"] >= 0 else ""
            if t["event"] == "half@level":
                tag = "H"
            elif t["pnl"] > 0:
                tag = "W"
            else:
                tag = "L"
            dur = f"  {t.get('held_min', '?')}min" if "held_min" in t else ""
            print(f"  {tag}  {t['direction'].upper():<5} {t['symbol']:6s}  "
                  f"${t['entry']:.2f}->${t['exit']:.2f}  x{t['shares']}sh  "
                  f"{sign}${t['pnl']:.0f}  [{t['reason']}]{dur}"
                  f"  ({t['entry_signal']})")
        print(f"  {'-' * 52}")
        print(f"  {len(exits)} trades  {len(partials)} partial exits  "
              f"{wins}W / {losses}L  TOTAL: ${total:+.0f}")
        print("=" * 58 + "\n")
        tlog.info(f"SESSION END  {len(exits)} trades  {wins}W/{losses}L  "
                  f"total=${total:+.0f}")
