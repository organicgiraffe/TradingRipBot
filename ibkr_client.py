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
import logging
import os
from datetime import datetime, date
from typing import Optional

import pandas as pd
from ib_insync import IB, Stock, MarketOrder, Order

from config import (TWS_HOST, TWS_PORT, TWS_CLIENT_ID,
                    BAR_SIZE_10M, BAR_SIZE_3M,
                    MAX_SIMULTANEOUS_POSITIONS,
                    FIXED_SHARES, FIXED_SHARES_HIGH, HIGH_PRICE_THRESHOLD,
                    MAX_RISK_DOLLARS, MAX_RISK_DOLLARS_HIGH, MIN_DAILY_RANGE,
                    MAX_TRADES_PER_DAY,
                    LEVEL_PROX_LONG, LEVEL_PROX_SHORT,
                    DTR_MAX_PCT, DTR_EXEMPT_ATR, FIRST_ENTRY_MINUTE, MARKET_OPEN_HOUR,
                    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
                    VOLUME_CONFIRM_MULT, DEBUG_SIGNALS,
                    PROFIT_TARGET_SHARE)
from ema_engine import (compute_emas, get_trend_10m,
                        get_entry_signal_3m, get_gap_signal_3m,
                        should_exit_10m,
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


def _entry_order(action: str, quantity: int) -> Order:
    """Entry order — Midprice on live (fills at bid/ask midpoint, saves spread),
    plain MarketOrder on paper (Midprice not needed in simulation).
    Exits always use MarketOrder so they are guaranteed to fill."""
    if TWS_PORT == 7496:   # live account
        o = Order()
        o.action        = action
        o.totalQuantity = quantity
        o.orderType     = "MIDPRICE"
        o.tif           = "DAY"
        return o
    else:                  # paper account (7497) — market order
        return MarketOrder(action, quantity)


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
        self._sym_atr: dict = {}     # symbol -> 5-day avg daily range (for DTR exemption)
        self.trend:    dict = {}     # symbol -> 'bullish' | 'bearish' | 'none'

        self.positions: dict[str, Position] = {}
        self.trade_log: list[dict]          = []   # all completed trade events

        self._trades_today:   dict = {s: 0    for s in self.symbols}
        self._lost_dir_today: dict = {s: None for s in self.symbols}
        self._last_trade_date      = None

        # Protective STP orders placed in TWS at entry — crash backstop.
        # If the bot crashes mid-trade, TWS closes the position at the initial stop.
        # Cancelled automatically when bot exits the position normally.
        self._twss_stop_orders: dict = {}  # symbol -> ib_insync Order object

        # Pre-market high / low per symbol — used for PMH/PML breakout signals
        self.pmh: dict = {s: None for s in self.symbols}
        self.pml: dict = {s: None for s in self.symbols}

    # ──────────────────────────────────────────────────────────────────────
    # Connection
    # ──────────────────────────────────────────────────────────────────────

    def connect(self):
        self.ib.connect(TWS_HOST, TWS_PORT, clientId=TWS_CLIENT_ID)
        log.info("Connected to IBKR TWS  (port %s)", TWS_PORT)

    def disconnect(self):
        self.ib.disconnect()
        log.info("Disconnected from IBKR TWS")

    # ──────────────────────────────────────────────────────────────────────
    # Bar subscriptions
    # ──────────────────────────────────────────────────────────────────────

    def subscribe_bars(self):
        """Initial historical data load (keepUpToDate=False).
        Live updates come from _refresh_bars() called on a schedule in run().
        keepUpToDate=True is unreliable on Windows with ib.sleep() — bars freeze.
        """
        for symbol in self.symbols:
            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)
            self.trend[symbol] = "none"

            b10 = self.ib.reqHistoricalData(
                contract, endDateTime="", durationStr="20 D",
                barSizeSetting=BAR_SIZE_10M, whatToShow="TRADES",
                useRTH=True, keepUpToDate=False,
            )

            # ATR check: compute 5-day avg daily range from 10m bars
            # Each 10m bar has date; group by day, sum (max-min), average last 5 days
            if b10:
                import collections, datetime as _dt
                day_hi: dict = collections.defaultdict(float)
                day_lo: dict = collections.defaultdict(lambda: float("inf"))
                for bar in b10:
                    d = bar.date.date() if hasattr(bar.date, "date") else bar.date
                    day_hi[d] = max(day_hi[d], bar.high)
                    day_lo[d] = min(day_lo[d], bar.low)
                recent_days = sorted(day_hi)[-5:]
                avg_range = sum(day_hi[d] - day_lo[d] for d in recent_days) / len(recent_days)
                if avg_range < MIN_DAILY_RANGE:
                    log.info(
                        f"  {symbol}: SKIPPED — 5-day ATR ${avg_range:.2f} < min ${MIN_DAILY_RANGE:.0f}"
                    )
                    self.symbols.remove(symbol)
                    continue
                self._sym_atr[symbol] = avg_range
                dtr_tag = " (DTR exempt — momentum stock)" if avg_range >= DTR_EXEMPT_ATR else ""
                log.info(f"  {symbol}: 5-day ATR ${avg_range:.2f}  (min=${MIN_DAILY_RANGE:.0f}) OK{dtr_tag}")

            self.bars_10m[symbol] = b10
            self._on_new_bar_10m(symbol, b10)   # set initial trend

            b3 = self.ib.reqHistoricalData(
                contract, endDateTime="", durationStr="3 D",
                barSizeSetting=BAR_SIZE_3M, whatToShow="TRADES",
                useRTH=True, keepUpToDate=False,
            )
            self.bars_3m[symbol] = b3

            log.info(f"  {symbol}: {len(b10)} x 10-min  |  {len(b3)} x 3-min bars loaded  trend={self.trend[symbol]}")

    # ──────────────────────────────────────────────────────────────────────
    # Bar refresh — re-request fresh data at each bar boundary
    # ──────────────────────────────────────────────────────────────────────

    def _refresh_bars(self, refresh_10m: bool = False):
        """Re-request historical bars for all symbols.
        Called every 3-min bar close for 3m bars, and every 10-min bar close
        for 10m bars.  Each request is a fresh snapshot (keepUpToDate=False).
        """
        for symbol in self.symbols:
            try:
                contract = Stock(symbol, "SMART", "USD")
                self.ib.qualifyContracts(contract)

                # ── 3-min bars ───────────────────────────────────────────
                new_b3 = self.ib.reqHistoricalData(
                    contract, endDateTime="", durationStr="3 D",
                    barSizeSetting=BAR_SIZE_3M, whatToShow="TRADES",
                    useRTH=True, keepUpToDate=False,
                )
                if new_b3:
                    old_date = self.bars_3m[symbol][-1].date if self.bars_3m.get(symbol) else None
                    self.bars_3m[symbol] = new_b3
                    if new_b3[-1].date != old_date:
                        log.info(f"  {symbol}: new 3m bar at {new_b3[-1].date}  close={new_b3[-1].close:.2f}")
                        self._on_new_bar_3m(symbol, new_b3)

                # ── 10-min bars (only when requested) ────────────────────
                if refresh_10m:
                    new_b10 = self.ib.reqHistoricalData(
                        contract, endDateTime="", durationStr="20 D",
                        barSizeSetting=BAR_SIZE_10M, whatToShow="TRADES",
                        useRTH=True, keepUpToDate=False,
                    )
                    if new_b10:
                        self.bars_10m[symbol] = new_b10
                        self._on_new_bar_10m(symbol, new_b10)

            except Exception as e:
                log.warning(f"  {symbol}: bar refresh error — {e}")

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
        print(f"  BAR {symbol} {len(bars)} bars", flush=True)
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

            # ── End-of-day forced close ───────────────────────────────────
            # Close any open position at or after MARKET_CLOSE_MINUTE.
            # MARKET_CLOSE_MINUTE = 50 → forces exit by 15:50 ET.
            # Change to 59 in config if you want to hold until 15:59.
            # This fires on the first 3m bar that closes AT or AFTER that minute.
            if (now.hour > MARKET_CLOSE_HOUR or
                    (now.hour == MARKET_CLOSE_HOUR
                     and now.minute >= MARKET_CLOSE_MINUTE)):
                tlog.warning(
                    f"EOD    forced close  {symbol}  {now.strftime('%H:%M')}"
                    f"  px=${cur.close:.2f}"
                )
                self._close_position(symbol, cur.close, now, "eod_close")
                return

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

            # ── Half-exit: Rip's level  OR  flat profit target ────────────
            # Takes 50% of shares off when the first of these triggers fires:
            #   1. Rip's level (resistance for longs, support for shorts)
            #      — checked every bar; level may not be hit until bars later.
            #   2. Profit target (+$5/share) when no Rip level is configured
            #      — rules-only fallback so every trade has a defined exit point.
            # Once either trigger fires, half_exited=True prevents re-firing.
            # Remaining shares are managed by the ratchet trailing stop only.
            if not pos.half_exited:
                half_sh  = pos.shares // 2
                half_px  = None
                half_rsn = "half@level"

                # Priority 1: Rip's key level — re-evaluated every bar
                if (pos.direction == "long" and pos.level_res is not None
                        and cur.high >= pos.level_res):
                    half_px = pos.level_res
                elif (pos.direction == "short" and pos.level_sup is not None
                        and cur.low <= pos.level_sup):
                    half_px = pos.level_sup

                # Priority 2: Flat profit target — when no usable level exists.
                # "No usable level" means either:
                #   a) No level configured (rules-only), OR
                #   b) Level is already behind us at entry (e.g. entry above resistance
                #      on a gap-up — the half-exit at that level would be at a loss).
                if pos.direction == "long":
                    level_still_ahead = (pos.level_res is not None
                                         and pos.level_res > pos.entry_price)
                else:
                    level_still_ahead = (pos.level_sup is not None
                                         and pos.level_sup < pos.entry_price)
                if half_px is None and not level_still_ahead and unrealised >= PROFIT_TARGET_SHARE:
                    half_px  = cur.close
                    half_rsn = "half@target"

                if half_px is not None and half_sh > 0:
                    half_pnl = ((half_px - pos.entry_price) * half_sh
                                if pos.direction == "long"
                                else (pos.entry_price - half_px) * half_sh)
                    if half_pnl > 0:
                        self._close_partial(symbol, half_sh, half_px, now,
                                            reason=half_rsn)
                        pos.shares -= half_sh
                    pos.half_exited = True  # don't fire again (even if pnl was 0)

            # Update HWM after the half-exit check (intrabar-safe)
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
            # Require BOTH low volume AND C2 flipped against position.
            # Low volume alone doesn't mean the move is over — if C2 still
            # aligned (ema5 above ema12 for longs), stay in; let ratchet manage.
            # Suppress entirely once ratchet has locked profit.
            stop_locked = (pos.stop_price > pos.entry_price if pos.direction == "long"
                           else pos.stop_price < pos.entry_price)
            c2_against  = (cur.ema5 < cur.ema12 if pos.direction == "long"
                           else cur.ema5 > cur.ema12)
            if should_exit_rvol(df_3m) and not stop_locked and c2_against:
                self._close_position(symbol, cur.close, now, "low rvol+C2")
            return

        # ─────────────────────────────────────────────────────────────────
        # ENTRY CHECKS
        # ─────────────────────────────────────────────────────────────────
        if len(self.positions) >= MAX_SIMULTANEOUS_POSITIONS:
            return
        if self._trades_today.get(symbol, 0) >= MAX_TRADES_PER_DAY:
            return

        # Rip's levels for this symbol (None = rules-only, no filter applied)
        plan_entry = self.plan.get(symbol, {})
        sup = plan_entry.get("support")
        res = plan_entry.get("resistance")

        trend = self.trend.get(symbol, "none")

        gap_signal, gap_stop, gap_reason = get_gap_signal_3m(
            df_3m, bar_time=now, pmh=self.pmh.get(symbol),
            support=sup, resistance=res)

        # Time gate — normal cloud/curl entries wait until FIRST_ENTRY_MINUTE,
        # but opening-drive gap entries may fire from 09:33 through 10:00.
        if gap_signal == "none":
            if now.hour == MARKET_OPEN_HOUR and now.minute < FIRST_ENTRY_MINUTE:
                return

        # DTR/ATR exhaustion gate — skip when daily range is >= 75% of ATR
        # Exempt high-ATR momentum stocks (≥ DTR_EXEMPT_ATR): they regularly exceed
        # their average on breakout days — blocking them misses the best trades.
        sym_atr_5d = getattr(self, "_sym_atr", {}).get(symbol, 0.0)
        dtr_ratio  = compute_dtr_atr_ratio(df_10m, today, bar_time=now)
        dtr_exempt = sym_atr_5d >= DTR_EXEMPT_ATR
        if not dtr_exempt and dtr_ratio > DTR_MAX_PCT:
            if DEBUG_SIGNALS:
                print(f"  {now.strftime('%H:%M')}  {symbol:6s}  SKIP: DTR {dtr_ratio:.0%} of ATR  (ATR=${sym_atr_5d:.0f})")
            return

        # ── Debug: cloud state + key values every bar ────────────────────
        if DEBUG_SIGNALS:
            _c = df_3m.iloc[-1]
            _p = df_3m.iloc[-2] if len(df_3m) >= 2 else _c
            _flip_l = _p.ema5 <= _p.ema12 and _c.ema5 > _c.ema12
            _flip_s = _p.ema5 >= _p.ema12 and _c.ema5 < _c.ema12
            _c2 = "GRN" if _c.ema5 > _c.ema12 else "RED"
            _c3 = "GRN" if _c.ema34 > _c.ema50 else "RED"
            _vol_ok = _c.vol_ma20 <= 0 or _c.volume >= VOLUME_CONFIRM_MULT * _c.vol_ma20
            _tag = " <<FLIP!" if (_flip_l or _flip_s) else ""
            print(
                f"  {now.strftime('%H:%M')}  {symbol:6s}"
                f"  C2:{_c2} C3:{_c3}{_tag}"
                f"  trend={trend:8s}"
                f"  vol={'OK ' if _vol_ok else 'LOW'}"
                f"  DTR={dtr_ratio:.0%}"
                f"  e5={_c.ema5:.2f} e12={_c.ema12:.2f} e50={_c.ema50:.2f}"
                f"  px={_c.close:.2f}"
            )

        if gap_signal != "none":
            signal, stop_price, entry_reason = gap_signal, gap_stop, gap_reason
        else:
            signal, stop_price, entry_reason = get_entry_signal_3m(
                df_3m, trend, bar_time=now,
                pmh=self.pmh.get(symbol), pml=self.pml.get(symbol),
                support=sup, resistance=res)

        if signal == "none":
            # ── Explain what blocked a flip (only interesting bars) ───────
            if DEBUG_SIGNALS:
                _c = df_3m.iloc[-1]
                _p = df_3m.iloc[-2] if len(df_3m) >= 2 else _c
                _flip_l = _p.ema5 <= _p.ema12 and _c.ema5 > _c.ema12
                _flip_s = _p.ema5 >= _p.ema12 and _c.ema5 < _c.ema12
                if _flip_l or _flip_s:
                    _dir = "LONG" if _flip_l else "SHORT"
                    # Collect ALL reasons — multiple can block simultaneously
                    _reasons = []
                    # 10m direction confirmed and opposes the flip
                    if trend == "bearish" and _flip_l:
                        _reasons.append("10m=bearish")
                    elif trend == "bullish" and _flip_s:
                        _reasons.append("10m=bullish")
                    # C3 cloud direction must match signal direction
                    # (when trend=none, this is the only bias filter)
                    if _flip_l and _c.ema34 < _c.ema50:
                        _reasons.append("C3 red (need green for long)")
                    if _flip_s and _c.ema34 > _c.ema50:
                        _reasons.append("C3 green (need red for short)")
                    # Volume confirmation
                    if _c.vol_ma20 > 0 and _c.volume < VOLUME_CONFIRM_MULT * _c.vol_ma20:
                        _reasons.append(f"vol {_c.volume:.0f}<{VOLUME_CONFIRM_MULT * _c.vol_ma20:.0f}")
                    if not _reasons:
                        _reasons.append("stop/level filter")
                    tlog.info(
                        f"BLOCKED  {symbol}  {_dir}  [{' | '.join(_reasons)}]"
                        f"  px=${_c.close:.2f}  e50=${_c.ema50:.2f}"
                    )
            return
        if signal == self._lost_dir_today.get(symbol):
            if DEBUG_SIGNALS:
                tlog.info(f"BLOCKED  {symbol}  {signal.upper()}  [lost same direction today]")
            return   # blocked after same-direction loss today

        entry_price = cur.close
        stop_dist   = abs(entry_price - stop_price)

        # Share sizing: 100 shares below $500, 50 shares at $500+
        n    = FIXED_SHARES_HIGH if entry_price >= HIGH_PRICE_THRESHOLD else FIXED_SHARES
        risk = stop_dist * n

        # Hard dollar risk cap — use higher cap for expensive stocks ($500+)
        risk_cap = MAX_RISK_DOLLARS_HIGH if entry_price >= HIGH_PRICE_THRESHOLD else MAX_RISK_DOLLARS
        if risk > risk_cap:
            tlog.info(
                f"SKIP  {symbol}  {signal.upper():<5}  ${entry_price:.2f}  "
                f"stop=${stop_price:.2f}  risk=${risk:.0f} > cap ${risk_cap}  "
                f"[{entry_reason}]"
            )
            return

        # Level proximity gate — only enter AT Rip's level, not mid-range
        is_gap_entry = entry_reason.startswith("gap_")
        if not is_gap_entry and signal == "long" and res is not None:
            if entry_price > res * (1 + LEVEL_PROX_LONG):
                tlog.info(
                    f"SKIP  {symbol}  LONG   ${entry_price:.2f}  "
                    f"chasing +{(entry_price/res - 1)*100:.1f}% above res ${res:.2f}"
                )
                return
        if not is_gap_entry and signal == "short" and sup is not None:
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
        order = self.ib.placeOrder(contract, _entry_order(action, shares))

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

        # Place a protective STP order in TWS as a crash backstop.
        # If the bot crashes while in a trade, TWS closes the position at
        # the initial stop — preventing an unmanaged open position overnight.
        # The bot cancels this order automatically on any normal exit.
        _crash_action = "SELL" if direction == "long" else "BUY"
        _stop_ord = Order()
        _stop_ord.action        = _crash_action
        _stop_ord.totalQuantity = shares
        _stop_ord.orderType     = "STP"
        _stop_ord.auxPrice      = round(stop_price, 2)
        _stop_ord.tif           = "DAY"
        try:
            _stop_trade = self.ib.placeOrder(contract, _stop_ord)
            self._twss_stop_orders[symbol] = _stop_trade.order
            log.info(f"  CRASH STOP: {_crash_action} {shares}x {symbol}  "
                     f"STP@${stop_price:.2f}  (order #{_stop_trade.order.orderId})")
        except Exception as _e:
            log.warning(f"  {symbol}: crash stop placement failed — {_e}")

    def _close_partial(self, symbol: str, shares: int,
                       price: float, time: datetime,
                       reason: str = "half@level"):
        """Place a partial exit order (half at Rip's level or profit target)."""
        pos    = self.positions[symbol]
        action = "SELL" if pos.direction == "long" else "BUY"
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        self.ib.placeOrder(contract, MarketOrder(action, shares))

        # Keep crash-stop quantity in sync with remaining shares.
        # Without this, the crash stop would try to sell the full original
        # quantity if the bot crashed after a half-exit — creating a short.
        _remaining = pos.shares - shares
        _crash_stp = self._twss_stop_orders.get(symbol)
        if _crash_stp is not None and _remaining > 0:
            _crash_stp.totalQuantity = _remaining
            try:
                self.ib.placeOrder(contract, _crash_stp)   # sends modify to TWS
                log.info(f"  CRASH STOP adjusted: {symbol}  qty={_remaining}sh")
            except Exception as _e:
                log.warning(f"  {symbol}: crash stop qty adjust failed — {_e}")

        half_pnl = ((price - pos.entry_price) * shares if pos.direction == "long"
                    else (pos.entry_price - price) * shares)
        sign = "+" if half_pnl >= 0 else ""
        tlog.info(
            f"HALF   {pos.direction.upper():<5}  {symbol}  x{shares}sh  "
            f"${pos.entry_price:.2f}->${price:.2f}  pnl={sign}${half_pnl:.0f}  "
            f"[{reason}]  remaining={_remaining}sh"
        )
        self.trade_log.append({
            "time":         time.strftime("%H:%M:%S"),
            "event":        reason,   # "half@level" or "half@target"
            "symbol":       symbol,
            "direction":    pos.direction,
            "entry":        pos.entry_price,
            "exit":         price,
            "shares":       shares,
            "pnl":          half_pnl,
            "reason":       reason,
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
        # Only block re-entries after a meaningful loss — scratches ($50 or less)
        # are often premature exits, not wrong direction calls.
        if pnl < -50:
            self._lost_dir_today[symbol] = pos.direction
            tlog.info(
                f"  BLOCK  {symbol}  {pos.direction.upper()} re-entries today (loss ${pnl:.0f})"
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

        # Cancel the TWS crash-backstop stop order.
        # Position is now flat — the standing STP order would try to sell
        # into nothing if price drifts down to it.  Cancel it cleanly.
        _twss = self._twss_stop_orders.pop(symbol, None)
        if _twss is not None:
            try:
                self.ib.cancelOrder(_twss)
                log.info(f"  CRASH STOP CANCELLED: {symbol}  (order #{_twss.orderId})")
            except Exception as _e:
                log.warning(f"  {symbol}: crash stop cancel failed — {_e}")

        del self.positions[symbol]

    # ──────────────────────────────────────────────────────────────────────
    # Run
    # ──────────────────────────────────────────────────────────────────────

    def run(self):
        self.connect()
        self.subscribe_bars()
        self.setup_premarket_levels()

        tlog.info("BOT LIVE  symbols=" + ", ".join(self.symbols))
        tlog.info(f"  entry gate: 09:{FIRST_ENTRY_MINUTE:02d} ET | "
                  f"max_pos={MAX_SIMULTANEOUS_POSITIONS} | "
                  f"risk_cap=${MAX_RISK_DOLLARS} (<${HIGH_PRICE_THRESHOLD:.0f}) "
                  f"/ ${MAX_RISK_DOLLARS_HIGH} (>=${HIGH_PRICE_THRESHOLD:.0f}) | "
                  f"DTR_max={DTR_MAX_PCT:.0%} | "
                  f"debug={'ON' if DEBUG_SIGNALS else 'OFF'}")
        for sym in self.symbols:
            p = self.plan.get(sym, {})
            tlog.info(f"  {sym:6s}  sup={p.get('support')}  res={p.get('resistance')}")

        # Scheduled refresh loop.
        # keepUpToDate=True freezes on Windows with ib.sleep() — instead we
        # re-request fresh bars at each bar-close boundary:
        #   3-min bars : minute divisible by 3  (9:33, 9:36, ..., 15:57)
        #   10-min bars: minute divisible by 10 (9:40, 9:50, 10:00, ...)
        last_3m_id  = -1   # bar-close ID already processed for 3m
        last_10m_id = -1   # bar-close ID already processed for 10m

        while True:
            self.ib.sleep(1)

            if not self.ib.isConnected():
                log.warning("TWS disconnected — stopping.")
                break

            now = datetime.now()

            # ── 3-min bar close detector ──────────────────────────────────
            # Bars close when clock-minute is divisible by 3, within first 5s
            bar_3m_id = now.hour * 100 + now.minute
            if now.minute % 3 == 0 and now.second <= 5 and bar_3m_id != last_3m_id:
                last_3m_id = bar_3m_id
                log.info(f"  Bar close {now.strftime('%H:%M')} — refreshing bars")
                self._refresh_bars(refresh_10m=True)   # always update trend

    # ──────────────────────────────────────────────────────────────────────
    # Session summary
    # ──────────────────────────────────────────────────────────────────────

    def print_session_summary(self):
        exits    = [t for t in self.trade_log if t["event"] == "exit"]
        partials = [t for t in self.trade_log if t["event"] in ("half@level", "half@target")]
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
            if t["event"] in ("half@level", "half@target"):
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
