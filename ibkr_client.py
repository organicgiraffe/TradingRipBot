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
import collections
import json
import logging
import os
import pathlib
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd
from ib_insync import IB, Stock, MarketOrder, Order

from config import (TWS_HOST, TWS_PORT, TWS_CLIENT_ID,
                    BAR_SIZE_10M, BAR_SIZE_3M,
                    MAX_SIMULTANEOUS_POSITIONS,
                    FIXED_SHARES, FIXED_SHARES_HIGH, HIGH_PRICE_THRESHOLD,
                    STARTER_RATIO, ADD_TRIGGER_PROFIT,
                    MAX_RISK_DOLLARS, MAX_RISK_DOLLARS_HIGH, MIN_DAILY_RANGE,
                    MAX_TRADES_PER_DAY,
                    LEVEL_PROX_LONG, LEVEL_PROX_SHORT,
                    DTR_MAX_PCT, DTR_EXEMPT_ATR, FIRST_ENTRY_MINUTE, MARKET_OPEN_HOUR,
                    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
                    GAP_ENTRY_START_HOUR, GAP_ENTRY_START_MINUTE,
                    GAP_ENTRY_END_HOUR,   GAP_ENTRY_END_MINUTE,
                    VOLUME_CONFIRM_MULT, DEBUG_SIGNALS,
                    PROFIT_TARGET_SHARE,
                    PAPER_DATA_DELAY_MINUTES)
from ema_engine import (compute_emas, get_trend_10m,
                        get_entry_signal_3m, get_gap_signal_3m,
                        get_open_cloud_break_signal_3m,
                        should_exit_10m,
                        should_exit_rvol, compute_trailing_stop,
                        compute_dtr_atr_ratio)
from position import Position

log = logging.getLogger(__name__)


# -- Trade event logger -----------------------------------------------------
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
    """All orders use MKT for guaranteed fills on both paper and live accounts.
    tif='DAY' set explicitly so the TWS order preset never needs to override it
    (error 10349 — preset TIF override cancels the order)."""
    o = Order()
    o.action        = action
    o.totalQuantity = quantity
    o.orderType     = "MKT"
    o.tif           = "DAY"
    return o


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

        self._trades_today:   dict = {s: 0 for s in self.symbols}
        self._last_trade_date      = None

        # Protective STP orders placed in TWS at entry — crash backstop.
        # If the bot crashes mid-trade, TWS closes the position at the initial stop.
        # Cancelled automatically when bot exits the position normally.
        self._twss_stop_orders: dict = {}  # symbol -> ib_insync Order object

        # Orders placed but not yet fill-confirmed.  Position object is created
        # inside the fillEvent callback — never before — so a cancelled entry
        # never creates a ghost position.  Also counted toward the slot limit.
        self._pending_entries: dict = {}   # symbol -> metadata dict

        # Real-time market data tickers — one per symbol, requested at startup.
        # (a) Activates market data subscription → IBKR error 354 goes away.
        # (b) Unlocks non-delayed reqHistoricalData — paper accounts serve
        #     15-min delayed bars without an active subscription; with one, bars
        #     come in real-time.
        # (c) Gives us a live last-price feed for second-by-second stop checks.
        self.tickers: dict = {}            # symbol -> ib_insync Ticker

        # Pre-market high / low per symbol — used for PMH/PML breakout signals
        self.pmh: dict = {s: None for s in self.symbols}
        self.pml: dict = {s: None for s in self.symbols}

        # Previous-day close price — used for gap % computation at startup.
        self._sym_prev_close: dict = {}   # symbol -> float

        # Symbols that received a bar update since the last main-loop tick.
        # Callbacks are kept lightweight (just set.add) to avoid reentrancy
        # deadlocks that happen when reqHistoricalData is called inside a callback.
        self._3m_update_set:  set = set()
        self._10m_update_set: set = set()

    # ----------------------------------------------------------------------
    # Connection
    # ----------------------------------------------------------------------

    def connect(self):
        self.ib.connect(TWS_HOST, TWS_PORT, clientId=TWS_CLIENT_ID)
        log.info("Connected to IBKR TWS  (port %s)", TWS_PORT)

    def disconnect(self):
        # Cancel all streaming market data subscriptions before disconnecting.
        for symbol in self.symbols:
            if symbol in self.tickers:
                try:
                    contract = Stock(symbol, "SMART", "USD")
                    self.ib.cancelMktData(contract)
                except Exception:
                    pass
        self.ib.disconnect()
        log.info("Disconnected from IBKR TWS")

    # ----------------------------------------------------------------------
    # Bar subscriptions
    # ----------------------------------------------------------------------

    def subscribe_bars(self):
        """Historical data load + live bar streaming (keepUpToDate=True).
        Each bar list auto-updates via updateEvent; completed-bar notifications
        are queued in _3m_update_set / _10m_update_set and processed in the
        main loop — no heavy work inside callbacks, no reentrancy risk.
        """
        # Iterate over a COPY of self.symbols.  The ATR filter below may call
        # self.symbols.remove() to drop a symbol — mutating the list during
        # iteration silently skips the next symbol.  Copying the list avoids
        # the footgun.
        for symbol in list(self.symbols):
            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)
            self.trend[symbol] = "none"

            # Subscribe to real-time market data BEFORE requesting historical bars.
            # With an active subscription, IBKR serves non-delayed historical data
            # on paper accounts.  Also satisfies the error-354 precautionary check
            # so orders go through without "no market data" rejection.
            # genericTickList='' = default tick types (bid/ask/last/volume/close)
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.tickers[symbol] = ticker
            self.ib.sleep(2.0)   # 2 s minimum: let subscription register before hist requests

            try:
                b10 = self.ib.reqHistoricalData(
                    contract, endDateTime="", durationStr="20 D",
                    barSizeSetting=BAR_SIZE_10M, whatToShow="TRADES",
                    useRTH=False, keepUpToDate=True,
                )
            except Exception as _e:
                log.warning(f"  {symbol}: 10m historical request failed — {_e}")
                b10 = []

            if not b10:
                log.warning(f"  {symbol}: NO 10m bars returned — dropping from watchlist")
                if symbol in self.symbols:
                    self.symbols.remove(symbol)
                continue

            # ATR check: compute 5-day avg daily range from 10m bars
            # Each 10m bar has date; group by day, sum (max-min), average last 5 days
            if b10:
                import collections, datetime as _dt
                day_hi: dict = collections.defaultdict(float)
                day_lo: dict = collections.defaultdict(lambda: float("inf"))
                for bar in self._closed_bars(b10):
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

                # Previous-day close — last bar strictly before today
                today_d = date.today()
                for bar in reversed(self._closed_bars(b10)):
                    d = bar.date.date() if hasattr(bar.date, "date") else bar.date
                    if d < today_d:
                        self._sym_prev_close[symbol] = bar.close
                        break

            self.bars_10m[symbol] = b10
            self._on_new_bar_10m(symbol, b10)   # set initial trend

            try:
                b3 = self.ib.reqHistoricalData(
                    contract, endDateTime="", durationStr="3 D",
                    barSizeSetting=BAR_SIZE_3M, whatToShow="TRADES",
                    useRTH=False, keepUpToDate=True,
                )
            except Exception as _e:
                log.warning(f"  {symbol}: 3m historical request failed — {_e}")
                b3 = []
            if not b3:
                log.warning(f"  {symbol}: NO 3m bars returned — dropping from watchlist")
                if symbol in self.symbols:
                    self.symbols.remove(symbol)
                continue
            self.bars_3m[symbol] = b3

            # Wire live-bar callbacks — lightweight queue additions only.
            # The default-arg trick (sym=symbol) captures the loop variable correctly.
            def _on_3m_update(bars, hasNewBar, sym=symbol):
                if hasNewBar:
                    self._3m_update_set.add(sym)

            def _on_10m_update(bars, hasNewBar, sym=symbol):
                if hasNewBar:
                    self._10m_update_set.add(sym)

            b3.updateEvent  += _on_3m_update
            b10.updateEvent += _on_10m_update

            log.info(f"  {symbol}: {len(b10)} x 10-min  |  {len(b3)} x 3-min bars loaded  trend={self.trend[symbol]}")

    # ----------------------------------------------------------------------
    # keepUpToDate=True helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _closed_bars(bars) -> list:
        """With keepUpToDate=True, bars[-1] is always the live (incomplete) bar
        being updated in real-time.  Return a plain list excluding it so that
        signal functions always operate on fully-closed bars only.
        Safe to call with keepUpToDate=False lists too — worst case we lose
        one completed bar, which is harmless given 200+ bars of history."""
        lst = list(bars)
        return lst[:-1] if len(lst) > 1 else lst

    # ----------------------------------------------------------------------
    # 1-min position management — ratchet + stop every minute while in trade
    # ----------------------------------------------------------------------

    def _refresh_bars_1m(self):
        """Re-request 1-min bars for open positions and run management checks.
        Called every minute when a position is live.  No entry signals here.
        """
        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if not pos or not pos.is_open:
                continue
            try:
                contract = Stock(symbol, "SMART", "USD")
                self.ib.qualifyContracts(contract)
                bars_1m = self.ib.reqHistoricalData(
                    contract, endDateTime="", durationStr="1 D",
                    barSizeSetting="1 min", whatToShow="TRADES",
                    useRTH=True, keepUpToDate=False,
                )
                if bars_1m:
                    self._on_new_bar_1m(symbol, bars_1m)
            except Exception as e:
                log.warning(f"  {symbol}: 1m management refresh error — {e}")

    def _on_new_bar_1m(self, symbol: str, bars_1m):
        """1-min position management: ratchet update + stop check.
        Does NOT check entry signals, RVOL, or cloud direction — those stay
        at the 3-min bar so they're based on confirmed bar closes.

        Ordering (anti-phantom guarantee):
          1. Update trailing stop with PREVIOUS best_unrealised
          2. Check if stop was hit
          3. Update best_unrealised with this bar's extreme — AFTER the stop check
             so the same bar that raises the ratchet floor can't immediately hit it.
        """
        pos = self.positions.get(symbol)
        if not pos or not pos.is_open:
            return

        now    = datetime.now()
        cur_1m = bars_1m[-1]   # most recent 1-min bar — raw bar object

        # ema50 from the last 3-min computation (close enough for the ratchet
        # floor — recomputing EMAs on 1-min bars every minute is unnecessary).
        df_3m = compute_emas(self._closed_bars(self.bars_3m[symbol]))

        # Step 1 — Update trailing stop using PREVIOUS best_unrealised
        new_stop = compute_trailing_stop(
            df_3m, pos.direction, pos.stop_price, pos.entry_price,
            best_unrealised=pos.best_unrealised)
        pos.update_stop(new_stop)

        # Step 2 — Half-exit at Rip's level or profit target (1-min resolution)
        # Mirrors the logic in _on_new_bar_3m but fires on 1-min bar highs/lows
        # so we don't wait 3 minutes to take profit at a level that was touched.
        if not pos.half_exited:
            half_px  = None
            half_rsn = "half@level"

            if pos.direction == "long" and pos.level_res is not None:
                if cur_1m.high >= pos.level_res:
                    half_px = pos.level_res
            elif pos.direction == "short" and pos.level_sup is not None:
                if cur_1m.low <= pos.level_sup:
                    half_px = pos.level_sup

            # Profit target fallback when no usable level is ahead of entry
            if pos.direction == "long":
                level_still_ahead = (pos.level_res is not None
                                     and pos.level_res > pos.entry_price)
            else:
                level_still_ahead = (pos.level_sup is not None
                                     and pos.level_sup < pos.entry_price)
            unr_1m = (cur_1m.close - pos.entry_price if pos.direction == "long"
                      else pos.entry_price - cur_1m.close)
            if half_px is None and not level_still_ahead and unr_1m >= PROFIT_TARGET_SHARE:
                half_px  = cur_1m.close
                half_rsn = "half@target"

            if half_px is not None:
                half_sh  = pos.shares // 2
                half_pnl = ((half_px - pos.entry_price) * half_sh
                            if pos.direction == "long"
                            else (pos.entry_price - half_px) * half_sh)
                if half_pnl > 0 and half_sh > 0:
                    self._close_partial(symbol, half_sh, half_px, now,
                                        reason=half_rsn)
                    pos.shares -= half_sh
                pos.half_exited = True   # block re-fire even if pnl was 0

        # Step 3 — Hard stop check using 1-min low/high
        if pos.direction == "long" and cur_1m.low <= pos.stop_price:
            tlog.info(
                f"1M STOP  LONG  {symbol}  stop=${pos.stop_price:.2f}"
                f"  (1m low=${cur_1m.low:.2f})  {now.strftime('%H:%M')}"
            )
            self._close_position(symbol, pos.stop_price, now, "stop")
            return
        if pos.direction == "short" and cur_1m.high >= pos.stop_price:
            tlog.info(
                f"1M STOP  SHORT {symbol}  stop=${pos.stop_price:.2f}"
                f"  (1m high=${cur_1m.high:.2f})  {now.strftime('%H:%M')}"
            )
            self._close_position(symbol, pos.stop_price, now, "stop")
            return

        # Step 3 — Update HWM with this bar's extreme (AFTER stop check)
        if pos.direction == "long":
            peak_unr = cur_1m.high - pos.entry_price
        else:
            peak_unr = pos.entry_price - cur_1m.low
        pos.best_unrealised = max(pos.best_unrealised, peak_unr)

        # Compact 1-min status line
        unr_px  = cur_1m.close - pos.entry_price if pos.direction == "long" \
                  else pos.entry_price - cur_1m.close
        sign    = "+" if unr_px >= 0 else ""
        locked  = (" [LOCKED]"
                   if (pos.direction == "long"  and pos.stop_price > pos.entry_price) or
                      (pos.direction == "short" and pos.stop_price < pos.entry_price)
                   else "")
        print(
            f"  {now.strftime('%H:%M')}  1m {pos.direction.upper()} {symbol}"
            f"  px={cur_1m.close:.2f}"
            f"  PnL={sign}${unr_px * pos.shares:.0f}"
            f"  stop={pos.stop_price:.2f}"
            f"  HWM=+${pos.best_unrealised:.2f}{locked}"
        )

    # ----------------------------------------------------------------------
    # Real-time price helper
    # ----------------------------------------------------------------------

    def _rt_price(self, symbol: str):
        """Best available real-time last price for a symbol.
        Returns None when the ticker hasn't received data yet (e.g. pre-market
        before the first trade).  Falls back to bid/ask midpoint."""
        t = self.tickers.get(symbol)
        if t is None:
            return None
        px = t.last
        if px and px == px:   # not None, not NaN (NaN != NaN is True)
            return px
        mid = t.midpoint()
        if mid and mid == mid:
            return mid
        return None

    # ----------------------------------------------------------------------
    # Bar refresh — re-request fresh data at each bar boundary
    # ----------------------------------------------------------------------

    def _refresh_bars(self, refresh_10m: bool = False):
        """Re-request historical bars for all symbols.
        Called every 3-min bar close for 3m bars, and every 10-min bar close
        for 10m bars.  Each request is a fresh snapshot (keepUpToDate=False).
        """
        for symbol in self.symbols:
            try:
                contract = Stock(symbol, "SMART", "USD")
                self.ib.qualifyContracts(contract)

                # -- 3-min bars -------------------------------------------
                new_b3 = self.ib.reqHistoricalData(
                    contract, endDateTime="", durationStr="3 D",
                    barSizeSetting=BAR_SIZE_3M, whatToShow="TRADES",
                    useRTH=False, keepUpToDate=False,
                )
                if new_b3:
                    old_date = self.bars_3m[symbol][-1].date if self.bars_3m.get(symbol) else None
                    self.bars_3m[symbol] = new_b3
                    if new_b3[-1].date != old_date:
                        log.info(f"  {symbol}: new 3m bar at {new_b3[-1].date}  close={new_b3[-1].close:.2f}")
                        self._on_new_bar_3m(symbol, new_b3)

                # -- 10-min bars (only when requested) --------------------
                if refresh_10m:
                    new_b10 = self.ib.reqHistoricalData(
                        contract, endDateTime="", durationStr="20 D",
                        barSizeSetting=BAR_SIZE_10M, whatToShow="TRADES",
                        useRTH=False, keepUpToDate=False,
                    )
                    if new_b10:
                        self.bars_10m[symbol] = new_b10
                        self._on_new_bar_10m(symbol, new_b10)

            except Exception as e:
                log.warning(f"  {symbol}: bar refresh error — {e}")

    # ----------------------------------------------------------------------
    # Pre-market levels  (call once before 09:30 ET each morning)
    # ----------------------------------------------------------------------

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

    # ----------------------------------------------------------------------
    # 10-min bar handler — trend direction only
    # ----------------------------------------------------------------------

    def _on_new_bar_10m(self, symbol: str, bars):
        df   = compute_emas(self._closed_bars(bars))
        prev = self.trend[symbol]
        self.trend[symbol] = get_trend_10m(df)
        if self.trend[symbol] != prev:
            log.info(f"  {symbol}  10m trend: {prev} -> {self.trend[symbol]}")

    # ----------------------------------------------------------------------
    # 3-min bar handler — entry + full position management
    # ----------------------------------------------------------------------

    def _on_new_bar_3m(self, symbol: str, bars):
        closed_3m = self._closed_bars(bars)
        # Guard: skip if we already processed this exact bar for this symbol.
        # The fallback polling can call this multiple times per bar-close window;
        # without the guard the same entry signal fires on every 3-min tick until
        # a new bar arrives.
        if not hasattr(self, "_last_bar_ts_3m"):
            self._last_bar_ts_3m = {}
        _last_bar = closed_3m[-1].date if closed_3m else None
        if _last_bar is not None and self._last_bar_ts_3m.get(symbol) == _last_bar:
            return   # same bar — already processed, skip
        self._last_bar_ts_3m[symbol] = _last_bar
        print(f"  BAR {symbol} {len(closed_3m)} bars", flush=True)
        df_3m  = compute_emas(closed_3m)
        df_10m = compute_emas(self._closed_bars(self.bars_10m[symbol]))
        now    = datetime.now()

        # Paper accounts serve data 15 minutes late.  Shift the "clock" we use
        # for entry-gate and signal-window checks so they align with what the
        # bars are actually showing.  Exit checks (EOD close, stop) use real
        # now so we never overstay on a delayed read.
        effective_now = (now - timedelta(minutes=PAPER_DATA_DELAY_MINUTES)
                         if TWS_PORT != 7496 else now)

        cur    = df_3m.iloc[-1]
        today  = now.date()

        # Reset daily counters on new trading day
        if today != self._last_trade_date:
            self._trades_today    = {s: 0 for s in self.symbols}
            self._last_trade_date = today
            tlog.info("=" * 55 + f"  {today}")

        # -----------------------------------------------------------------
        # MANAGE OPEN POSITION
        # -----------------------------------------------------------------
        pos = self.positions.get(symbol)
        if pos and pos.is_open:

            # -- End-of-day forced close -----------------------------------
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

            # -- Half-exit: Rip's level  OR  flat profit target ------------
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

            # -- Hard stop hit ---------------------------------------------
            if pos.direction == "long" and cur.low <= pos.stop_price:
                self._close_position(symbol, pos.stop_price, now, "stop")
                return
            if pos.direction == "short" and cur.high >= pos.stop_price:
                self._close_position(symbol, pos.stop_price, now, "stop")
                return

            # -- 10-min fast cloud exit ------------------------------------
            # Higher-timeframe momentum reversal — more reliable than 3m exit
            if should_exit_10m(df_10m, pos.direction):
                self._close_position(symbol, cur.close, now, "10m exit")
                return

            # -- RVOL exit — momentum dried up ----------------------------
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

        # -----------------------------------------------------------------
        # ENTRY CHECKS
        # -----------------------------------------------------------------
        # Count pending (order placed, awaiting fill) toward the slot limit
        # so we don't fire a second entry while the first is still in-flight.
        if len(self.positions) + len(self._pending_entries) >= MAX_SIMULTANEOUS_POSITIONS:
            return
        if self._trades_today.get(symbol, 0) >= MAX_TRADES_PER_DAY:
            return

        # Rip's levels for this symbol (None = rules-only, no filter applied)
        plan_entry = self.plan.get(symbol, {})
        sup = plan_entry.get("support")
        res = plan_entry.get("resistance")

        trend = self.trend.get(symbol, "none")

        gap_signal, gap_stop, gap_reason = get_open_cloud_break_signal_3m(
            df_3m, bar_time=effective_now)
        if gap_signal == "none":
            gap_signal, gap_stop, gap_reason = get_gap_signal_3m(
                df_3m, bar_time=effective_now, pmh=self.pmh.get(symbol),
                support=sup, resistance=res)

        # Time gate — normal cloud/curl entries wait until FIRST_ENTRY_MINUTE.
        # Gap/cloud-break entries and cloud_cont_crash fire from 09:33.
        # Use effective_now so paper-account delay doesn't shift the gate.
        if gap_signal == "none":
            if (effective_now.hour == MARKET_OPEN_HOUR
                    and effective_now.minute < FIRST_ENTRY_MINUTE):
                # Peek at the signal; only crash starters bypass the early gate
                _early_sig, _, _early_rsn = get_entry_signal_3m(
                    df_3m, trend, bar_time=effective_now, support=sup, resistance=res)
                if _early_rsn != "cloud_cont_crash":
                    return   # non-crash signal: honour the 09:40 gate

        # DTR/ATR exhaustion gate — skip when daily range is >= 75% of ATR
        # Exempt high-ATR momentum stocks (≥ DTR_EXEMPT_ATR): they regularly exceed
        # their average on breakout days — blocking them misses the best trades.
        sym_atr_5d = getattr(self, "_sym_atr", {}).get(symbol, 0.0)
        dtr_ratio  = compute_dtr_atr_ratio(df_10m, today, bar_time=effective_now)
        dtr_exempt = sym_atr_5d >= DTR_EXEMPT_ATR
        if not dtr_exempt and dtr_ratio > DTR_MAX_PCT:
            if DEBUG_SIGNALS:
                print(f"  {now.strftime('%H:%M')}  {symbol:6s}  SKIP: DTR {dtr_ratio:.0%} of ATR  (ATR=${sym_atr_5d:.0f})")
            return

        # -- Debug: cloud state + key values every bar --------------------
        if DEBUG_SIGNALS:
            _c = df_3m.iloc[-1]
            _p = df_3m.iloc[-2] if len(df_3m) >= 2 else _c
            _flip_l = _p.ema5 <= _p.ema12 and _c.ema5 > _c.ema12
            _flip_s = _p.ema5 >= _p.ema12 and _c.ema5 < _c.ema12
            _c2 = "GRN" if _c.ema5 > _c.ema12 else "RED"
            _c3 = "GRN" if _c.ema34 > _c.ema50 else "RED"
            _vol_ok = _c.vol_ma20 <= 0 or _c.volume >= VOLUME_CONFIRM_MULT * _c.vol_ma20
            _tag = " <<FLIP!" if (_flip_l or _flip_s) else ""
            # Show real clock + effective (bar-aligned) time on paper so the
            # offset is visible at a glance.  On live they're the same.
            _time_str = (f"{now.strftime('%H:%M')}(eff {effective_now.strftime('%H:%M')})"
                         if TWS_PORT != 7496 else now.strftime('%H:%M'))
            print(
                f"  {_time_str}  {symbol:6s}"
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
                df_3m, trend, bar_time=effective_now,
                pmh=self.pmh.get(symbol), pml=self.pml.get(symbol),
                support=sup, resistance=res)

        if signal == "none":
            # -- Explain what blocked a 5/12 flip --------------------------
            # As of 2026-05-28: C2 (5/12) flip is the ONLY core trigger.
            # C3 (34/50), 10m trend, and volume are NO LONGER blockers.
            # The only thing that can stop a flip from firing is _stop_ok():
            #   - stop too wide (>2.5% of entry price)
            #   - stop too narrow (<0.25% of entry price = noise)
            #   - stop on wrong side of entry (bad direction)
            if DEBUG_SIGNALS:
                _c = df_3m.iloc[-1]
                _p = df_3m.iloc[-2] if len(df_3m) >= 2 else _c
                _flip_l = _p.ema5 <= _p.ema12 and _c.ema5 > _c.ema12
                _flip_s = _p.ema5 >= _p.ema12 and _c.ema5 < _c.ema12
                if _flip_l or _flip_s:
                    _dir = "LONG" if _flip_l else "SHORT"
                    _e12, _close = _c.ema12, _c.close
                    _low, _high  = _c.low, _c.high
                    # Recompute the engine's stop to show the user EXACTLY why
                    if _flip_l:
                        _stop = min(_e12, _low)
                        _dist = _close - _stop
                    else:
                        _stop = max(_e12, _high)
                        _dist = _stop - _close
                    _pct = _dist / _close if _close > 0 else 0
                    _reasons = []
                    if _dist <= 0:
                        _reasons.append(f"stop=${_stop:.2f} on wrong side of entry ${_close:.2f}")
                    elif _pct < 0.0025:
                        _reasons.append(f"stop ${_dist:.2f} too tight ({_pct*100:.2f}% < 0.25%)")
                    elif _pct > 0.025:
                        _reasons.append(f"stop ${_dist:.2f} too wide ({_pct*100:.2f}% > 2.5%)")
                    else:
                        _reasons.append(f"unknown — stop=${_stop:.2f} dist=${_dist:.2f}")
                    tlog.info(
                        f"BLOCKED  {symbol}  {_dir}  FLIP  [{' | '.join(_reasons)}]"
                        f"  px=${_close:.2f}  e12=${_e12:.2f}  e50=${_c.ema50:.2f}"
                    )
            return
        entry_price = cur.close
        stop_dist   = abs(entry_price - stop_price)

        # Share sizing: full position is 100sh (< $500) or 50sh ($500+).
        # Starter enters at STARTER_RATIO of full; the rest is added once
        # the trade shows ADD_TRIGGER_PROFIT per share.
        n_full  = FIXED_SHARES_HIGH if entry_price >= HIGH_PRICE_THRESHOLD else FIXED_SHARES
        n       = max(1, int(n_full * STARTER_RATIO))   # starter shares
        n_add   = n_full - n                             # shares added on confirmation
        # Risk check uses STARTER shares only (bounded, smaller initial risk)
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

        # Level proximity gate — only enter AT Rip's level, not mid-range.
        # Exempt: gap entries (is_gap_entry) and cloud continuation entries
        # (cloud_cont / cloud_cont_crash) — those are specifically for stocks
        # that have already moved past the key level and are continuing.
        is_gap_entry  = (entry_reason.startswith("gap_") or
                         entry_reason.startswith("open_cloud_break"))
        is_cont_entry = (entry_reason in ("cloud_cont", "cloud_cont_crash") or
                         entry_reason.startswith("cloud_512_flip"))
        if not is_gap_entry and not is_cont_entry and signal == "long" and res is not None:
            if entry_price > res * (1 + LEVEL_PROX_LONG):
                tlog.info(
                    f"SKIP  {symbol}  LONG   ${entry_price:.2f}  "
                    f"chasing +{(entry_price/res - 1)*100:.1f}% above res ${res:.2f}"
                )
                return
        if not is_gap_entry and not is_cont_entry and signal == "short" and sup is not None:
            if entry_price < sup * (1 - LEVEL_PROX_SHORT):
                tlog.info(
                    f"SKIP  {symbol}  SHORT  ${entry_price:.2f}  "
                    f"chasing -{(1 - entry_price/sup)*100:.1f}% below sup ${sup:.2f}"
                )
                return

        # -- All checks passed — enter the starter position ---------------
        slot = len(self.positions) + 1
        tlog.info(
            f"ENTRY  [{slot}/{MAX_SIMULTANEOUS_POSITIONS}]  "
            f"{signal.upper():<5}  {symbol}  "
            f"x{n}sh(+{n_add} add)  "
            f"${entry_price:.2f}  stop=${stop_price:.2f}  "
            f"dist=${stop_dist:.2f} ({stop_dist/entry_price*100:.2f}%)  "
            f"risk=${risk:.0f}  [{entry_reason}]  "
            f"trend={trend}  sup={sup}  res={res}"
        )
        self._open_position(symbol, signal, entry_price, stop_price, n, now,
                            shares_full=n_full, shares_add=n_add,
                            entry_reason=entry_reason, level_res=res, level_sup=sup)

    # ----------------------------------------------------------------------
    # Order + position helpers
    # ----------------------------------------------------------------------

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
                       shares_full: int = 0,
                       shares_add: int = 0,
                       entry_reason: str = "",
                       level_res: Optional[float] = None,
                       level_sup: Optional[float] = None):
        """Place the starter entry order and register fill/cancel callbacks.

        The Position object is created ONLY inside the fillEvent callback —
        never immediately after placeOrder.  This prevents ghost positions
        when TWS cancels the order (error 10349, 354, etc.).

        shares      = starter shares to buy/sell now
        shares_full = intended full position (starter + add)
        shares_add  = shares to add later when ADD_TRIGGER_PROFIT is reached
        """
        action   = "BUY" if direction == "long" else "SELL"
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        trade = self.ib.placeOrder(contract, _entry_order(action, shares))
        log.info(f"  ORDER: {action} {shares}sh starter (full={shares_full}sh)  {symbol}  "
                 f"(order #{trade.order.orderId}  ~${entry_price:.2f})")

        # Park metadata — Position is built in _on_fill, not here.
        self._pending_entries[symbol] = {
            "direction":    direction,
            "shares":       shares,
            "shares_full":  shares_full or shares,
            "shares_add":   shares_add,
            "entry_price":  entry_price,   # signal-bar close (estimate)
            "stop_price":   stop_price,
            "time":         time,
            "entry_reason": entry_reason,
            "level_res":    level_res,
            "level_sup":    level_sup,
            "contract":     contract,
        }

        def _on_fill(t, fill):
            if symbol not in self._pending_entries:
                return   # partial-fill re-fire guard
            info      = self._pending_entries.pop(symbol)
            actual_px = fill.execution.avgPrice
            pos = Position(
                symbol=symbol,
                direction=info["direction"],
                shares=info["shares"],
                entry_price=actual_px,
                entry_time=info["time"],
                stop_price=info["stop_price"],
                ibkr_order_id=t.order.orderId,
                entry_signal=info["entry_reason"],
                level_res=info["level_res"],
                level_sup=info["level_sup"],
                shares_full=info["shares_full"],
                shares_add=info["shares_add"],
            )
            self.positions[symbol] = pos
            log.info(f"  FILLED: {action} {info['shares']}sh starter  {symbol}  "
                     f"@${actual_px:.2f}  (add {info['shares_add']}sh when +${ADD_TRIGGER_PROFIT:.0f}/sh)")
            # Crash stop placed only after confirmed fill — no phantom stops.
            self._place_crash_stop(symbol, info["contract"],
                                   info["direction"], info["shares"], info["stop_price"])

        def _on_cancelled(t):
            removed = self._pending_entries.pop(symbol, None)
            if removed is not None:
                log.warning(f"  {symbol}: entry order cancelled — no position opened  "
                            f"(order #{t.order.orderId})")

        trade.fillEvent      += _on_fill
        trade.cancelledEvent += _on_cancelled

    def _place_crash_stop(self, symbol: str, contract,
                          direction: str, shares: int, stop_price: float):
        """Place TWS crash-backstop STP after a confirmed fill.
        If the bot crashes mid-trade, TWS closes the position at the initial
        stop — preventing an unmanaged overnight position.
        Cancelled automatically by _close_position on any normal exit.
        """
        _crash_action = "SELL" if direction == "long" else "BUY"
        _stop_ord               = Order()
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
        self.ib.placeOrder(contract, _entry_order(action, shares))

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

        # Cancel the crash-backstop STP FIRST — before placing the exit MKT.
        # If STP and managed stop are at the same price, both can trigger
        # simultaneously: STP buys to close the short, then MKT also buys,
        # leaving an unintended long position.  Cancelling first prevents this.
        _twss = self._twss_stop_orders.pop(symbol, None)
        if _twss is not None:
            try:
                self.ib.cancelOrder(_twss)
                log.info(f"  CRASH STOP CANCELLED: {symbol}  (order #{_twss.orderId})")
            except Exception as _e:
                log.warning(f"  {symbol}: crash stop cancel failed — {_e}")

        action   = "SELL" if pos.direction == "long" else "BUY"
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        self.ib.placeOrder(contract, _entry_order(action, pos.shares))

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

        # (crash stop already cancelled at the top of this method)

        del self.positions[symbol]

    # ----------------------------------------------------------------------
    # Hot-add a symbol mid-session
    # ----------------------------------------------------------------------

    def _hot_add_symbol(self, symbol: str,
                        support=None, resistance=None):
        """Load a new symbol into the watchlist while the bot is running.
        Called when hot_add.json appears in the working directory.
        Runs the same ATR check, bar load, and trend init as subscribe_bars().
        """
        symbol = symbol.upper().strip()
        if symbol in self.symbols:
            log.warning(f"HOT-ADD {symbol}: already in watchlist — skipped")
            return
        try:
            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            # Real-time data subscription first (same reason as subscribe_bars)
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.tickers[symbol] = ticker
            self.ib.sleep(2.0)   # 2 s: let subscription register before hist requests

            # 10-min history for ATR check + trend
            b10 = self.ib.reqHistoricalData(
                contract, endDateTime="", durationStr="20 D",
                barSizeSetting=BAR_SIZE_10M, whatToShow="TRADES",
                useRTH=False, keepUpToDate=True,
            )
            if not b10:
                log.warning(f"HOT-ADD {symbol}: no 10m bar data — skipped")
                return

            day_hi: dict = collections.defaultdict(float)
            day_lo: dict = collections.defaultdict(lambda: float("inf"))
            for bar in self._closed_bars(b10):
                d = bar.date.date() if hasattr(bar.date, "date") else bar.date
                day_hi[d] = max(day_hi[d], bar.high)
                day_lo[d] = min(day_lo[d], bar.low)
            recent_days = sorted(day_hi)[-5:]
            avg_range = sum(day_hi[d] - day_lo[d] for d in recent_days) / len(recent_days)

            if avg_range < MIN_DAILY_RANGE:
                log.warning(
                    f"HOT-ADD {symbol}: ATR ${avg_range:.2f} < min ${MIN_DAILY_RANGE:.0f} — skipped"
                )
                return

            self._sym_atr[symbol] = avg_range
            dtr_tag = " (DTR exempt)" if avg_range >= DTR_EXEMPT_ATR else ""

            # 3-min history for entry signals
            b3 = self.ib.reqHistoricalData(
                contract, endDateTime="", durationStr="3 D",
                barSizeSetting=BAR_SIZE_3M, whatToShow="TRADES",
                useRTH=False, keepUpToDate=True,
            )
            if not b3:
                log.warning(f"HOT-ADD {symbol}: no 3m bar data — skipped")
                return

            # Wire live-bar callbacks (same pattern as subscribe_bars)
            def _on_3m_update(bars, hasNewBar, sym=symbol):
                if hasNewBar:
                    self._3m_update_set.add(sym)

            def _on_10m_update(bars, hasNewBar, sym=symbol):
                if hasNewBar:
                    self._10m_update_set.add(sym)

            b3.updateEvent  += _on_3m_update
            b10.updateEvent += _on_10m_update

            # Wire up all state
            self.bars_10m[symbol]         = b10
            self.bars_3m[symbol]          = b3
            self.trend[symbol]            = "none"
            self.plan[symbol]             = {"support": support, "resistance": resistance}
            self.pmh[symbol]              = None   # no pre-market data mid-session
            self.pml[symbol]              = None
            self._trades_today[symbol]    = 0
            self.symbols.append(symbol)

            self._on_new_bar_10m(symbol, b10)   # set initial trend

            log.info(
                f"HOT-ADD  {symbol}  ATR=${avg_range:.2f}{dtr_tag}"
                f"  trend={self.trend[symbol]}"
                f"  sup={support}  res={resistance}"
            )
            tlog.info(
                f"HOT-ADD  {symbol}  ATR=${avg_range:.2f}"
                f"  trend={self.trend[symbol]}"
                f"  sup={support}  res={resistance}"
            )

        except Exception as e:
            log.error(f"HOT-ADD {symbol} failed: {e}", exc_info=True)

    # ----------------------------------------------------------------------
    # Startup catch-up — evaluate bars that closed before updateEvent was wired
    # ----------------------------------------------------------------------

    def _startup_catchup(self):
        """Evaluate the most recent closed 3m bar for opening-drive signals.

        Problem: if the bot starts at 09:37, the 09:33 bar closed BEFORE
        subscribe_bars() wired up the updateEvent handler, so we never received
        the hasNewBar=True notification for it.  This method re-evaluates the
        last closed bar for each symbol to catch that missed opening signal.

        Only runs during the 09:33-10:10 ET window.  Outside that window there
        is no opening-drive opportunity to catch up on.
        """
        now      = datetime.now()
        now_mins = now.hour * 60 + now.minute
        gap_start = GAP_ENTRY_START_HOUR * 60 + GAP_ENTRY_START_MINUTE   # 09:33 = 573
        gap_end   = GAP_ENTRY_END_HOUR   * 60 + GAP_ENTRY_END_MINUTE     # 10:00 = 600
        if not (gap_start <= now_mins <= gap_end + 10):
            return   # not in the opening window — nothing to catch up on

        log.info("Startup catch-up: checking for missed opening-drive bars")
        for symbol in self.symbols:
            bars = self.bars_3m.get(symbol)
            if not bars:
                continue
            closed = self._closed_bars(bars)
            if not closed:
                continue
            last_bar = closed[-1]
            try:
                bar_dt = pd.Timestamp(last_bar.date)
                # Normalise to tz-naive Eastern for comparison with datetime.now()
                if bar_dt.tzinfo is not None:
                    bar_dt = bar_dt.tz_convert("America/New_York").tz_localize(None)
                bar_mins = bar_dt.hour * 60 + bar_dt.minute
                bar_date = bar_dt.date()
            except Exception as exc:
                log.warning(f"  {symbol}: startup_catchup date parse error — {exc}")
                continue

            # Only evaluate a bar from today that falls inside the opening window
            if bar_date == now.date() and gap_start <= bar_mins <= gap_end + 10:
                log.info(f"  {symbol}: catch-up evaluating {bar_dt.strftime('%H:%M')} bar")
                self._on_new_bar_3m(symbol, bars)
            else:
                log.info(f"  {symbol}: catch-up — last bar {bar_dt.strftime('%H:%M')} "
                         f"not in window or not today, skip")

    # ----------------------------------------------------------------------
    # Run
    # ----------------------------------------------------------------------

    def run(self):
        self.connect()
        self.subscribe_bars()
        self.setup_premarket_levels()
        self._startup_catchup()   # catch any opening-drive bar we missed during startup

        _mode = "PAPER" if TWS_PORT != 7496 else "LIVE"
        _delay = f"  data delay={PAPER_DATA_DELAY_MINUTES}min (gates shifted)" if TWS_PORT != 7496 else ""
        tlog.info(f"BOT LIVE [{_mode}]  symbols=" + ", ".join(self.symbols))
        tlog.info(f"  entry gate: 09:{FIRST_ENTRY_MINUTE:02d} ET | "
                  f"max_pos={MAX_SIMULTANEOUS_POSITIONS} | "
                  f"risk_cap=${MAX_RISK_DOLLARS} (<${HIGH_PRICE_THRESHOLD:.0f}) "
                  f"/ ${MAX_RISK_DOLLARS_HIGH} (>=${HIGH_PRICE_THRESHOLD:.0f}) | "
                  f"DTR_max={DTR_MAX_PCT:.0%} | "
                  f"debug={'ON' if DEBUG_SIGNALS else 'OFF'}{_delay}")
        tlog.info("-" * 60)
        for sym in self.symbols:
            p          = self.plan.get(sym, {})
            pmh        = self.pmh.get(sym)
            pml        = self.pml.get(sym)
            prev_close = self._sym_prev_close.get(sym)
            # Gap estimate: compare pre-market high vs previous close
            # (rough proxy — will refine once the 09:33 bar arrives)
            if pmh is not None and prev_close:
                gap_pct  = (pmh - prev_close) / prev_close * 100
                gap_arrow = "+" if gap_pct > 0 else "-"
                gap_str  = f"  gap~{gap_arrow}{abs(gap_pct):.1f}%  (prev=${prev_close:.2f})"
            elif prev_close:
                gap_str = f"  prev=${prev_close:.2f}"
            else:
                gap_str = ""
            pm_str = (f"  PM_H=${pmh:.2f}  PM_L=${pml:.2f}"
                      if pmh is not None and pml is not None else "  PM=n/a")
            tlog.info(
                f"  {sym:6s}  sup={p.get('support')}  res={p.get('resistance')}"
                f"{pm_str}{gap_str}"
            )
        tlog.info("-" * 60)

        # Main event loop.
        # keepUpToDate=True (wired in subscribe_bars) pushes completed bars
        # via updateEvent.  Callbacks just queue the symbol; heavy work runs
        # here after ib.sleep() returns — no reentrancy risk.
        # 1-min management still polls because 1m bars use a one-shot request.
        last_1m_id  = -1   # bar-close ID already processed for 1m position management

        # Reconnect counter — try 3x then give up gracefully
        _reconnect_attempts = 0

        while True:
            try:
                self.ib.sleep(1)
            except Exception as _e:
                log.warning(f"ib.sleep error: {_e}")
                continue

            if not self.ib.isConnected():
                if _reconnect_attempts >= 3:
                    log.error("TWS disconnected — 3 reconnect attempts failed, stopping.")
                    break
                _reconnect_attempts += 1
                log.warning(f"TWS disconnected — reconnect attempt {_reconnect_attempts}/3 ...")
                try:
                    self.ib.sleep(5)
                    self.ib.connect(TWS_HOST, TWS_PORT, clientId=TWS_CLIENT_ID)
                    log.info("Reconnected to TWS")
                    _reconnect_attempts = 0
                except Exception as _conn_e:
                    log.warning(f"Reconnect failed: {_conn_e}")
                continue

            now = datetime.now()

            # -- Hot-add check (fires every second) ------------------------
            # Drop hot_add.json in the bot directory from a second terminal:
            #   python add_symbol.py NVDA 120.00 125.00
            _hot_path = pathlib.Path("hot_add.json")
            if _hot_path.exists():
                try:
                    _data = json.loads(_hot_path.read_text())
                    _sym  = _data.get("symbol", "").upper().strip()
                    if _sym:
                        self._hot_add_symbol(
                            _sym,
                            support=_data.get("support"),
                            resistance=_data.get("resistance"),
                        )
                except Exception as _e:
                    log.warning(f"hot_add.json error: {_e}")
                finally:
                    _hot_path.unlink(missing_ok=True)

            # -- Real-time stop + half-exit check (fires every second) -----
            # Uses live ticker last-price — no bar delay.
            # This is the primary stop-loss mechanism now that reqMktData is
            # active.  The 1-min bar handler still updates the ratchet (needs
            # EMA50 from the 3-min bars) but the hard exit fires here first.
            for _sym in list(self.positions.keys()):
                _pos = self.positions.get(_sym)
                if not _pos or not _pos.is_open:
                    continue
                _rt = self._rt_price(_sym)
                if _rt is None:
                    continue

                # -- Add-in trigger — fire when starter shows ADD_TRIGGER_PROFIT --
                # Placed BEFORE half-exit so the add is at full size when we hit level.
                if not _pos.add_triggered and _pos.shares_add > 0:
                    _starter_profit = (
                        (_pos.entry_price - _rt) if _pos.direction == "short"
                        else (_rt - _pos.entry_price)
                    )
                    if _starter_profit >= ADD_TRIGGER_PROFIT:
                        _pos.add_triggered = True   # flag immediately → prevent double
                        _add_action   = "SELL" if _pos.direction == "short" else "BUY"
                        _add_sh       = _pos.shares_add
                        _add_contract = Stock(_sym, "SMART", "USD")
                        try:
                            self.ib.qualifyContracts(_add_contract)
                            _add_trade = self.ib.placeOrder(
                                _add_contract, _entry_order(_add_action, _add_sh))
                            tlog.info(
                                f"ADD  {_sym}  {_add_action}  x{_add_sh}sh  "
                                f"trigger +${_starter_profit:.2f}/sh  "
                                f"(order #{_add_trade.order.orderId})"
                            )
                            log.info(f"  ADD-IN TRIGGERED: {_sym}  +${_starter_profit:.2f}/sh  "
                                     f"placing {_add_action} {_add_sh}sh")

                            def _on_add_fill(t, fill,
                                             _p=_pos, _s=_sym, _ash=_add_sh):
                                add_px   = fill.execution.avgPrice
                                total_sh = _p.shares + _ash
                                avg_px   = (_p.entry_price * _p.shares
                                            + add_px * _ash) / total_sh
                                _p.add_entry_price = add_px
                                _p.entry_price     = avg_px
                                _p.shares          = total_sh
                                _p.original_shares = total_sh
                                log.info(
                                    f"  ADD FILLED: {_s}  @${add_px:.2f}  "
                                    f"avg_entry=${avg_px:.2f}  total={total_sh}sh"
                                )
                                tlog.info(
                                    f"ADD_FILL  {_s}  x{_ash}sh  @${add_px:.2f}  "
                                    f"avg=${avg_px:.2f}  total={total_sh}sh"
                                )

                            _add_trade.fillEvent += _on_add_fill
                        except Exception as _e:
                            log.warning(f"  {_sym}: add-in order failed — {_e}")
                            _pos.add_triggered = False  # allow retry on next second

                # Half-exit at Rip's level — real-time resolution
                if not _pos.half_exited:
                    _half_px = None
                    if _pos.direction == "long" and _pos.level_res and _rt >= _pos.level_res:
                        _half_px = _pos.level_res
                    elif _pos.direction == "short" and _pos.level_sup and _rt <= _pos.level_sup:
                        _half_px = _pos.level_sup
                    if _half_px is not None:
                        _half_sh  = _pos.shares // 2
                        _half_pnl = ((_half_px - _pos.entry_price) * _half_sh
                                     if _pos.direction == "long"
                                     else (_pos.entry_price - _half_px) * _half_sh)
                        if _half_pnl > 0 and _half_sh > 0:
                            self._close_partial(_sym, _half_sh, _half_px, now,
                                                reason="half@level_rt")
                            _pos.shares -= _half_sh
                        _pos.half_exited = True

                # Hard stop — exit immediately if real-time price crosses stop
                if _pos.direction == "long" and _rt <= _pos.stop_price:
                    tlog.info(
                        f"RT STOP  LONG  {_sym}  last=${_rt:.2f}"
                        f"  stop=${_pos.stop_price:.2f}  {now.strftime('%H:%M:%S')}"
                    )
                    self._close_position(_sym, _pos.stop_price, now, "stop_rt")
                elif _pos.direction == "short" and _rt >= _pos.stop_price:
                    tlog.info(
                        f"RT STOP  SHORT {_sym}  last=${_rt:.2f}"
                        f"  stop=${_pos.stop_price:.2f}  {now.strftime('%H:%M:%S')}"
                    )
                    self._close_position(_sym, _pos.stop_price, now, "stop_rt")

            # -- 1-min position management ---------------------------------
            # Only fires when we're in a trade.  Ratchet + stop check every
            # minute so we don't give back intrabar gains.  No entry logic.
            bar_1m_id = now.hour * 100 + now.minute
            if bar_1m_id != last_1m_id and now.second <= 5 and self.positions:
                last_1m_id = bar_1m_id
                # Skip if this minute is also the 3-min bar close — the
                # 3-min handler will run immediately after and covers it.
                if now.minute % 3 != 0:
                    try:
                        self._refresh_bars_1m()
                    except Exception as _e:
                        log.warning(f"  _refresh_bars_1m error: {_e}")

            # -- Event-driven bar processing -------------------------------
            # keepUpToDate=True pushes completed bars via updateEvent callbacks.
            # Callbacks just add the symbol to a pending set; heavy work
            # (EMA compute + signal eval) runs here after ib.sleep() returns.

            # 10-min trend updates — process first so trend is current when
            # the 3m handler runs (both can arrive in the same second).
            # Per-symbol try/except so one bad bar can't crash the loop.
            if self._10m_update_set:
                batch10 = list(self._10m_update_set)
                self._10m_update_set.clear()
                for sym in batch10:
                    if sym in self.bars_10m:
                        try:
                            self._on_new_bar_10m(sym, self.bars_10m[sym])
                        except Exception as _e:
                            log.exception(f"  {sym}: 10m bar handler error — {_e}")

            # 3-min entry + management signals
            if self._3m_update_set:
                batch3 = list(self._3m_update_set)
                self._3m_update_set.clear()
                for sym in batch3:
                    if sym in self.bars_3m:
                        try:
                            self._on_new_bar_3m(sym, self.bars_3m[sym])
                        except Exception as _e:
                            log.exception(f"  {sym}: 3m bar handler error — {_e}")

            # -- Fallback polling: fire at every 3-min mark ----------------
            # keepUpToDate=True is supposed to keep bars[-1] live, but the
            # stream goes silent (paper restarts, network blips, IBKR cashfarm
            # disconnects) and the cached bars list stops advancing.  When that
            # happens, just adding symbols to _3m_update_set is useless — the
            # dedup guard sees the same last-bar timestamp and skips.
            #
            # Fix: at every 3-min boundary, do a FRESH reqHistoricalData pull
            # via _refresh_bars().  That replaces self.bars_3m[sym] with a
            # current snapshot and runs the entry handler with the new bar.
            # Runs at :00-:15 seconds of each 3-min boundary (09:33, 09:36 …).
            if now.minute % 3 == 0 and now.second <= 15:
                _bar3_id = now.hour * 100 + now.minute
                if not hasattr(self, "_last_fallback_3m") or self._last_fallback_3m != _bar3_id:
                    self._last_fallback_3m = _bar3_id
                    # Fresh historical pull — guarantees the latest closed bar.
                    # Refresh 10m too at every 10-min boundary.
                    refresh_10m = (now.minute % 10 == 0)
                    try:
                        self._refresh_bars(refresh_10m=refresh_10m)
                    except Exception as _e:
                        log.warning(f"  fallback _refresh_bars error: {_e}")

    # ----------------------------------------------------------------------
    # Session summary
    # ----------------------------------------------------------------------

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
