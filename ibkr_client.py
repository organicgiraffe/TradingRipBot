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
                    RATCHET_START, RATCHET_GIVEBACK,
                    MAX_RISK_DOLLARS, MAX_RISK_DOLLARS_HIGH, MIN_DAILY_RANGE,
                    MIN_DAILY_RANGE_PCT, MAX_STOP_PCT,
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
    def __init__(self, symbols: list[str], plan: dict, ib=None):
        """
        symbols : list of tickers, e.g. ['TSLA', 'NVDA', 'AMD']
        plan    : {symbol: {'support': float|None, 'resistance': float|None}}
                  Missing symbols default to rules-only (no level filter).
        ib      : optional IB-compatible instance.  Production passes None
                  (real ib_insync.IB used).  Tests pass MockIB for
                  deterministic scenario testing without TWS.
        """
        self.symbols = [s.upper() for s in symbols]
        self.plan    = plan          # Rip's levels for today's session

        self.ib        = ib if ib is not None else IB()
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
        self._twss_stop_orders: dict = {}  # symbol -> ib_insync Trade object
                                            # (full Trade, NOT just Order — we need
                                            # Trade.orderStatus.status to confirm
                                            # cancellation before placing exit MKT.
                                            # On 5/29 storing only Order caused the
                                            # crash-stop race: status never updated,
                                            # wait loop timed out, STP + exit both
                                            # fired → unintended reverse position.)

        # Orders placed but not yet fill-confirmed.  Position object is created
        # inside the fillEvent callback — never before — so a cancelled entry
        # never creates a ghost position.  Also counted toward the slot limit.
        self._pending_entries: dict = {}   # symbol -> metadata dict

        # Symbols where EMERGENCY_FLATTEN / FILL_DRIFT couldn't confirm the
        # flatten order filled within the 3-second window.  The bot ADOPTS
        # the original fill as a managed Position (with a tight safe stop)
        # rather than walk away and leave an orphan in TWS.  Every 10 seconds
        # the main loop logs MANUAL_INTERVENTION_REQUIRED for symbols in here
        # until they're closed — so the operator notices and reviews TWS.
        # Cleared in _close_position once the position is properly exited.
        self._manual_intervention: dict = {}  # symbol -> reason string

        # Symbols currently being closed — RT loop must NOT re-check stops
        # on these or it'll fire a duplicate exit while the first one is in
        # flight (root cause of PLTR 14:14 ghost-LONG → orphan-SHORT bug).
        # Added the moment _close_position starts, removed only when the
        # position record is fully cleaned up.  If close hangs (timeout),
        # the symbol stays in this set so RT loop never re-fires.
        self._exit_in_progress: set = set()

        # In-flight EMERGENCY_FLATTEN / FILL_DRIFT orders awaiting confirmation.
        # CRITICAL: the flatten order is PLACED inside the _on_fill callback
        # (placeOrder is non-blocking, safe in a callback) but the confirm-or-
        # adopt decision is deferred to the MAIN LOOP.  The old code polled
        # with self.ib.sleep(0.1) INSIDE the fill callback — that re-enters
        # the ib_insync event loop and pumps nested events (other fills, bar
        # updates) mid-callback, which corrupts state.  This is the reentrancy
        # footgun the bot's own design comment warns about, and a root cause
        # of the 5/29 orphan trades.  Now the callback just queues here and
        # returns; _process_pending_flattens() runs in the safe main-loop
        # context where ib.sleep is legal.
        #   symbol -> {trade, deadline, info, actual_px, dir, kind, orig_order_id}
        self._pending_flattens: dict = {}

        # Last TWS reconciliation timestamp — runs every 30s in the main loop
        # to detect orphan positions (in TWS but not in self.positions) and
        # ghost positions (in self.positions but not in TWS).  Belt-and-
        # suspenders against any remaining race condition.
        self._last_reconcile_dt: Optional[datetime] = None

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

    def _resubscribe_after_reconnect(self):
        """Re-establish ALL streaming subscriptions after a TWS reconnect.

        A reconnect via ib.connect() restores the socket but does NOT restore
        the market-data tickers or the keepUpToDate bar streams created before
        the disconnect — those objects are dead.  Without this, the bot reports
        'Reconnected' and looks healthy but is BLIND: _rt_price returns None,
        bars stop updating, and RT stops / ratchet / EOD never fire.  Any open
        position is then effectively unmanaged (only the TWS-side crash STP
        protects it).  This re-subscribes everything so management resumes.
        """
        log.warning("  RECONNECT: re-subscribing market data + bar streams ...")
        # Allow the per-symbol auto-recovery path to fire again post-reconnect.
        if hasattr(self, "_resub_attempted"):
            self._resub_attempted = set()

        for symbol in list(self.symbols):
            try:
                contract = Stock(symbol, "SMART", "USD")
                self.ib.qualifyContracts(contract)

                # Market-data ticker (drives _rt_price)
                try:
                    self.ib.cancelMktData(contract)
                except Exception:
                    pass
                self.ib.sleep(0.3)
                ticker = self.ib.reqMktData(contract, '', False, False)
                self.tickers[symbol] = ticker

                # keepUpToDate bar streams (drive bar-based exits)
                b10 = self.ib.reqHistoricalData(
                    contract, endDateTime="", durationStr="20 D",
                    barSizeSetting=BAR_SIZE_10M, whatToShow="TRADES",
                    useRTH=False, keepUpToDate=True,
                )
                b3 = self.ib.reqHistoricalData(
                    contract, endDateTime="", durationStr="3 D",
                    barSizeSetting=BAR_SIZE_3M, whatToShow="TRADES",
                    useRTH=False, keepUpToDate=True,
                )
                if b10:
                    self.bars_10m[symbol] = b10
                if b3:
                    self.bars_3m[symbol] = b3

                # Re-wire the lightweight update callbacks (queue-only).
                def _on_3m_update(bars, hasNewBar, sym=symbol):
                    if hasNewBar:
                        self._3m_update_set.add(sym)

                def _on_10m_update(bars, hasNewBar, sym=symbol):
                    if hasNewBar:
                        self._10m_update_set.add(sym)

                if b3:
                    b3.updateEvent  += _on_3m_update
                if b10:
                    b10.updateEvent += _on_10m_update

                log.info(f"  RECONNECT: {symbol} re-subscribed "
                         f"({len(b10)} x10m, {len(b3)} x3m)")
            except Exception as _e:
                log.error(f"  RECONNECT: {symbol} re-subscribe FAILED — {_e}.  "
                          f"This symbol is blind until next reconnect; crash STP "
                          f"still protects any open position.")

        # Re-sync open orders / positions so _twss_stop_orders Trade objects and
        # self.positions reflect post-reconnect TWS truth.  Reconciliation on the
        # next 30s tick will flag/clean any divergence.
        log.warning("  RECONNECT: re-subscribe complete — reconciliation will "
                    "verify position state on next tick.")

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
                # MATCH BACKTEST: filter on both absolute ($) AND percentage of price.
                # The 3m bar close is the best price proxy at startup.
                last_px = float(self._closed_bars(b10)[-1].close) if self._closed_bars(b10) else 0.0
                range_pct = (avg_range / last_px) if last_px > 0 else 0.0
                if avg_range < MIN_DAILY_RANGE:
                    log.info(
                        f"  {symbol}: SKIPPED — 5-day ATR ${avg_range:.2f} < min ${MIN_DAILY_RANGE:.2f}"
                    )
                    self.symbols.remove(symbol)
                    continue
                if last_px > 0 and range_pct < MIN_DAILY_RANGE_PCT:
                    log.info(
                        f"  {symbol}: SKIPPED — range {range_pct*100:.2f}% of price "
                        f"< min {MIN_DAILY_RANGE_PCT*100:.1f}%"
                    )
                    self.symbols.remove(symbol)
                    continue
                self._sym_atr[symbol] = avg_range
                # Match backtest DTR exemption: absolute ATR OR percent of price
                _exempt_pct = range_pct >= 0.030
                _exempt_atr = avg_range >= DTR_EXEMPT_ATR
                dtr_tag = " (DTR exempt — momentum stock)" if (_exempt_atr or _exempt_pct) else ""
                log.info(f"  {symbol}: 5-day ATR ${avg_range:.2f} ({range_pct*100:.1f}% of price) OK{dtr_tag}")

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
        """Best available real-time price for a symbol.  Tier fallback:
          1. ticker.last        (live trade price)
          2. ticker.midpoint    (bid/ask mid)
          3. ticker.close       (yesterday close, last resort if market just opened)
          4. portfolioItems     (IBKR portfolio marketPrice — updates every ~3s
                                 even when ticker subs are silent)
          5. last 3m bar close  (final fallback — at least 3 min old)
        Logs a warning the FIRST time we drop to portfolio/bar fallback per
        symbol so silent ticker failures don't go unnoticed (5/29 PLTR/CRM bug).
        """
        # Tier 1+2+3: ib_insync ticker
        t = self.tickers.get(symbol)
        if t is not None:
            px = t.last
            if px and px == px:
                return px
            mid = t.midpoint()
            if mid and mid == mid:
                return mid
            cl = getattr(t, "close", None)
            if cl and cl == cl:
                return cl

        # Tier 4: IBKR portfolio (updates without ticker subscription)
        try:
            for item in self.ib.portfolio():
                if item.contract.symbol == symbol and item.marketPrice:
                    mp = float(item.marketPrice)
                    if mp == mp:   # not NaN
                        if not getattr(self, "_warned_rt_fallback", set()).__contains__(symbol):
                            log.warning(f"  {symbol}: ticker silent — falling back to "
                                        f"portfolio marketPrice (${mp:.2f}).  "
                                        f"Ticker subscription may be broken.")
                            if not hasattr(self, "_warned_rt_fallback"):
                                self._warned_rt_fallback = set()
                            self._warned_rt_fallback.add(symbol)
                        return mp
        except Exception as _e:
            log.warning(f"  {symbol}: portfolio price lookup failed — {_e}")

        # Tier 5: last 3m bar close
        bars = self.bars_3m.get(symbol)
        if bars:
            try:
                _last_close = float(bars[-1].close)
                if _last_close == _last_close:
                    if not getattr(self, "_warned_rt_bar_fallback", set()).__contains__(symbol):
                        log.warning(f"  {symbol}: no live price ANYWHERE — using last 3m bar "
                                    f"close ${_last_close:.2f}.  This is up to 3 minutes stale.  "
                                    f"Manual stop/exit recommended.")
                        if not hasattr(self, "_warned_rt_bar_fallback"):
                            self._warned_rt_bar_fallback = set()
                        self._warned_rt_bar_fallback.add(symbol)
                    return _last_close
            except Exception:
                pass

        # Truly nothing — first time only per symbol, log it loudly
        if not getattr(self, "_warned_rt_none", set()).__contains__(symbol):
            log.error(f"  {symbol}: _rt_price returned NONE — real-time loop "
                      f"(stop/half/ratchet) will SKIP all checks for this symbol.  "
                      f"Check market data subscription.")
            if not hasattr(self, "_warned_rt_none"):
                self._warned_rt_none = set()
            self._warned_rt_none.add(symbol)
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
        # Count pending (order placed, awaiting fill) AND in-flight flattens
        # toward the slot limit, so we don't fire a second entry while the
        # first is still in-flight or while an EMERGENCY_FLATTEN/FILL_DRIFT is
        # still resolving (it may yet ADOPT into a managed position).
        if (len(self.positions) + len(self._pending_entries)
                + len(self._pending_flattens)) >= MAX_SIMULTANEOUS_POSITIONS:
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
        # Exempt high-ATR momentum stocks: absolute ATR ≥ DTR_EXEMPT_ATR OR
        # ATR/price ≥ 3% (small-cap monsters).  MATCHES BACKTEST EXACTLY.
        # Without the vol_pct branch, PLTR-style mid-caps get DTR-blocked live
        # while passing in backtest → live vs backtest mismatch.
        sym_atr_5d = getattr(self, "_sym_atr", {}).get(symbol, 0.0)
        _px        = float(cur.close) if cur is not None else 0.0
        _vol_pct   = (sym_atr_5d / _px) if _px > 0 else 0.0
        dtr_ratio  = compute_dtr_atr_ratio(df_10m, today, bar_time=effective_now)
        dtr_exempt = (sym_atr_5d >= DTR_EXEMPT_ATR) or (_vol_pct >= 0.030)
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

        # -- PRE-FILL STOP-DIRECTION CHECK ---------------------------------
        # The engine computed the stop using the SIGNAL-BAR CLOSE.  Between
        # signal generation and the MKT order actually filling, the live
        # price can drift past the stop level — turning a valid stop into
        # an inverted one (long with stop ABOVE entry, short with stop
        # BELOW entry).  EMERGENCY_FLATTEN catches this after the fill, but
        # that's still an unnecessary round-trip + spread cost.  Better to
        # check the live ticker price BEFORE placing the order and abort
        # if the stop would already be inverted at the current price.
        _live_px = self._rt_price(symbol)
        if _live_px is not None:
            _would_be_inverted = (
                (signal == "long"  and stop_price >= _live_px) or
                (signal == "short" and stop_price <= _live_px)
            )
            if _would_be_inverted:
                tlog.info(
                    f"SKIP_INVERTED  {signal.upper():<5}  {symbol}  "
                    f"signal_px=${entry_price:.2f}->live=${_live_px:.2f}  "
                    f"stop=${stop_price:.2f} now on wrong side  [{entry_reason}]"
                )
                log.warning(
                    f"  PRE-FILL inversion: {symbol} {signal.upper()} skipped.  "
                    f"Signal price ${entry_price:.2f} → live ${_live_px:.2f} moved "
                    f"past stop ${stop_price:.2f}.  No order placed."
                )
                return

            # Also recompute stop_dist against the LIVE price, since the
            # MKT order will fill near it.  If live-based dist exceeds the
            # max stop %, abort — same logic as _stop_ok at signal time.
            _live_dist = abs(_live_px - stop_price)
            _live_pct  = _live_dist / _live_px if _live_px > 0 else 1.0
            if _live_pct > MAX_STOP_PCT:
                tlog.info(
                    f"SKIP_WIDE  {signal.upper():<5}  {symbol}  "
                    f"live=${_live_px:.2f} stop=${stop_price:.2f}  "
                    f"dist={_live_pct*100:.2f}% > {MAX_STOP_PCT*100:.1f}%  [{entry_reason}]"
                )
                return

        # -- All checks passed — place the order.  ENTRY is logged inside
        # the fill callback (in _open_position) ONLY after TWS confirms the fill.
        # This way cancelled orders (Error 354, etc.) never write a misleading
        # "ENTRY" line into the trades log.
        slot = (len(self.positions) + len(self._pending_entries)
                + len(self._pending_flattens) + 1)
        log.info(
            f"  SIGNAL: {signal.upper():<5} {symbol}  "
            f"x{n}sh+{n_add}add  "
            f"~${entry_price:.2f}  stop=${stop_price:.2f}  "
            f"risk=${risk:.0f}  [{entry_reason}]  slot {slot}/{MAX_SIMULTANEOUS_POSITIONS}"
        )
        self._open_position(symbol, signal, entry_price, stop_price, n, now,
                            shares_full=n_full, shares_add=n_add,
                            entry_reason=entry_reason, level_res=res, level_sup=sup,
                            # Pass extra metadata so the fill callback can write
                            # an accurate ENTRY line using the real fill price.
                            entry_meta={
                                "slot": slot, "n_add": n_add,
                                "stop_dist": stop_dist, "risk": risk,
                                "trend": trend, "sup": sup, "res": res,
                            })

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
                       level_sup: Optional[float] = None,
                       entry_meta: Optional[dict] = None):
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
            "entry_meta":   entry_meta or {},
        }

        def _on_fill(t, fill):
            if symbol not in self._pending_entries:
                return   # partial-fill re-fire guard
            info      = self._pending_entries.pop(symbol)
            actual_px = fill.execution.avgPrice

            # ── STOP DIRECTION SAFETY (added 5/29 after SNDK debacle) ────────
            # If the fill price moved past the signal-bar's stop level, the
            # stop is on the WRONG SIDE of the entry — placing a crash STP
            # there will execute as a profit-taker (BUY STP below mkt for a
            # short, SELL STP above mkt for a long) instead of a stop-loss.
            # The rest of the bot's price-vs-stop logic doesn't validate
            # direction either, so all subsequent exits get mangled.
            # When detected, immediately flatten with an opposite-side MKT
            # order and log an EMERGENCY_FLATTEN — better to take the small
            # spread cost than run with an inverted stop for an hour.
            _stop = info["stop_price"]
            _dir  = info["direction"]
            _stop_inverted = (
                (_dir == "long"  and _stop >= actual_px) or
                (_dir == "short" and _stop <= actual_px)
            )
            if _stop_inverted:
                _opposite = "SELL" if _dir == "long" else "BUY"
                log.error(
                    f"  EMERGENCY_FLATTEN: {symbol} {_dir.upper()} filled @${actual_px:.2f} "
                    f"but stop ${_stop:.2f} is on WRONG SIDE (would execute as "
                    f"profit-taker, not stop-loss).  Closing position immediately."
                )
                tlog.info(
                    f"EMERGENCY_FLATTEN  {_dir.upper():<5}  {symbol}  "
                    f"x{info['shares']}sh  @${actual_px:.2f}  stop=${_stop:.2f}  "
                    f"reason=inverted_stop  signal={info['entry_reason']}"
                )
                # Place the flatten order and DEFER the confirm/adopt decision
                # to the main loop.  Do NOT poll with ib.sleep here — this is a
                # fill callback and sleeping re-enters the event loop (reentrancy
                # footgun, root cause of 5/29 orphans).  _queue_flatten just
                # places the order (non-blocking) and records it; the main loop's
                # _process_pending_flattens confirms or adopts on later ticks.
                self._queue_flatten(symbol, info, actual_px, _dir,
                                    kind="emergency_flatten",
                                    orig_order_id=t.order.orderId)
                return   # Decision deferred to main loop — no sleep in callback

            # ── Drift guard — if fill drifted > 1% from signal estimate,
            # something is wrong (stale paper data, fast move).  Abort
            # before placing crash stop on a price we didn't expect.
            _signal_px = info.get("entry_price", actual_px)
            if _signal_px > 0:
                _drift_pct = abs(actual_px - _signal_px) / _signal_px
                if _drift_pct > 0.01:
                    _opposite = "SELL" if _dir == "long" else "BUY"
                    log.error(
                        f"  FILL_DRIFT: {symbol} {_dir.upper()} signal=${_signal_px:.2f} "
                        f"but filled @${actual_px:.2f} ({_drift_pct*100:.2f}% drift). "
                        f"Closing position to avoid trading on stale signal."
                    )
                    tlog.info(
                        f"FILL_DRIFT  {_dir.upper():<5}  {symbol}  "
                        f"x{info['shares']}sh  signal=${_signal_px:.2f}->fill=${actual_px:.2f}  "
                        f"drift={_drift_pct*100:.2f}%  signal={info['entry_reason']}"
                    )
                    # Place flatten, defer confirm/adopt to main loop (no sleep
                    # in callback — same reentrancy fix as EMERGENCY_FLATTEN).
                    self._queue_flatten(symbol, info, actual_px, _dir,
                                        kind="fill_drift",
                                        orig_order_id=t.order.orderId)
                    return   # Decision deferred to main loop — no sleep in callback

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
            # ENTRY tlog written HERE — only after fill confirmation.
            # Uses the ACTUAL fill price, not the signal-bar estimate.
            _em = info.get("entry_meta") or {}
            _stop_dist_actual = abs(actual_px - info["stop_price"])
            _stop_pct_actual  = (_stop_dist_actual / actual_px * 100) if actual_px > 0 else 0.0
            tlog.info(
                f"ENTRY  [{_em.get('slot', '?')}/{MAX_SIMULTANEOUS_POSITIONS}]  "
                f"{info['direction'].upper():<5}  {symbol}  "
                f"x{info['shares']}sh(+{info['shares_add']} add)  "
                f"${actual_px:.2f}  stop=${info['stop_price']:.2f}  "
                f"dist=${_stop_dist_actual:.2f} ({_stop_pct_actual:.2f}%)  "
                f"risk=${_em.get('risk', 0):.0f}  [{info['entry_reason']}]  "
                f"trend={_em.get('trend', '')}  "
                f"sup={_em.get('sup')}  res={_em.get('res')}"
            )
            # Crash stop placed only after confirmed fill — no phantom stops.
            self._place_crash_stop(symbol, info["contract"],
                                   info["direction"], info["shares"], info["stop_price"])

        def _on_cancelled(t):
            removed = self._pending_entries.pop(symbol, None)
            if removed is not None:
                log.warning(f"  {symbol}: entry order cancelled — no position opened  "
                            f"(order #{t.order.orderId})")
                # Write to trades log too so we don't mistake a cancelled order
                # for a real entry in the daily review.
                tlog.info(
                    f"CANCELLED  {removed['direction'].upper():<5}  {symbol}  "
                    f"x{removed['shares']}sh  ~${removed['entry_price']:.2f}  "
                    f"[{removed['entry_reason']}]  order #{t.order.orderId}"
                )

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
            # Store the full Trade object so _close_position can read the live
            # orderStatus.status to confirm cancellation before placing exit MKT.
            self._twss_stop_orders[symbol] = _stop_trade
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
        _crash_trade = self._twss_stop_orders.get(symbol)   # full Trade now
        if _crash_trade is not None and _remaining > 0:
            _crash_ord = _crash_trade.order
            _crash_ord.totalQuantity = _remaining
            try:
                self.ib.placeOrder(contract, _crash_ord)   # sends modify to TWS
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

        # Mark exit as in-flight so the RT-stop loop in run() doesn't re-fire
        # _close_position on the same symbol while this one is mid-wait.
        # Without this, the 5/29 PLTR 14:14 sequence happened: 14:13:01 first
        # _close_position aborts after seeing STP filled, returns leaving
        # self.positions[PLTR] intact; 14:14:00 RT loop sees position still
        # open, stop_price still set; 14:14:01 fires _close_position again,
        # places a fresh SELL MKT into already-flat TWS account → SHORT 50
        # orphan.  Belongs in the set for the entire close attempt.
        if symbol in self._exit_in_progress:
            log.warning(f"  {symbol}: _close_position called while exit already "
                        f"in flight — skipping duplicate attempt.")
            return
        self._exit_in_progress.add(symbol)

        # Cancel the crash-backstop STP and WAIT for confirmation before
        # placing the exit MKT.  IB cancelOrder is async — without the wait,
        # the STP may still fire between cancelOrder and placeOrder,
        # causing a double-close that leaves an unintended reverse position.
        #
        # On 5/28 the race caused LONG 100 AVGO after a short close.  My first
        # fix (5/28 night) stored only the Order and polled Order.status —
        # which doesn't auto-update — so the wait silently timed out and the
        # race kept happening (5/29 SNDK -25 + CRM +50 phantom positions).
        # This version stores the Trade and polls Trade.orderStatus.status
        # which IS the live, auto-updating field.  If the cancel doesn't
        # confirm in 3 seconds, we ABORT the exit instead of placing a market
        # order that races the STP — better to leave the position open and
        # let the STP do its job.
        _crash_trade = self._twss_stop_orders.pop(symbol, None)
        if _crash_trade is not None:
            _crash_ord = _crash_trade.order
            try:
                self.ib.cancelOrder(_crash_ord)
                log.info(f"  CRASH STOP cancel sent: {symbol}  "
                         f"(order #{_crash_ord.orderId})  waiting for confirmation ...")
                # Poll Trade.orderStatus.status — this DOES auto-update.
                _confirmed = False
                _final_status = "?"
                for _i in range(30):   # up to 3 seconds
                    self.ib.sleep(0.1)
                    _final_status = _crash_trade.orderStatus.status
                    if _final_status in ("Cancelled", "Inactive", "ApiCancelled", "Filled"):
                        _confirmed = True
                        break
                if _final_status == "Filled":
                    # STP already filled — position is already closed in TWS.
                    # Do NOT place exit MKT or we'll open a REVERSE position.
                    # CRITICAL: must still close the Position RECORD here, or
                    # the RT loop will keep checking stops on this ghost and
                    # eventually fire another exit MKT (5/29 PLTR 14:14 bug).
                    # Use crash-stop price as exit price (close enough — the
                    # STP triggered at its stop level, slippage is small).
                    _stp_exit_px = float(_crash_trade.order.auxPrice or price)
                    log.warning(
                        f"  {symbol}: crash STP already FILLED @${_stp_exit_px:.2f} — "
                        f"closing Position record (NOT placing exit MKT, would create reverse)."
                    )
                    pos.close(_stp_exit_px, time, "crash_stp_filled")
                    pnl    = pos.pnl or 0.0
                    sign   = "+" if pnl >= 0 else ""
                    result = "WIN " if pnl > 0 else ("EVEN" if pnl == 0 else "LOSS")
                    dur    = int((time - pos.entry_time).total_seconds() // 60)
                    tlog.info(
                        f"EXIT   {result}  {pos.direction.upper():<5}  {symbol}  x{pos.shares}sh  "
                        f"${pos.entry_price:.2f}->${_stp_exit_px:.2f}  pnl={sign}${pnl:.0f}  "
                        f"[crash_stp_raced_us]  held={dur}min  signal={pos.entry_signal}"
                    )
                    self._trades_today[symbol] += 1
                    self.trade_log.append({
                        "time": time.strftime("%H:%M:%S"), "event": "exit",
                        "symbol": symbol, "direction": pos.direction,
                        "entry": pos.entry_price, "exit": _stp_exit_px,
                        "shares": pos.shares, "pnl": pnl,
                        "reason": "crash_stp_raced_us",
                        "entry_signal": pos.entry_signal,
                        "held_min": dur,
                    })
                    if symbol in self._manual_intervention:
                        self._manual_intervention.pop(symbol, None)
                    del self.positions[symbol]
                    self._exit_in_progress.discard(symbol)
                    return   # Done — STP already closed it, record is clean
                elif _confirmed:
                    log.info(f"  CRASH STOP CANCELLED confirmed: {symbol}  "
                             f"(status={_final_status})")
                else:
                    # Timed out — cancel did NOT confirm.  ABORT the exit
                    # rather than race the STP.  The STP will close the
                    # position on its own terms; we just lose the managed exit.
                    # CRITICAL: also mark manual_intervention + keep symbol in
                    # _exit_in_progress so the RT loop NEVER re-fires another
                    # close attempt while the STP is in limbo.  Reconciliation
                    # against TWS (every 30s) will clean up self.positions
                    # once the STP actually fills.
                    log.error(f"  {symbol}: crash stop cancel NOT confirmed "
                              f"after 3s (status={_final_status}) — "
                              f"ABORTING exit MKT to avoid race condition.  "
                              f"Reconciliation will clean up once STP resolves.")
                    self._manual_intervention[symbol] = (
                        f"crash_stop cancel timeout (status={_final_status}); "
                        f"STP order may still fire — review TWS"
                    )
                    # Put it back so reconciliation knows about the live STP
                    self._twss_stop_orders[symbol] = _crash_trade
                    # NOTE: leaving symbol in _exit_in_progress on purpose —
                    # do NOT discard it here.  Reconciliation removes it when
                    # the position is detected as flat in TWS.
                    return
            except Exception as _e:
                log.warning(f"  {symbol}: crash stop cancel failed — {_e}")
                self._manual_intervention[symbol] = (
                    f"crash_stop cancel exception: {_e}; review TWS"
                )
                # Same — leave in _exit_in_progress for reconciliation cleanup
                return

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

        # Clear manual-intervention flag if this was an adopted orphan that's
        # now been properly closed.  The 10s alert loop will stop nagging.
        if symbol in self._manual_intervention:
            log.info(f"  {symbol}: manual-intervention flag cleared (position closed)")
            self._manual_intervention.pop(symbol, None)

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
        self._exit_in_progress.discard(symbol)   # close complete, RT can re-engage

    # ----------------------------------------------------------------------
    # Deferred flatten handling — EMERGENCY_FLATTEN / FILL_DRIFT
    # ----------------------------------------------------------------------

    def _queue_flatten(self, symbol: str, info: dict, actual_px: float,
                       direction: str, kind: str, orig_order_id: int):
        """Place an opposite-side MKT to flatten an inverted/drifted fill, then
        record it for deferred confirmation in the main loop.

        CRITICAL: this is called from inside the _on_fill event callback.
        placeOrder() is non-blocking and safe in a callback, but we must NOT
        poll for the result with ib.sleep() here — that re-enters the event
        loop and pumps nested events (the 5/29 reentrancy bug).  The confirm-
        or-adopt decision happens in _process_pending_flattens() which runs in
        the main loop where ib.sleep is legal.
        """
        _opposite = "SELL" if direction == "long" else "BUY"
        _flat_trade = None
        try:
            _flat_trade = self.ib.placeOrder(
                info["contract"], _entry_order(_opposite, info["shares"]))
            log.info(f"  {kind.upper()}: flatten {_opposite} {info['shares']}sh "
                     f"{symbol} placed (order #{_flat_trade.order.orderId}) — "
                     f"awaiting confirmation in main loop")
        except Exception as _flat_e:
            log.error(f"  {kind.upper()} flatten placeOrder failed: {_flat_e}  "
                      f"— will ADOPT on next main-loop tick")

        self._pending_flattens[symbol] = {
            "trade":          _flat_trade,
            "deadline":       datetime.now() + timedelta(seconds=3),
            "info":           info,
            "actual_px":      actual_px,
            "dir":            direction,
            "kind":           kind,
            "orig_order_id":  orig_order_id,
        }

    def _process_pending_flattens(self, now: datetime):
        """Main-loop processor for in-flight EMERGENCY_FLATTEN / FILL_DRIFT
        orders.  For each pending flatten:
          * status == Filled  → flatten confirmed, drop it (clean exit).
          * rejected / deadline passed / order never placed → ADOPT the
            original fill as a managed Position with a tight safe stop, place
            a crash STP, and flag manual_intervention for operator review.
        Runs in the safe main-loop context (ib.sleep is legal here, though we
        don't even need it — we just read orderStatus once per tick).
        """
        for symbol, pf in list(self._pending_flattens.items()):
            _trade  = pf["trade"]
            _status = _trade.orderStatus.status if _trade is not None else "?"

            if _status == "Filled":
                log.info(f"  {pf['kind'].upper()} confirmed: {symbol} closed "
                         f"via opposite MKT (status=Filled)")
                self._pending_flattens.pop(symbol, None)
                continue

            _rejected = _status in ("Cancelled", "Inactive", "ApiCancelled")
            _expired  = now >= pf["deadline"]
            if not (_rejected or _expired or _trade is None):
                continue   # still working, within deadline — wait

            # ── ADOPT ──────────────────────────────────────────────────────
            info      = pf["info"]
            _dir      = pf["dir"]
            actual_px = pf["actual_px"]
            _safe_stop = (round(actual_px * 1.005, 2) if _dir == "short"
                          else round(actual_px * 0.995, 2))
            log.error(
                f"  {pf['kind'].upper()} UNCONFIRMED: {symbol} flatten status="
                f"{_status}.  ADOPTING original fill with safe stop "
                f"${_safe_stop:.2f} — review in TWS."
            )
            tlog.info(
                f"ADOPTED  {_dir.upper():<5}  {symbol}  x{info['shares']}sh  "
                f"@${actual_px:.2f}  safe_stop=${_safe_stop:.2f}  "
                f"reason={pf['kind']}_unconfirmed  signal={info['entry_reason']}"
            )
            pos = Position(
                symbol=symbol, direction=_dir, shares=info["shares"],
                entry_price=actual_px, entry_time=info["time"],
                stop_price=_safe_stop,
                ibkr_order_id=pf["orig_order_id"],
                entry_signal=info["entry_reason"] + "_ADOPTED",
                level_res=info["level_res"], level_sup=info["level_sup"],
                shares_full=info["shares_full"], shares_add=0,  # no add on adopted
            )
            self.positions[symbol] = pos
            self._place_crash_stop(symbol, info["contract"],
                                   _dir, info["shares"], _safe_stop)
            self._manual_intervention[symbol] = (
                f"adopted after {pf['kind']} unconfirmed (flat_status={_status}); "
                f"review and close manually if needed"
            )
            self._pending_flattens.pop(symbol, None)

    # ----------------------------------------------------------------------
    # TWS reconciliation — detect orphans (TWS has, bot doesn't) and
    # ghosts (bot has, TWS doesn't).  Runs every 30s from the main loop.
    # ----------------------------------------------------------------------

    def _reconcile_positions(self):
        """Compare self.positions to TWS account positions.  Detects:
          * ORPHAN  — symbol exists in TWS but bot has no Position record
                      (e.g. EMERGENCY_FLATTEN didn't reconcile, manual entry
                       in TWS, partial-fill of opposite-side flatten).
                      Fires loud MANUAL_INTERVENTION_REQUIRED.
          * GHOST   — symbol in self.positions but TWS shows 0 shares
                      (e.g. crash STP closed it, manual close in TWS).
                      Auto-removes the ghost from self.positions and logs.
          * MISMATCH — share count or direction differs between TWS and bot.
                      Logs MANUAL_INTERVENTION_REQUIRED.
        Belt-and-suspenders against any race condition the inline guards miss.
        Read-only against TWS — never places orders here; logging only.
        """
        try:
            tws_positions = self.ib.positions()
        except Exception as _e:
            log.warning(f"  reconcile: ib.positions() failed — {_e}")
            return

        # Filter to symbols this bot session tracks
        tws_by_sym: dict = {}
        for _p in tws_positions:
            _sym = _p.contract.symbol
            if _sym not in self.symbols:
                continue
            tws_by_sym[_sym] = int(_p.position)   # signed: + long, - short

        # ── Check each TWS-held symbol against bot state ──────────────────
        for _sym, _tws_qty in tws_by_sym.items():
            if abs(_tws_qty) < 1:
                continue   # TWS shows flat — handled in ghost-check loop
            _bot_pos = self.positions.get(_sym)
            if _bot_pos is None:
                # ORPHAN — TWS has position, bot doesn't know
                if self._manual_intervention.get(_sym, "").startswith("orphan_detected"):
                    continue   # already nagging
                _dir_str = "LONG" if _tws_qty > 0 else "SHORT"
                log.error(
                    f"  RECONCILE ORPHAN: {_sym} TWS shows {_tws_qty:+d}sh "
                    f"({_dir_str}) but bot has no Position record.  "
                    f"Likely cause: failed EMERGENCY_FLATTEN, manual TWS entry, "
                    f"or partial-fill race.  CLOSE MANUALLY in TWS."
                )
                tlog.info(
                    f"ORPHAN  {_dir_str:<5}  {_sym}  tws_qty={_tws_qty:+d}  "
                    f"bot_record=NONE  reason=reconcile_detected"
                )
                self._manual_intervention[_sym] = (
                    f"orphan_detected: TWS={_tws_qty:+d}sh, bot has no record. "
                    f"CLOSE MANUALLY in TWS — bot will not manage this position."
                )
            else:
                # Bot has a record — check direction + size match
                _bot_signed = (_bot_pos.shares if _bot_pos.direction == "long"
                               else -_bot_pos.shares)
                if _tws_qty != _bot_signed:
                    if self._manual_intervention.get(_sym, "").startswith("mismatch"):
                        continue
                    log.error(
                        f"  RECONCILE MISMATCH: {_sym} TWS={_tws_qty:+d}sh "
                        f"but bot tracks {_bot_signed:+d}sh "
                        f"({_bot_pos.direction.upper()} {_bot_pos.shares}sh @${_bot_pos.entry_price:.2f}).  "
                        f"REVIEW TWS — share count or direction diverged."
                    )
                    tlog.info(
                        f"MISMATCH  {_sym}  tws={_tws_qty:+d}sh  "
                        f"bot={_bot_signed:+d}sh  reason=reconcile_detected"
                    )
                    self._manual_intervention[_sym] = (
                        f"mismatch: TWS={_tws_qty:+d}sh vs bot={_bot_signed:+d}sh; "
                        f"REVIEW TWS"
                    )

        # ── Check each bot-tracked Position against TWS ───────────────────
        # GHOST = bot thinks it's open but TWS has zero shares.
        for _sym in list(self.positions.keys()):
            _bot_pos = self.positions[_sym]
            if not _bot_pos.is_open:
                continue
            _tws_qty = tws_by_sym.get(_sym, 0)
            if _tws_qty == 0:
                # GHOST — bot's position no longer in TWS (closed externally)
                _bot_signed = (_bot_pos.shares if _bot_pos.direction == "long"
                               else -_bot_pos.shares)
                log.error(
                    f"  RECONCILE GHOST: {_sym} bot tracks "
                    f"{_bot_pos.direction.upper()} {_bot_pos.shares}sh @${_bot_pos.entry_price:.2f} "
                    f"but TWS shows 0 shares.  Closed externally (STP fill, manual close, "
                    f"or external order).  AUTO-REMOVING from bot state."
                )
                # Record as exit at last-known price (best estimate available)
                _ghost_exit_px = self._rt_price(_sym) or _bot_pos.stop_price
                _now           = datetime.now()
                _bot_pos.close(_ghost_exit_px, _now, "ghost_reconciled")
                _pnl  = _bot_pos.pnl or 0.0
                _sign = "+" if _pnl >= 0 else ""
                _result = "WIN " if _pnl > 0 else ("EVEN" if _pnl == 0 else "LOSS")
                _dur  = int((_now - _bot_pos.entry_time).total_seconds() // 60)
                tlog.info(
                    f"EXIT   {_result}  {_bot_pos.direction.upper():<5}  {_sym}  "
                    f"x{_bot_pos.shares}sh  ${_bot_pos.entry_price:.2f}->${_ghost_exit_px:.2f}  "
                    f"pnl={_sign}${_pnl:.0f}  [ghost_reconciled]  held={_dur}min  "
                    f"signal={_bot_pos.entry_signal}"
                )
                self._trades_today[_sym] += 1
                self.trade_log.append({
                    "time": _now.strftime("%H:%M:%S"), "event": "exit",
                    "symbol": _sym, "direction": _bot_pos.direction,
                    "entry": _bot_pos.entry_price, "exit": _ghost_exit_px,
                    "shares": _bot_pos.shares, "pnl": _pnl,
                    "reason": "ghost_reconciled",
                    "entry_signal": _bot_pos.entry_signal,
                    "held_min": _dur,
                })
                # Clean up associated state
                self._twss_stop_orders.pop(_sym, None)
                self._manual_intervention.pop(_sym, None)
                self._exit_in_progress.discard(_sym)
                del self.positions[_sym]

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
                    # Fully tear down the old socket first.  Reconnecting with
                    # the same clientId while the stale session lingers on TWS's
                    # side triggers 'clientId already in use' rejections; an
                    # explicit disconnect avoids that.
                    try:
                        self.ib.disconnect()
                    except Exception:
                        pass
                    self.ib.sleep(1)
                    self.ib.connect(TWS_HOST, TWS_PORT, clientId=TWS_CLIENT_ID)
                    log.info("Reconnected to TWS")
                    _reconnect_attempts = 0
                    # CRITICAL: a bare reconnect leaves the bot blind — the old
                    # tickers and bar streams are dead.  Re-subscribe everything
                    # so RT stops / ratchet / EOD resume managing open positions.
                    self._resubscribe_after_reconnect()
                    # Force an immediate reconciliation so any position that
                    # changed during the outage (STP fired, etc.) is caught now
                    # rather than waiting up to 30s.
                    self._last_reconcile_dt = None
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

            # -- Deferred flatten processing (every tick) ------------------
            # EMERGENCY_FLATTEN / FILL_DRIFT place their flatten order inside
            # the fill callback but defer the confirm/adopt decision here, in
            # the safe main-loop context (no reentrant ib.sleep in callbacks).
            if self._pending_flattens:
                self._process_pending_flattens(now)

            # -- TWS reconciliation (every 30 seconds) ---------------------
            # Compares self.positions to self.ib.positions() to catch any
            # state divergence the inline guards missed.  Detects:
            #   ORPHAN  — TWS holds shares the bot doesn't know about
            #   GHOST   — bot tracks a position TWS shows as closed
            #   MISMATCH — bot/TWS disagree on size or direction
            # Read-only; no orders placed.  Flags orphans/mismatches into
            # _manual_intervention so the 10s nag below shouts loudly.
            # Ghosts are auto-removed from self.positions.
            if (self._last_reconcile_dt is None or
                    (now - self._last_reconcile_dt).total_seconds() >= 30.0):
                self._last_reconcile_dt = now
                self._reconcile_positions()

            # -- Manual-intervention nag (every 10s while any adopted) -----
            # When EMERGENCY_FLATTEN / FILL_DRIFT can't confirm the flatten
            # in 3 seconds, the bot ADOPTS the orphan as a managed Position
            # with a tight safe stop.  Operator MUST review TWS — the adopted
            # position may have unexpected size, direction, or duplicates
            # from a partial flatten.  This nag log keeps the issue visible
            # every 10 seconds until the position is properly closed.
            if self._manual_intervention and now.second % 10 == 0:
                for _ms, _mreason in list(self._manual_intervention.items()):
                    _mpos = self.positions.get(_ms)
                    _mpos_str = (f"{_mpos.direction.upper()} {_mpos.shares}sh @${_mpos.entry_price:.2f} "
                                 f"stop=${_mpos.stop_price:.2f}"
                                 if _mpos else "position not in bot state")
                    log.error(
                        f"  MANUAL_INTERVENTION_REQUIRED: {_ms} — {_mreason}.  "
                        f"Bot state: {_mpos_str}.  CHECK TWS NOW."
                    )

            # -- Real-time stop + half-exit check (fires every second) -----
            # Uses live ticker last-price — no bar delay.
            # This is the primary stop-loss mechanism now that reqMktData is
            # active.  The 1-min bar handler still updates the ratchet (needs
            # EMA50 from the 3-min bars) but the hard exit fires here first.
            # The fast ratchet (every 10 seconds, see below) also bumps the
            # stop floor between 1-min ticks so a fast reversal can't take
            # back profit you already had.
            for _sym in list(self.positions.keys()):
                _pos = self.positions.get(_sym)
                if not _pos or not _pos.is_open:
                    continue
                # Skip positions that are mid-close — running stop logic on
                # them would queue a duplicate exit MKT while the first one
                # is still in flight (5/29 PLTR 14:14 → SHORT 50 orphan bug).
                if _sym in self._exit_in_progress:
                    continue
                _rt = self._rt_price(_sym)
                if _rt is None:
                    # Ticker is fully dead AND fallback chain returned None.
                    # Try to re-subscribe ONCE — maybe the original subscription
                    # broke silently.  If even this fails we genuinely have
                    # no live price for the symbol.
                    if not hasattr(self, "_resub_attempted"):
                        self._resub_attempted = set()
                    if _sym not in self._resub_attempted:
                        self._resub_attempted.add(_sym)
                        try:
                            _contract = Stock(_sym, "SMART", "USD")
                            self.ib.qualifyContracts(_contract)
                            self.ib.cancelMktData(_contract)
                            self.ib.sleep(0.5)
                            _new_ticker = self.ib.reqMktData(_contract, '', False, False)
                            self.tickers[_sym] = _new_ticker
                            log.warning(f"  {_sym}: ticker re-subscribed (auto-recovery)")
                        except Exception as _re_e:
                            log.warning(f"  {_sym}: ticker re-sub failed — {_re_e}")
                    continue

                # -- Heartbeat: once per minute, log that the real-time loop
                # is actually processing this position.  If you don't see this
                # line in the log when you should, the per-second loop is broken
                # (or _rt_price was returning None — see warning above).
                if now.second == 0:
                    _unr_now = ((_rt - _pos.entry_price)
                                if _pos.direction == "long"
                                else (_pos.entry_price - _rt))
                    log.info(
                        f"  RT {_pos.direction.upper()} {_sym}  px=${_rt:.2f}  "
                        f"PnL={'+' if _unr_now >= 0 else ''}${_unr_now * _pos.shares:.0f}  "
                        f"stop=${_pos.stop_price:.2f}  HWM=+${_pos.best_unrealised:.2f}"
                    )

                # -- Fast ratchet — every 10 seconds, using LIVE PRICE -----
                # Tracks high-water-mark profit per share and raises the stop
                # floor accordingly.  Doesn't touch ema50 (that's the 1-min
                # bar handler's job) — purely a floor based on best_unrealised.
                # Means a $5 favorable move at 09:42:30 immediately locks
                # in $3 of it (vs waiting 30s for the 1-min bar close).
                if now.second % 10 == 0:
                    _unrealised_now = ((_rt - _pos.entry_price)
                                       if _pos.direction == "long"
                                       else (_pos.entry_price - _rt))
                    if _unrealised_now > _pos.best_unrealised:
                        _pos.best_unrealised = _unrealised_now
                    # Apply ratchet floor only once profit has cleared
                    # RATCHET_START.  Same math as compute_trailing_stop.
                    if _pos.best_unrealised >= RATCHET_START:
                        _floor_offset = max(0.0, _pos.best_unrealised - RATCHET_GIVEBACK)
                        if _pos.direction == "long":
                            _floor = _pos.entry_price + _floor_offset
                            if _floor > _pos.stop_price:
                                _old = _pos.stop_price
                                _pos.stop_price = _floor
                                log.info(
                                    f"  RT RATCHET  LONG  {_sym}  stop ${_old:.2f}->"
                                    f"${_floor:.2f}  best=+${_pos.best_unrealised:.2f}"
                                )
                        else:
                            _floor = _pos.entry_price - _floor_offset
                            if _floor < _pos.stop_price:
                                _old = _pos.stop_price
                                _pos.stop_price = _floor
                                log.info(
                                    f"  RT RATCHET  SHORT {_sym}  stop ${_old:.2f}->"
                                    f"${_floor:.2f}  best=+${_pos.best_unrealised:.2f}"
                                )

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
