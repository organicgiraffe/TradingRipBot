import asyncio
import logging
from datetime import datetime
from typing import Optional

from ib_insync import IB, Stock, MarketOrder

from config import (TWS_HOST, TWS_PORT, TWS_CLIENT_ID,
                    BAR_SIZE_10M, BAR_SIZE_3M, MAX_TRADES_PER_DAY)
from ema_engine import (compute_emas, get_trend_10m,
                        get_entry_signal_3m, should_exit_3m,
                        compute_trailing_stop)
from position import Position

log = logging.getLogger(__name__)


class TradingBot:
    def __init__(self, symbols: list[str], shares: int):
        self.symbols  = [s.upper() for s in symbols]
        self.shares   = shares
        self.ib       = IB()

        self.bars_10m: dict = {}   # symbol -> BarDataList  (trend direction)
        self.bars_3m:  dict = {}   # symbol -> BarDataList  (entry + management)
        self.trend:    dict = {}   # symbol -> 'bullish' | 'bearish' | 'none'

        self.position: Optional[Position] = None
        self.trade_log: list[Position]    = []

        # Per-symbol counters — each symbol independently tracks its own daily
        # trade count and direction block.  Only self.position is global (one
        # open trade at a time across all symbols).
        self._trades_today:   dict = {s: 0    for s in self.symbols}
        self._lost_dir_today: dict = {s: None for s in self.symbols}
        self._last_trade_date = None

        # Pre-market high / low per symbol (TYPE 4 breakout signal).
        # Populated by setup_premarket_levels() before market open.
        self.pmh: dict = {s: None for s in self.symbols}
        self.pml: dict = {s: None for s in self.symbols}

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #

    async def connect(self):
        await self.ib.connectAsync(TWS_HOST, TWS_PORT, clientId=TWS_CLIENT_ID)
        log.info("Connected to IBKR TWS")

    def disconnect(self):
        self.ib.disconnect()
        log.info("Disconnected from IBKR TWS")

    # ------------------------------------------------------------------ #
    # Subscriptions
    # ------------------------------------------------------------------ #

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

            log.info(f"{symbol}: {len(b10)} x 10-min, {len(b3)} x 3-min bars loaded")

    def _make_handler(self, symbol: str, tf: str):
        def on_bar(bars, has_new_bar):
            if has_new_bar:
                if tf == "10m":
                    self._on_new_bar_10m(symbol, bars)
                else:
                    self._on_new_bar_3m(symbol, bars)
        return on_bar

    # ------------------------------------------------------------------ #
    # Pre-market levels  (call once before 09:30 ET each morning)
    # ------------------------------------------------------------------ #

    def setup_premarket_levels(self):
        """
        Request 1-min pre-market bars from IBKR for each symbol and compute
        the session's pre-market high (PMH) and low (PML).
        Call this after connect() but before the regular session opens.
        """
        import datetime
        for symbol in self.symbols:
            try:
                contract = Stock(symbol, "SMART", "USD")
                self.ib.qualifyContracts(contract)
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime="",          # up to now (pre-market)
                    durationStr="1 D",
                    barSizeSetting="1 min",
                    whatToShow="TRADES",
                    useRTH=False,            # include pre/after-market
                    keepUpToDate=False,
                )
                if not bars:
                    log.warning(f"{symbol}: no pre-market bars returned")
                    continue

                import pandas as pd
                df = pd.DataFrame({
                    "time":  [b.date for b in bars],
                    "high":  [b.high for b in bars],
                    "low":   [b.low  for b in bars],
                })
                df["time"] = pd.to_datetime(df["time"])
                if df["time"].dt.tz is None:
                    df["time"] = df["time"].dt.tz_localize("US/Eastern")
                else:
                    df["time"] = df["time"].dt.tz_convert("US/Eastern")
                df = df.set_index("time")

                pre = df.between_time("04:00", "09:29")
                if pre.empty:
                    log.warning(f"{symbol}: pre-market bars empty after time filter")
                    continue

                self.pmh[symbol] = float(pre["high"].max())
                self.pml[symbol] = float(pre["low"].min())
                log.info(f"{symbol} pre-market: H=${self.pmh[symbol]:.2f}"
                         f"  L=${self.pml[symbol]:.2f}"
                         f"  range=${self.pmh[symbol] - self.pml[symbol]:.2f}")
            except Exception as e:
                log.warning(f"{symbol}: pre-market level error — {e}")

    # ------------------------------------------------------------------ #
    # 10-min handler — trend direction only
    # ------------------------------------------------------------------ #

    def _on_new_bar_10m(self, symbol: str, bars):
        df = compute_emas(list(bars))
        prev = self.trend[symbol]
        self.trend[symbol] = get_trend_10m(df)
        if self.trend[symbol] != prev:
            log.info(f"{symbol} 10m trend: {prev} -> {self.trend[symbol]}")

    # ------------------------------------------------------------------ #
    # 3-min handler — entry + trailing stop + cloud exit
    # ------------------------------------------------------------------ #

    def _on_new_bar_3m(self, symbol: str, bars):
        df_3m = compute_emas(list(bars))
        now   = datetime.now()
        cur   = df_3m.iloc[-1]

        today = now.date()
        if today != self._last_trade_date:
            self._trades_today   = {s: 0    for s in self.symbols}
            self._lost_dir_today = {s: None for s in self.symbols}
            self._last_trade_date = today

        # ---- Manage open position --------------------------------- #
        if (self.position and self.position.is_open
                and self.position.symbol == symbol):

            # Update trailing stop
            new_stop = compute_trailing_stop(
                df_3m, self.position.direction,
                self.position.stop_price, self.position.entry_price
            )
            self.position.update_stop(new_stop)

            # Live P&L display every bar
            self._print_live_pnl(cur, now)

            # Stop hit?
            if (self.position.direction == "long"
                    and cur.low <= self.position.stop_price):
                self._close_position(self.position.stop_price, now, "trailing stop")
                return
            if (self.position.direction == "short"
                    and cur.high >= self.position.stop_price):
                self._close_position(self.position.stop_price, now, "trailing stop")
                return

            # 3-min cloud flip?
            if should_exit_3m(df_3m, self.position.direction):
                self._close_position(cur.close, now, "3m cloud flip")
            return

        # ---- Entry ------------------------------------------------ #
        if self.position is not None:
            return   # different symbol in play

        if self._trades_today[symbol] >= MAX_TRADES_PER_DAY:
            return

        trend = self.trend.get(symbol, "none")
        signal, stop_price = get_entry_signal_3m(
            df_3m, trend, bar_time=now,
            pmh=self.pmh.get(symbol), pml=self.pml.get(symbol))
        if signal == "none":
            return

        # Block same-direction re-entry on THIS symbol after a loss today
        if signal == self._lost_dir_today[symbol]:
            return

        entry_price = cur.close
        log.info(f"ENTRY {signal.upper()} {symbol} @ {entry_price:.2f}  "
                 f"stop={stop_price:.2f}  "
                 f"[trade {self._trades_today[symbol] + 1}/{MAX_TRADES_PER_DAY}]")
        self._open_position(symbol, signal, entry_price, stop_price, now)

    # ------------------------------------------------------------------ #
    # Order helpers
    # ------------------------------------------------------------------ #

    def _print_live_pnl(self, cur, now: datetime):
        """Compact one-line P&L update printed every 3-min bar while in a trade."""
        p = self.position
        price = cur.close
        if p.direction == "long":
            unrealised = (price - p.entry_price) * p.shares
            stop_dist  = price - p.stop_price
        else:
            unrealised = (p.entry_price - price) * p.shares
            stop_dist  = p.stop_price - price

        sign   = "+" if unrealised >= 0 else ""
        locked = " [LOCKED]" if unrealised >= 0 and abs(p.stop_price - p.entry_price) < 0.01 else ""
        print(
            f"  {now.strftime('%H:%M')}  {p.direction.upper()} {p.symbol}"
            f"  entry={p.entry_price:.2f}  now={price:.2f}"
            f"  PnL={sign}${unrealised:.0f}"
            f"  stop={p.stop_price:.2f} ({stop_dist:.2f} away)"
            f"{locked}"
        )

    def _open_position(self, symbol: str, direction: str,
                       entry_price: float, stop_price: float, time: datetime):
        action   = "BUY" if direction == "long" else "SELL"
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        trade = self.ib.placeOrder(contract, MarketOrder(action, self.shares))

        self.position = Position(
            symbol=symbol, direction=direction, shares=self.shares,
            entry_price=entry_price, entry_time=time,
            stop_price=stop_price,
            ibkr_order_id=trade.order.orderId,
        )
        log.info(f"OPENED  {self.position.summary()}")

    def _close_position(self, price: float, time: datetime, reason: str = ""):
        if not self.position:
            return
        action   = "SELL" if self.position.direction == "long" else "BUY"
        contract = Stock(self.position.symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        self.ib.placeOrder(contract, MarketOrder(action, self.shares))

        sym = self.position.symbol
        self.position.close(price, time, reason)
        self._trades_today[sym] += 1

        log.info(f"CLOSED  {self.position.summary()}")
        if self.position.pnl is not None and self.position.pnl < 0:   # scratch = not a loss
            self._lost_dir_today[sym] = self.position.direction
            log.info(f"Loss on {sym} — blocking {self._lost_dir_today[sym]} re-entries for {sym} today.")
        if self._trades_today[sym] >= MAX_TRADES_PER_DAY:
            log.info(f"{sym} daily limit reached ({MAX_TRADES_PER_DAY} trades). Done for {sym} today.")
        self.trade_log.append(self.position)
        self.position = None

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    async def run(self):
        await self.connect()
        await self.subscribe_bars()
        log.info("Bot live — watching: " + ", ".join(self.symbols))
        await asyncio.sleep(float("inf"))

    def print_session_summary(self):
        print("\n===== SESSION SUMMARY =====")
        if not self.trade_log:
            print("No completed trades today.")
            return
        total_pnl = 0.0
        for p in self.trade_log:
            print(f"  {p.summary()}")
            if p.pnl is not None:
                total_pnl += p.pnl
        print(f"  TOTAL P&L: ${total_pnl:+.2f}")
        print("===========================\n")
