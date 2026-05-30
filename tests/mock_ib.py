"""
mock_ib.py — deterministic stand-in for ib_insync.IB() used in scenario tests.

The MockIB class implements the subset of ib_insync.IB methods the bot calls
(see ibkr_client.py grep "self.ib."), with hooks to script broker-side
behavior:

    * Fill behavior per symbol: instant / delayed / partial / rejected
    * STP race conditions: STP fills before cancel confirms (the 5/29 PLTR
      14:14 bug)
    * Flatten-unconfirmed: opposite-side MKT placed but never fills (the
      5/29 MU 12:09 orphan bug)
    * TWS state divergence: positions(), portfolio() can be scripted to
      simulate orphans or ghosts

Every MockIB action is recorded in self.events for assertion in tests.
Time is controlled: sleep() advances a virtual clock; the bot's real-time
sleep is never blocked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Mock dataclasses mirroring ib_insync's surface area
# ---------------------------------------------------------------------------

@dataclass
class MockContract:
    symbol: str
    secType: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    primaryExchange: str = ""
    conId: int = 0
    localSymbol: str = ""
    tradingClass: str = ""


@dataclass
class MockOrder:
    """Mirrors ib_insync.Order for fields the bot reads/sets."""
    orderId: int = 0
    action: str = ""
    totalQuantity: int = 0
    orderType: str = "MKT"
    auxPrice: float = 0.0
    lmtPrice: float = 0.0
    tif: str = "DAY"


@dataclass
class MockOrderStatus:
    """Mirrors ib_insync.OrderStatus.  The bot polls .status for state."""
    status: str = "PendingSubmit"     # PendingSubmit→PreSubmitted→Submitted→Filled
    filled: float = 0.0
    remaining: float = 0.0
    avgFillPrice: float = 0.0


@dataclass
class MockExecution:
    avgPrice: float = 0.0
    shares: float = 0.0
    side: str = ""


@dataclass
class MockFill:
    execution: MockExecution = field(default_factory=MockExecution)


@dataclass
class MockTrade:
    """Mirrors ib_insync.Trade.  The bot reads .order, .orderStatus,
    attaches callbacks via += to fillEvent / cancelledEvent."""
    order: MockOrder = field(default_factory=MockOrder)
    contract: MockContract = field(default_factory=lambda: MockContract(""))
    orderStatus: MockOrderStatus = field(default_factory=MockOrderStatus)
    fills: list = field(default_factory=list)
    # Event lists — bot does `trade.fillEvent += callback`
    fillEvent: list = field(default_factory=list)
    cancelledEvent: list = field(default_factory=list)

    def __iadd__(self, other):
        # Not used directly — fillEvent/cancelledEvent are lists with +=
        return self


class MockEventList(list):
    """Tiny shim so `trade.fillEvent += cb` appends to the list."""
    def __iadd__(self, cb):
        self.append(cb)
        return self


@dataclass
class MockTWSPosition:
    """Mirrors ib_insync's Position return from ib.positions()."""
    contract: MockContract = field(default_factory=lambda: MockContract(""))
    position: float = 0.0       # signed: positive = long, negative = short
    avgCost: float = 0.0


@dataclass
class MockTicker:
    """Mirrors ib_insync.Ticker.  Bot reads .last, .midpoint, .close."""
    contract: MockContract = field(default_factory=lambda: MockContract(""))
    last: float = float("nan")
    bid: float = float("nan")
    ask: float = float("nan")
    close: float = float("nan")

    @property
    def midpoint(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return float("nan")


# ---------------------------------------------------------------------------
# Scripted fill behavior
# ---------------------------------------------------------------------------

@dataclass
class FillScript:
    """Configures how MockIB responds when a specific order is placed.

    Attributes:
        delay_seconds: virtual seconds to wait before transitioning to Filled.
                       0 = fill immediately on placeOrder.
        fill_price: avgFillPrice reported to the bot.  None = use order's
                    aux/lmt or last seen price for the symbol.
        partial_fill_shares: if set, fill this many shares and stop (rest
                             stays Submitted).
        reject: if True, transition to Cancelled instead of Filled.
        stp_races_cancel: only meaningful for STP orders.  If True, the STP
                          fills BEFORE any cancelOrder() call against it
                          can take effect (simulates 5/29 PLTR 14:14).
        cancel_status: final status reported when bot polls after cancelOrder.
                       Useful values: "Cancelled" (normal), "Filled" (race),
                       "PreSubmitted" (timeout — never confirms).
    """
    delay_seconds: float = 0.0
    fill_price: Optional[float] = None
    partial_fill_shares: Optional[int] = None
    reject: bool = False
    stp_races_cancel: bool = False
    cancel_status: str = "Cancelled"


# ---------------------------------------------------------------------------
# MockIB — drop-in replacement for ib_insync.IB()
# ---------------------------------------------------------------------------

class MockIB:
    """In-memory simulator implementing the IB surface area the bot uses.

    Test pattern:
        ib = MockIB()
        ib.set_price("PLTR", 156.22)
        ib.set_position("PLTR", 50, avg_cost=156.22)     # pre-existing TWS state
        ib.set_fill_script("PLTR", FillScript(delay_seconds=0))   # instant
        bot = TradingBot(["PLTR"], plan={...}, ib=ib)
        # ... drive bot, assert state

    All side effects go through these methods; tests inspect self.events
    to verify the bot called what it should have called.
    """

    def __init__(self):
        self._connected = False
        self._next_order_id = 1000
        self._tickers: dict[str, MockTicker] = {}
        self._positions: dict[str, MockTWSPosition] = {}      # signed qty
        self._fill_scripts: dict[str, FillScript] = {}        # default per symbol
        self._stp_fill_script: dict[str, FillScript] = {}     # for STP orders
        self._open_trades: dict[int, MockTrade] = {}          # orderId -> Trade
        self._virtual_time: float = 0.0                        # advanced by sleep()
        self.events: list[tuple] = []                          # (kind, payload)

        # Replace event list construction with our shim — so `+=` works
        # ib_insync Trade has these as Event objects; MockTrade uses lists.

    # ── Connection ────────────────────────────────────────────────────────
    def connect(self, host: str, port: int, clientId: int):
        self._connected = True
        self.events.append(("connect", (host, port, clientId)))

    def disconnect(self):
        self._connected = False
        self.events.append(("disconnect", None))

    def isConnected(self) -> bool:
        return self._connected

    def sleep(self, seconds: float):
        """Virtual sleep — advances internal clock, processes any pending
        order state transitions.  Does NOT actually block."""
        self._virtual_time += seconds
        self._process_pending_orders()

    # ── Contract & market data ───────────────────────────────────────────
    def qualifyContracts(self, contract):
        """Real ib_insync fills in conId etc.; mock is a no-op."""
        if not contract.conId:
            contract.conId = hash(contract.symbol) & 0xFFFFFF
        return [contract]

    def reqMktData(self, contract, *args, **kwargs) -> MockTicker:
        sym = contract.symbol
        t = self._tickers.setdefault(sym, MockTicker(contract=contract))
        t.contract = contract
        self.events.append(("reqMktData", sym))
        return t

    def cancelMktData(self, contract):
        self.events.append(("cancelMktData", contract.symbol))

    def reqHistoricalData(self, contract, *args, **kwargs):
        """Tests don't usually need bars — bot's bar handling is exercised
        directly by calling _on_new_bar_3m with synthetic bars."""
        return []

    def portfolio(self):
        """Return list of PortfolioItem-like objects.  Tests rarely use."""
        return []

    def positions(self) -> list[MockTWSPosition]:
        """Mirrors ib.positions() — returns signed TWS positions."""
        return [p for p in self._positions.values() if p.position != 0]

    # ── Order placement / cancellation ───────────────────────────────────
    def placeOrder(self, contract, order: MockOrder) -> MockTrade:
        """Place an order.  Behavior driven by per-symbol FillScript.

        For MKT orders: uses self._fill_scripts[symbol] (default: instant fill).
        For STP orders: uses self._stp_fill_script[symbol] (default: rests).
        """
        if order.orderId == 0:
            order.orderId = self._next_order_id
            self._next_order_id += 1

        sym = contract.symbol
        trade = MockTrade(
            order=order,
            contract=contract,
            orderStatus=MockOrderStatus(
                status="PendingSubmit",
                remaining=float(order.totalQuantity),
            ),
            fillEvent=MockEventList(),
            cancelledEvent=MockEventList(),
        )
        self._open_trades[order.orderId] = trade
        self.events.append(("placeOrder", (sym, order.action, order.totalQuantity,
                                            order.orderType, order.orderId)))

        # Resolve which script applies
        is_stp = (order.orderType == "STP")
        script = (self._stp_fill_script.get(sym) if is_stp
                  else self._fill_scripts.get(sym, FillScript(delay_seconds=0)))

        if is_stp and script is None:
            # STP without script = resting (waits for trigger)
            trade.orderStatus.status = "PreSubmitted"
            return trade

        if script.reject:
            trade.orderStatus.status = "Cancelled"
            for cb in trade.cancelledEvent:
                cb(trade)
            return trade

        # Schedule fill — instant if delay=0
        if script.delay_seconds <= 0:
            self._fill_trade(trade, script)
        else:
            # Mark pending; _process_pending_orders fills when sleep() catches up
            trade.orderStatus.status = "Submitted"
            trade._fill_at_virtual_time = self._virtual_time + script.delay_seconds
            trade._fill_script = script

        return trade

    def cancelOrder(self, order: MockOrder):
        """Cancel an order.  If FillScript.stp_races_cancel was set for this
        symbol's STP, the STP fills BEFORE the cancel can take effect — bot's
        next status poll will see status=Filled (the 5/29 race)."""
        trade = self._open_trades.get(order.orderId)
        self.events.append(("cancelOrder", (order.orderId,
                                             trade.contract.symbol if trade else "?")))
        if not trade:
            return

        is_stp = (order.orderType == "STP")
        script = (self._stp_fill_script.get(trade.contract.symbol)
                  if is_stp else None)

        # Race scenario: STP fires NOW instead of cancelling
        if is_stp and script and script.stp_races_cancel:
            self._fill_trade(trade, script)
            return

        # Apply scripted cancel_status (default: Cancelled)
        final = script.cancel_status if script else "Cancelled"
        trade.orderStatus.status = final
        if final == "Cancelled":
            for cb in trade.cancelledEvent:
                cb(trade)
        elif final == "Filled":
            # Sometimes cancel races a real fill
            self._fill_trade(trade, script or FillScript())

    # ── Internals ────────────────────────────────────────────────────────
    def _fill_trade(self, trade: MockTrade, script: FillScript):
        """Mark a trade as Filled, update positions, invoke callbacks."""
        order = trade.order
        sym = trade.contract.symbol

        # Determine fill price
        fill_px = script.fill_price
        if fill_px is None:
            if order.orderType == "STP":
                fill_px = order.auxPrice
            elif order.orderType == "LMT":
                fill_px = order.lmtPrice
            else:
                # MKT — use last seen price for the symbol
                t = self._tickers.get(sym)
                fill_px = (t.last if t and t.last == t.last else order.auxPrice or 0.0)

        # Partial fill?
        qty = (script.partial_fill_shares
               if script.partial_fill_shares is not None
               else order.totalQuantity)

        # Update internal TWS position
        signed_delta = qty if order.action == "BUY" else -qty
        pos = self._positions.setdefault(
            sym, MockTWSPosition(contract=trade.contract, position=0.0, avgCost=0.0))
        # Weighted-avg cost when adding to existing position; reset when crossing zero
        new_qty = pos.position + signed_delta
        if pos.position * new_qty <= 0:
            # Position closed or flipped — reset avgCost
            pos.avgCost = fill_px if new_qty != 0 else 0.0
        else:
            # Same direction — weighted average
            pos.avgCost = ((abs(pos.position) * pos.avgCost
                           + abs(signed_delta) * fill_px)
                           / (abs(pos.position) + abs(signed_delta)))
        pos.position = new_qty

        # Update trade status
        trade.orderStatus.status = "Filled" if qty == order.totalQuantity else "PartiallyFilled"
        trade.orderStatus.filled = qty
        trade.orderStatus.remaining = order.totalQuantity - qty
        trade.orderStatus.avgFillPrice = fill_px

        fill = MockFill(execution=MockExecution(
            avgPrice=fill_px, shares=qty,
            side="BOT" if order.action == "BUY" else "SLD"))
        trade.fills.append(fill)

        # Invoke fill callbacks
        for cb in trade.fillEvent:
            try:
                cb(trade, fill)
            except Exception as e:
                self.events.append(("fillEvent_exception", str(e)))

        self.events.append(("fill", (sym, order.action, qty, fill_px)))

    def _process_pending_orders(self):
        """Called from sleep() — fills any trade whose delay has elapsed."""
        for trade in list(self._open_trades.values()):
            fill_time = getattr(trade, "_fill_at_virtual_time", None)
            if fill_time is not None and self._virtual_time >= fill_time:
                script = getattr(trade, "_fill_script", FillScript())
                trade._fill_at_virtual_time = None   # don't double-fill
                self._fill_trade(trade, script)

    # ── Test setup helpers ───────────────────────────────────────────────
    def set_price(self, symbol: str, price: float):
        """Set last/close for the symbol's ticker (creates ticker if needed)."""
        t = self._tickers.setdefault(
            symbol, MockTicker(contract=MockContract(symbol)))
        t.last = price
        t.close = price

    def set_position(self, symbol: str, signed_qty: float, avg_cost: float = 0.0):
        """Pre-populate a TWS-side position.  Useful for orphan-at-startup
        scenarios where TWS has something the bot doesn't know about."""
        self._positions[symbol] = MockTWSPosition(
            contract=MockContract(symbol),
            position=float(signed_qty),
            avgCost=float(avg_cost),
        )

    def clear_position(self, symbol: str):
        """Remove a symbol from TWS positions.  Useful for ghost scenarios
        where the position closes externally (e.g. crash STP fires)."""
        self._positions.pop(symbol, None)

    def set_fill_script(self, symbol: str, script: FillScript):
        """Script how the NEXT MKT placeOrder for this symbol behaves."""
        self._fill_scripts[symbol] = script

    def set_stp_script(self, symbol: str, script: FillScript):
        """Script how the NEXT STP placeOrder for this symbol behaves.
        Most useful: stp_races_cancel=True to reproduce the 5/29 PLTR
        crash-stop race condition."""
        self._stp_fill_script[symbol] = script

    def event_names(self) -> list[str]:
        """Convenience: list of event kinds in order, for assertions."""
        return [e[0] for e in self.events]

    def orders_placed(self, symbol: str = None) -> list[tuple]:
        """Return all placeOrder events, optionally filtered by symbol."""
        out = [e[1] for e in self.events if e[0] == "placeOrder"]
        if symbol:
            out = [e for e in out if e[0] == symbol]
        return out
