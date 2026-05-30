"""
test_close_position_scenarios.py — scenario tests for _close_position
exit paths.  Each test reproduces a real production bug, then asserts
the fix prevents recurrence.

Run with:  python -m pytest tests/test_close_position_scenarios.py -v

Or directly:  python tests/test_close_position_scenarios.py
"""
from __future__ import annotations

import sys
import pathlib
from datetime import datetime

# Make the project root importable when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tests.mock_ib import (MockIB, MockContract, MockOrder,
                            MockTrade, MockOrderStatus, MockEventList,
                            FillScript)
from ibkr_client import TradingBot
from position import Position


# ---------------------------------------------------------------------------
# Helpers — build a bot wired to a MockIB and seed a Position
# ---------------------------------------------------------------------------

def make_bot(symbols=None) -> tuple[TradingBot, MockIB]:
    """Create a bot with MockIB attached.  Returns (bot, ib) for assertions."""
    symbols = symbols or ["PLTR"]
    plan = {s: {"support": 100.0, "resistance": 200.0} for s in symbols}
    ib = MockIB()
    bot = TradingBot(symbols, plan, ib=ib)
    return bot, ib


def seed_position(bot: TradingBot, symbol: str, direction: str,
                  shares: int, entry_price: float, stop_price: float,
                  entry_time: datetime = None) -> Position:
    """Insert a Position directly into bot.positions, also place a mock
    crash STP in the bot's _twss_stop_orders dict.  Simulates the state
    that exists between entry-fill and exit-trigger."""
    entry_time = entry_time or datetime(2026, 5, 29, 13, 36, 2)
    pos = Position(
        symbol=symbol, direction=direction, shares=shares,
        entry_price=entry_price, entry_time=entry_time,
        stop_price=stop_price, ibkr_order_id=1001,
        entry_signal="test_seed", shares_full=shares, shares_add=0,
    )
    bot.positions[symbol] = pos

    # Create a mock crash STP trade — bot expects a Trade object in
    # _twss_stop_orders that supports .order, .orderStatus.status polling.
    stp_action = "SELL" if direction == "long" else "BUY"
    stp_order = MockOrder(
        orderId=2000, action=stp_action, totalQuantity=shares,
        orderType="STP", auxPrice=stop_price, tif="DAY",
    )
    contract = MockContract(symbol=symbol)
    stp_trade = MockTrade(
        order=stp_order, contract=contract,
        orderStatus=MockOrderStatus(status="PreSubmitted", remaining=shares),
        fillEvent=MockEventList(),
        cancelledEvent=MockEventList(),
    )
    bot.ib._open_trades[stp_order.orderId] = stp_trade
    bot._twss_stop_orders[symbol] = stp_trade
    return pos


# ---------------------------------------------------------------------------
# Scenario 1 — Clean exit (baseline happy path)
# ---------------------------------------------------------------------------

def test_clean_exit_happy_path():
    """Bot closes a position normally: crash STP cancels cleanly, exit MKT
    fills, position record removed.  Verifies the happy path still works."""
    bot, ib = make_bot(["PLTR"])
    seed_position(bot, "PLTR", "long", 50,
                  entry_price=156.22, stop_price=155.67)
    ib.set_price("PLTR", 156.00)
    ib.set_fill_script("PLTR", FillScript(delay_seconds=0))   # instant exit MKT fill

    # Bot calls _close_position when stop hits.  Mock cancel returns
    # "Cancelled" (no race) by default → bot proceeds to place exit MKT.
    bot._close_position("PLTR", price=155.67,
                        time=datetime(2026, 5, 29, 14, 14, 1), reason="stop")

    # Position should be removed from bot state
    assert "PLTR" not in bot.positions, "Position should be removed after clean exit"
    assert "PLTR" not in bot._exit_in_progress, "_exit_in_progress should be cleared"
    assert "PLTR" not in bot._twss_stop_orders, "Crash STP entry should be removed"

    # Exactly one cancel + one exit MKT order should have been placed
    cancels = [e for e in ib.events if e[0] == "cancelOrder"]
    exits   = [e for e in ib.events if e[0] == "placeOrder"
               and e[1][3] == "MKT"]
    assert len(cancels) == 1, f"Expected 1 cancelOrder, got {len(cancels)}"
    assert len(exits)   == 1, f"Expected 1 exit MKT order, got {len(exits)}"
    print("  PASS  test_clean_exit_happy_path")


# ---------------------------------------------------------------------------
# Scenario 2 — Crash STP races cancel (PLTR 14:14 bug)
# ---------------------------------------------------------------------------

def test_crash_stp_races_cancel_no_orphan():
    """Reproduce the 5/29 PLTR 14:14 bug:
       * Stop fires → bot tries to cancel crash STP
       * STP has already fired in TWS (price gapped past it)
       * Bot detects status=Filled when polling

    Pre-fix behavior: bot bailed early with _safe_to_exit=False,
    LEFT the Position in self.positions, RT loop re-triggered another
    exit MKT 1 minute later → opened SHORT 50 orphan in TWS.

    Post-fix behavior: bot detects STP-fill, closes Position record
    properly, removes from self.positions, NO second exit MKT placed."""
    bot, ib = make_bot(["PLTR"])
    seed_position(bot, "PLTR", "long", 50,
                  entry_price=156.22, stop_price=155.67)
    ib.set_price("PLTR", 155.50)

    # Configure: when bot cancels the STP, it actually FILLED instead
    # (this is the 5/29 race condition).
    ib.set_stp_script("PLTR", FillScript(stp_races_cancel=True,
                                          fill_price=155.67))

    # Simulate the bot detecting stop hit and calling _close_position
    bot._close_position("PLTR", price=155.78,
                        time=datetime(2026, 5, 29, 14, 14, 1), reason="stop")

    # CRITICAL assertions — these would all FAIL pre-fix:
    assert "PLTR" not in bot.positions, (
        "BUG: Position still in self.positions after STP-fill detection. "
        "RT loop will re-trigger another exit MKT → orphan SHORT.")
    assert "PLTR" not in bot._exit_in_progress, (
        "_exit_in_progress not cleared — RT loop will skip future entries.")
    assert "PLTR" not in bot._twss_stop_orders, (
        "Crash STP entry not removed from _twss_stop_orders.")

    # NO exit MKT should have been placed — that's the bug.  Only the
    # cancel attempt should appear (STP fill happens during cancel).
    mkt_orders = [e for e in ib.events if e[0] == "placeOrder"
                  and e[1][3] == "MKT"]
    assert len(mkt_orders) == 0, (
        f"BUG: bot placed {len(mkt_orders)} MKT order(s) after STP "
        f"already filled — would create reverse position. "
        f"Orders: {mkt_orders}")

    # An EXIT should have been logged via trade_log
    exit_log = [t for t in bot.trade_log
                if t.get("symbol") == "PLTR" and t.get("event") == "exit"]
    assert len(exit_log) == 1, f"Expected 1 exit log entry, got {len(exit_log)}"
    assert exit_log[0]["reason"] == "crash_stp_raced_us"

    print("  PASS  test_crash_stp_races_cancel_no_orphan")


# ---------------------------------------------------------------------------
# Scenario 3 — Duplicate _close_position call (race lock)
# ---------------------------------------------------------------------------

def test_duplicate_close_call_skipped():
    """Two stop-checks fire on the same symbol within milliseconds —
    common when the RT-loop and 1-min bar handler both detect the
    same stop hit.  Pre-fix: both calls executed → double SELL.
    Post-fix: _exit_in_progress lock makes the second call a no-op."""
    bot, ib = make_bot(["PLTR"])
    seed_position(bot, "PLTR", "long", 50,
                  entry_price=156.22, stop_price=155.67)
    ib.set_price("PLTR", 155.00)
    ib.set_fill_script("PLTR", FillScript(delay_seconds=10.0))   # exit slow

    # First call — claims the lock
    bot._exit_in_progress.add("PLTR")   # simulate first call mid-flight
    # Second call — should be skipped
    events_before = len(ib.events)
    bot._close_position("PLTR", price=155.00,
                        time=datetime(2026, 5, 29, 14, 14, 1), reason="stop_rt")
    events_after = len(ib.events)

    # No orders should have been placed by the second call
    new_orders = [e for e in ib.events[events_before:events_after]
                  if e[0] in ("placeOrder", "cancelOrder")]
    assert len(new_orders) == 0, (
        f"BUG: duplicate _close_position call placed {len(new_orders)} "
        f"new orders.  Lock not honored.")

    print("  PASS  test_duplicate_close_call_skipped")


# ---------------------------------------------------------------------------
# Scenario 4 — Reconciliation detects ghost
# ---------------------------------------------------------------------------

def test_reconcile_ghost_auto_cleanup():
    """Bot tracks LONG 50 PLTR but TWS shows 0 shares.  This is the
    'ghost' case — happens when an external close (manual TWS close,
    crash STP fill the bot missed) leaves bot.positions stale.
    Reconciliation should auto-remove the ghost."""
    bot, ib = make_bot(["PLTR"])
    seed_position(bot, "PLTR", "long", 50,
                  entry_price=156.22, stop_price=155.67)
    ib.set_price("PLTR", 155.50)
    # TWS has 0 shares — position closed externally
    ib.clear_position("PLTR")

    bot._reconcile_positions()

    assert "PLTR" not in bot.positions, (
        "Ghost position should have been removed by reconciliation.")
    ghost_exits = [t for t in bot.trade_log
                   if t.get("reason") == "ghost_reconciled"]
    assert len(ghost_exits) == 1, (
        f"Expected ghost_reconciled exit log entry, got {len(ghost_exits)}")
    print("  PASS  test_reconcile_ghost_auto_cleanup")


# ---------------------------------------------------------------------------
# Scenario 5 — Reconciliation detects orphan
# ---------------------------------------------------------------------------

def test_reconcile_orphan_flagged():
    """TWS shows SHORT 50 PLTR but bot has no Position record.  This is
    the 'orphan' case (e.g. EMERGENCY_FLATTEN failed to reconcile).
    Reconciliation should flag manual_intervention — does NOT auto-close
    (don't know direction intent)."""
    bot, ib = make_bot(["PLTR"])
    ib.set_price("PLTR", 155.50)
    # TWS has -50sh but bot knows nothing
    ib.set_position("PLTR", -50, avg_cost=156.00)

    bot._reconcile_positions()

    assert "PLTR" not in bot.positions, (
        "Bot should NOT have auto-created a Position record from orphan.")
    assert "PLTR" in bot._manual_intervention, (
        "Orphan should have been flagged in _manual_intervention.")
    assert "orphan_detected" in bot._manual_intervention["PLTR"]
    print("  PASS  test_reconcile_orphan_flagged")


# ---------------------------------------------------------------------------
# Scenario 6 — Reconciliation detects size mismatch
# ---------------------------------------------------------------------------

def test_reconcile_size_mismatch_flagged():
    """Bot tracks LONG 25 but TWS shows LONG 50.  Could happen if a
    partial flatten or unexpected fill changed TWS state without the
    bot knowing.  Should flag manual_intervention."""
    bot, ib = make_bot(["PLTR"])
    seed_position(bot, "PLTR", "long", 25,
                  entry_price=156.22, stop_price=155.67)
    ib.set_price("PLTR", 156.50)
    # TWS shows 50 (bot thinks 25)
    ib.set_position("PLTR", 50, avg_cost=156.22)

    bot._reconcile_positions()

    assert "PLTR" in bot._manual_intervention
    assert "mismatch" in bot._manual_intervention["PLTR"]
    print("  PASS  test_reconcile_size_mismatch_flagged")


# ---------------------------------------------------------------------------
# Runner — works with or without pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_clean_exit_happy_path,
        test_crash_stp_races_cancel_no_orphan,
        test_duplicate_close_call_skipped,
        test_reconcile_ghost_auto_cleanup,
        test_reconcile_orphan_flagged,
        test_reconcile_size_mismatch_flagged,
    ]
    passed = 0
    failed = []
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
        except Exception as e:
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {t.__name__}")
            print(f"        {type(e).__name__}: {e}")

    print()
    print("=" * 60)
    print(f"  {passed}/{len(tests)} passed")
    if failed:
        print(f"  {len(failed)} failed:")
        for name, msg in failed:
            print(f"    - {name}: {msg}")
        sys.exit(1)
