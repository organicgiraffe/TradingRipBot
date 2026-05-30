"""
test_on_fill_scenarios.py — scenario tests for the _on_fill entry callback
and the deferred EMERGENCY_FLATTEN / FILL_DRIFT flatten handling.

This is the code path that produced the 5/29 orphan trades.  Before this
suite it had ZERO coverage (the older test file only called _close_position
and _reconcile_positions directly).  These tests drive a REAL entry order
through MockIB so the fillEvent callback fires exactly as it does in
production, then assert:

  * inverted stops queue a flatten (do NOT create a Position synchronously)
  * the callback does NOT block-wait (reentrancy fix — flatten is still
    in-flight immediately after _on_fill returns)
  * confirmed flattens leave no Position
  * unconfirmed flattens ADOPT with a safe stop + manual_intervention flag

Run:  python tests/test_on_fill_scenarios.py
"""
from __future__ import annotations

import sys
import pathlib
from datetime import datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tests.mock_ib import MockIB, FillScript
from ibkr_client import TradingBot


def make_bot(symbols=None):
    symbols = symbols or ["MU"]
    plan = {s: {"support": 100.0, "resistance": 200.0} for s in symbols}
    ib = MockIB()
    bot = TradingBot(symbols, plan, ib=ib)
    return bot, ib


def drive_entry(bot, ib, symbol, direction, entry_price, stop_price,
                shares, fill_price, fill_delay=1.0):
    """Place an entry via the REAL _open_position path with a delayed fill,
    then advance the virtual clock so the fillEvent fires AFTER the callback
    is registered (mirrors production timing)."""
    ib.set_fill_script(symbol, FillScript(delay_seconds=fill_delay,
                                          fill_price=fill_price))
    bot._open_position(
        symbol, direction, entry_price=entry_price, stop_price=stop_price,
        shares=shares, time=datetime(2026, 5, 29, 10, 27, 0),
        shares_full=shares, shares_add=0,
        entry_reason="cloud_cont", level_res=200.0, level_sup=100.0,
        entry_meta={"slot": 1, "risk": 100, "trend": "bullish",
                    "sup": 100.0, "res": 200.0},
    )
    # Fill hasn't fired yet (delayed).  Advance clock → _on_fill runs.
    ib.sleep(fill_delay)


# ---------------------------------------------------------------------------
# Scenario 1 — Clean entry creates a managed Position + crash stop
# ---------------------------------------------------------------------------

def test_clean_entry_creates_position():
    bot, ib = make_bot(["MU"])
    ib.set_price("MU", 967.95)
    # LONG, stop BELOW fill → valid (not inverted), small drift
    drive_entry(bot, ib, "MU", "long",
                entry_price=967.95, stop_price=959.74,
                shares=25, fill_price=967.95)

    assert "MU" in bot.positions, "Clean entry should create a Position"
    assert bot.positions["MU"].stop_price == 959.74
    assert "MU" not in bot._pending_flattens, "No flatten should be queued"
    # Crash STP should have been placed (STP order in events)
    stp_orders = [e for e in ib.events if e[0] == "placeOrder"
                  and e[1][3] == "STP"]
    assert len(stp_orders) == 1, f"Expected 1 crash STP, got {len(stp_orders)}"
    print("  PASS  test_clean_entry_creates_position")


# ---------------------------------------------------------------------------
# Scenario 2 — Inverted stop QUEUES flatten, does NOT create Position,
#              and does NOT block-wait (reentrancy fix)
# ---------------------------------------------------------------------------

def test_inverted_stop_queues_flatten_no_blocking():
    """The 5/29 MU 10:27 case: LONG filled @963.54 but stop 967.41 is ABOVE
    fill → inverted.  _on_fill must queue a flatten and return immediately.

    KEY regression assertions for the reentrancy fix:
      * No Position created synchronously
      * Flatten is queued in _pending_flattens
      * Flatten order is still IN-FLIGHT (not yet Filled) right after
        _on_fill returns — proving the callback did NOT sleep-poll for it
    """
    bot, ib = make_bot(["MU"])
    ib.set_price("MU", 963.54)
    # Flatten (opposite SELL) should NOT auto-fill — keep it delayed so we
    # can observe it still in-flight after the callback returns.
    drive_entry(bot, ib, "MU", "long",
                entry_price=967.41, stop_price=967.41,   # stop == price → inverted
                shares=25, fill_price=963.54)

    assert "MU" not in bot.positions, (
        "BUG: inverted-stop entry created a Position — should have flattened.")
    assert "MU" in bot._pending_flattens, (
        "Inverted stop should have queued a flatten for main-loop processing.")

    # The flatten order must still be in-flight (Submitted), NOT Filled —
    # this proves _on_fill returned without sleep-polling (reentrancy fix).
    pf = bot._pending_flattens["MU"]
    assert pf["trade"] is not None, "Flatten order should have been placed"
    assert pf["trade"].orderStatus.status != "Filled", (
        "BUG: flatten already Filled inside the callback — implies _on_fill "
        "blocked/pumped events (the reentrancy bug we removed).")
    assert pf["kind"] == "emergency_flatten"
    print("  PASS  test_inverted_stop_queues_flatten_no_blocking")


# ---------------------------------------------------------------------------
# Scenario 3 — EMERGENCY_FLATTEN confirmed (flatten fills) → no Position
# ---------------------------------------------------------------------------

def test_emergency_flatten_confirmed_no_position():
    bot, ib = make_bot(["MU"])
    ib.set_price("MU", 963.54)
    drive_entry(bot, ib, "MU", "long",
                entry_price=967.41, stop_price=967.41,
                shares=25, fill_price=963.54)
    assert "MU" in bot._pending_flattens

    # Advance the clock so the (delayed) flatten order fills.
    ib.sleep(1.0)
    # Main-loop processor confirms it.
    bot._process_pending_flattens(datetime(2026, 5, 29, 10, 27, 5))

    assert "MU" not in bot._pending_flattens, "Confirmed flatten should be dropped"
    assert "MU" not in bot.positions, "Confirmed flatten must NOT leave a Position"
    assert "MU" not in bot._manual_intervention
    print("  PASS  test_emergency_flatten_confirmed_no_position")


# ---------------------------------------------------------------------------
# Scenario 4 — EMERGENCY_FLATTEN UNCONFIRMED → ADOPT with safe stop
# ---------------------------------------------------------------------------

def test_emergency_flatten_unconfirmed_adopts():
    """The 5/29 MU 12:09 orphan: flatten order never confirms.  Old code
    walked away → orphan in TWS.  New code ADOPTS: creates a managed Position
    with a tight safe stop on the CORRECT side, places a crash STP, and flags
    manual_intervention."""
    bot, ib = make_bot(["MU"])
    ib.set_price("MU", 963.54)
    drive_entry(bot, ib, "MU", "long",
                entry_price=967.41, stop_price=967.41,
                shares=25, fill_price=963.54)
    assert "MU" in bot._pending_flattens

    # Do NOT advance the clock → flatten stays Submitted (unconfirmed).
    # Process with a wall-clock time PAST the 3s deadline → forces adopt.
    # (_queue_flatten sets deadline = real datetime.now()+3s, matching the
    # real `now` the production main loop passes in.)
    bot._process_pending_flattens(datetime.now() + timedelta(seconds=5))

    assert "MU" not in bot._pending_flattens, "Should resolve out of pending"
    assert "MU" in bot.positions, "Unconfirmed flatten should ADOPT a Position"
    pos = bot.positions["MU"]
    # Safe stop for a LONG must be BELOW the fill price (correct side)
    assert pos.stop_price < 963.54, (
        f"Adopted LONG safe stop {pos.stop_price} must be below fill 963.54")
    assert abs(pos.stop_price - 963.54 * 0.995) < 0.01
    assert pos.entry_signal.endswith("_ADOPTED")
    assert pos.shares_add == 0, "Adopted position must not pyramid-add"
    assert "MU" in bot._manual_intervention, "Adopt must flag manual review"
    # Crash STP placed for the adopted position
    stp_orders = [e for e in ib.events if e[0] == "placeOrder"
                  and e[1][3] == "STP"]
    assert len(stp_orders) >= 1, "Adopt should place a crash STP"
    print("  PASS  test_emergency_flatten_unconfirmed_adopts")


# ---------------------------------------------------------------------------
# Scenario 5 — FILL_DRIFT queues a flatten
# ---------------------------------------------------------------------------

def test_fill_drift_queues_flatten():
    """Signal price 967.41 but fill 980.00 = ~1.3% drift (>1%).  Stop 960
    is on the correct side (not inverted), so the DRIFT guard — not the
    inversion guard — must fire and queue a flatten."""
    bot, ib = make_bot(["MU"])
    ib.set_price("MU", 980.00)
    drive_entry(bot, ib, "MU", "long",
                entry_price=967.41, stop_price=960.00,   # valid stop, big drift
                shares=25, fill_price=980.00)

    assert "MU" not in bot.positions, "Drifted fill should not create Position yet"
    assert "MU" in bot._pending_flattens, "Drift should queue a flatten"
    assert bot._pending_flattens["MU"]["kind"] == "fill_drift"
    print("  PASS  test_fill_drift_queues_flatten")


# ---------------------------------------------------------------------------
# Scenario 6 — pending flatten counts toward the slot limit
# ---------------------------------------------------------------------------

def test_pending_flatten_counts_toward_slot():
    """While a flatten is resolving it may yet ADOPT into a position, so it
    must occupy a slot — otherwise the bot could over-allocate (the 3-position
    bug class)."""
    bot, ib = make_bot(["MU"])
    ib.set_price("MU", 963.54)
    drive_entry(bot, ib, "MU", "long",
                entry_price=967.41, stop_price=967.41,
                shares=25, fill_price=963.54)
    assert "MU" in bot._pending_flattens

    occupied = (len(bot.positions) + len(bot._pending_entries)
                + len(bot._pending_flattens))
    assert occupied == 1, (
        f"Pending flatten should occupy 1 slot, counted {occupied}")
    print("  PASS  test_pending_flatten_counts_toward_slot")


if __name__ == "__main__":
    tests = [
        test_clean_entry_creates_position,
        test_inverted_stop_queues_flatten_no_blocking,
        test_emergency_flatten_confirmed_no_position,
        test_emergency_flatten_unconfirmed_adopts,
        test_fill_drift_queues_flatten,
        test_pending_flatten_counts_toward_slot,
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
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"  {passed}/{len(tests)} passed")
    if failed:
        for name, msg in failed:
            print(f"    - {name}: {msg}")
        sys.exit(1)
