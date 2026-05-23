"""
Offline unit tests — no IBKR, no internet required.
Tests candle pattern detection and stop loss logic with synthetic data.

Run:  python test_signals.py
"""
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, ".")

from ema_engine import (is_hammer, is_shooting_star, is_doji,
                        compute_stop, get_entry_signal_3m, get_trend_10m)
from config import EMA_PERIODS, MIN_BARS_10M, MIN_BARS_3M, LARGE_CANDLE_STOP

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name: str, expected, actual):
    status = PASS if expected == actual else FAIL
    results.append(status)
    tag = f"[{status}]"
    print(f"  {tag:<8} {name}")
    if status == FAIL:
        print(f"           expected={expected!r}  got={actual!r}")


def make_row(open_, high, low, close,
             ema5=100, ema12=100, ema20=100, ema34=100, ema50=100, ema200=100):
    """Create a single bar as a pandas Series with all required fields."""
    return pd.Series({
        "open": open_, "high": high, "low": low, "close": close,
        "ema5": ema5, "ema12": ema12, "ema20": ema20,
        "ema34": ema34, "ema50": ema50, "ema200": ema200,
    })


# ------------------------------------------------------------------ #
# Hammer tests
# ------------------------------------------------------------------ #
print("\n=== Hammer detection ===")

# Classic hammer: open 100, close 100.50, high 101, low 97
# Body=0.50, lower wick=3.0 (6x body), upper wick=0.50 (1x body)  → hammer
check("Classic hammer",
      True,
      is_hammer(make_row(100, 101, 97, 100.50)))

# Candle with NO lower wick → not a hammer
check("No lower wick is NOT a hammer",
      False,
      is_hammer(make_row(97, 101, 97, 101)))

# Inverted hammer (long upper wick) → not a hammer
check("Inverted hammer (long upper wick) is NOT a hammer",
      False,
      is_hammer(make_row(100, 104, 99.5, 100.50)))

# Tiny body hammer — 0.10 body, 0.40 lower wick  → hammer
check("Tiny body hammer",
      True,
      is_hammer(make_row(100, 100.10, 99.60, 100.10)))


# ------------------------------------------------------------------ #
# Shooting star tests
# ------------------------------------------------------------------ #
print("\n=== Shooting star detection ===")

# Classic shooting star: open 100.50, close 100, high 104, low 100
# Body=0.50, upper wick=3.50 (7x body), lower wick=0  → shooting star
check("Classic shooting star",
      True,
      is_shooting_star(make_row(100.50, 104, 100, 100)))

# No upper wick → not a shooting star
check("No upper wick is NOT a shooting star",
      False,
      is_shooting_star(make_row(100, 100, 97, 97)))

# Hammer should NOT be a shooting star
check("Hammer is NOT a shooting star",
      False,
      is_shooting_star(make_row(100, 101, 97, 100.50)))


# ------------------------------------------------------------------ #
# Doji tests
# ------------------------------------------------------------------ #
print("\n=== Doji detection ===")

# Perfect doji: open == close
check("Perfect doji (open == close)",
      True,
      is_doji(make_row(100, 102, 98, 100)))

# Doji: body is 10% of range — within 15% threshold
check("Body 10% of range = doji",
      True,
      is_doji(make_row(100, 102, 98, 100.2)))  # body=0.2, range=4 = 5%

# Large body: body is 60% of range = NOT a doji
check("Body 60% of range = NOT doji",
      False,
      is_doji(make_row(100, 102, 98, 101.2)))  # body=1.2, range=4  → 30%


# ------------------------------------------------------------------ #
# Stop loss calculation tests
# ------------------------------------------------------------------ #
print("\n=== Stop loss calculation ===")

def make_df_for_stop(prev_low, prev_high, entry_close):
    """Make a 3-min DataFrame with 2 bars for stop calculation testing."""
    times = pd.date_range("2024-01-01 10:00", periods=2, freq="3min")
    rows = [
        {"open": prev_low, "high": prev_high, "low": prev_low, "close": prev_high,
         **{f"ema{p}": 100 for p in EMA_PERIODS}},
        {"open": entry_close - 0.10, "high": entry_close + 0.10,
         "low": entry_close - 0.20, "close": entry_close,
         **{f"ema{p}": 100 for p in EMA_PERIODS}},
    ]
    return pd.DataFrame(rows, index=times)


# Small candle ($2 range) — use full prev candle
df = make_df_for_stop(prev_low=98.0, prev_high=100.0, entry_close=100.50)
stop = compute_stop(df, "long", 100.50)
check("Small candle long stop = prev.low (98.0)", 98.0, stop)

df = make_df_for_stop(prev_low=100.0, prev_high=102.0, entry_close=100.0)
stop = compute_stop(df, "short", 100.0)
check("Small candle short stop = prev.high (102.0)", 102.0, stop)

# Large candle ($6 range) — use half the bar
df = make_df_for_stop(prev_low=94.0, prev_high=100.0, entry_close=100.50)
stop = compute_stop(df, "long", 100.50)
expected = 100.50 - 6.0 / 2   # 97.50
check(f"Large candle long stop = entry minus range/2 ({expected:.2f})", expected, stop)

df = make_df_for_stop(prev_low=100.0, prev_high=106.0, entry_close=100.0)
stop = compute_stop(df, "short", 100.0)
expected = 100.0 + 6.0 / 2    # 103.00
check(f"Large candle short stop = entry plus range/2 ({expected:.2f})", expected, stop)

# Exactly $5 range — NOT large, use full candle
df = make_df_for_stop(prev_low=95.0, prev_high=100.0, entry_close=100.50)
stop = compute_stop(df, "long", 100.50)
check(f"Exactly $5 candle uses full bar stop (prev.low=95.0)", 95.0, stop)


# ------------------------------------------------------------------ #
# Summary
# ------------------------------------------------------------------ #
passed = results.count(PASS)
failed = results.count(FAIL)
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed — signal logic is working correctly.")
else:
    print("Some tests FAILED — review ema_engine.py before going live.")
print("="*40)
