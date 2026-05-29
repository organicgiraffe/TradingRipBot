"""Test each symbol ALONE to see if its signal would have fired today.
Run: python test_solo.py HOOD PLTR
"""
import sys
from week_backtest import load_symbols, sim_day, _print_trade

symbols = sys.argv[1:] if len(sys.argv) > 1 else ["HOOD", "PLTR"]

for sym in symbols:
    print(f"\n{'='*60}")
    print(f"  SOLO TEST: {sym}")
    print(f"{'='*60}")
    sym_data = load_symbols([sym])
    if not sym_data:
        print(f"  no data for {sym}")
        continue
    today = sorted({d for d in sym_data[sym]["df_3m"].index.date})[-1]
    plan = {sym: {"support": None, "resistance": None, "bias": "both"}}
    trades = sim_day(today, sym_data, daily_plan=plan)
    for t in trades:
        _print_trade(t)
    pnl = sum(t["pnl"] for t in trades)
    n_entries = sum(1 for t in trades if t.get("event") == "entry")
    n_exits = sum(1 for t in trades if t.get("event") == "exit")
    print(f"  {sym}: {n_entries} entries / {n_exits} exits  PNL ${pnl:+.0f}")
