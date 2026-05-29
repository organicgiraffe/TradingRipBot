"""Test MU alone on yesterday (May 27)."""
import sys
from week_backtest import load_symbols, sim_day, _print_trade

symbols = sys.argv[1:] if len(sys.argv) > 1 else ["MU"]

sym_data = load_symbols(symbols)
all_dates = sorted({d for sd in sym_data.values() for d in sd["df_3m"].index.date})
# Yesterday = 2nd to last available date
yesterday = all_dates[-2] if len(all_dates) >= 2 else all_dates[-1]
print(f"\nTESTING {' / '.join(symbols)} — {yesterday}")

plan = {s: {"support": None, "resistance": None, "bias": "both"} for s in symbols}
trades = sim_day(yesterday, sym_data, daily_plan=plan)
for t in trades:
    _print_trade(t)

pnl = sum(t["pnl"] for t in trades)
by_sym = {}
for t in trades:
    by_sym.setdefault(t["symbol"], 0)
    by_sym[t["symbol"]] += t["pnl"]

print(f"\nResult: ${pnl:+.0f}")
for s, p in by_sym.items():
    print(f"  {s}: ${p:+.0f}")
