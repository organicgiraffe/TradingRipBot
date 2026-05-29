"""
Custom backtest — MU, SNDK, HOOD, ZS, last 5 trading days.
Pure 5/12 flip strategy (no C3 filter, no 10m trend filter, no volume gate).
"""
import datetime
from week_backtest import load_symbols, sim_day, _print_trade
from config import MAX_SIMULTANEOUS_POSITIONS, MAX_RISK_PER_TRADE


def main():
    # Random 6-stock pick — different sectors, different price ranges
    symbols = ["COIN", "NVDA", "UBER", "BABA", "ABNB", "SHOP"]
    print(f"\n{'='*60}")
    print(f"  CUSTOM BACKTEST — {' | '.join(symbols)}")
    print(f"  Pure 5/12 flip — no C3, no 10m trend, no volume filter")
    print(f"  MAX_POS={MAX_SIMULTANEOUS_POSITIONS}  RISK/TRADE=~${MAX_RISK_PER_TRADE:.0f}")
    print(f"{'='*60}")

    sym_data = load_symbols(symbols)
    if not sym_data:
        print("ERROR: no data loaded")
        return

    all_dates = sorted({
        d for sd in sym_data.values()
        for d in sd["df_3m"].index.date
    })
    # YESTERDAY — 2nd-to-last available trading date
    last5 = all_dates[-2:-1] if len(all_dates) >= 2 else all_dates[-1:]

    # Empty plan — no support/resistance, no bias → pure cloud-flip
    # sim_day requires a non-empty plan to run, so give each symbol a None-level entry.
    plan = {s: {"support": None, "resistance": None, "bias": "both"} for s in symbols}

    all_trades   = []
    day_results  = []

    for target in last5:
        dow = target.strftime("%A")
        print(f"\n{'='*60}")
        print(f"  {target}  ({dow})")
        print(f"{'='*60}")
        day_trades = sim_day(target, sym_data, daily_plan=plan)

        # Show each trade
        for t in day_trades:
            _print_trade(t)

        all_trades.extend(day_trades)
        day_pnl  = sum(t["pnl"] for t in day_trades)
        exits    = [t for t in day_trades if t.get("event") == "exit"]
        day_wins = sum(1 for t in exits if t["pnl"] > 0)
        day_loss = sum(1 for t in exits if t["pnl"] <= 0)
        if day_trades:
            print(f"  Day total: {len(exits)} exits  "
                  f"{day_wins}W/{day_loss}L  ${day_pnl:+.0f}")
        else:
            print("  No trades.")
        day_results.append((target, len(exits), day_wins, day_loss, day_pnl))

    # Summary
    exits   = [t for t in all_trades if t.get("event") == "exit"]
    total   = sum(t["pnl"] for t in all_trades)
    wins    = sum(1 for t in exits if t["pnl"] > 0)
    losses  = sum(1 for t in exits if t["pnl"] <= 0)

    print(f"\n{'='*60}")
    print(f"  WEEK SUMMARY — MU SNDK HOOD ZS")
    print(f"{'='*60}")
    for (d, n, w, l, pnl) in day_results:
        bar = ("+" * w + "-" * l) if n else "."
        print(f"  {d}  {n:2d} trades  {w}W/{l}L  ${pnl:+7.0f}  {bar}")
    print(f"  {'-'*50}")
    print(f"  TOTAL          {len(exits):2d} trades  "
          f"{wins}W/{losses}L  ${total:+.0f}")
    if exits:
        print(f"  Win rate: {wins/len(exits)*100:.1f}%")
        print(f"  Avg/trade: ${total/len(exits):+.0f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
