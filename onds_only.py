"""ONDS-only test — figure out why the bot missed today's 22% move."""
from week_backtest import load_symbols, sim_day, _print_trade
from config import MAX_SIMULTANEOUS_POSITIONS

symbols = ["ONDS"]
sym_data = load_symbols(symbols)
all_dates = sorted({d for sd in sym_data.values() for d in sd["df_3m"].index.date})
today = all_dates[-1]
print(f"\nTESTING ONDS ALONE — {today}")
print(f"MAX_POS={MAX_SIMULTANEOUS_POSITIONS}")

plan = {"ONDS": {"support": None, "resistance": None, "bias": "both"}}
trades = sim_day(today, sym_data, daily_plan=plan)
for t in trades:
    _print_trade(t)

exits = [t for t in trades if t.get("event") == "exit"]
total = sum(t["pnl"] for t in trades)
print(f"\nONDS-only result: {len(trades)} events, ${total:+.0f}")

# Diagnostic — what do ONDS bars actually look like?
df3 = sym_data["ONDS"]["df_3m"]
today_bars = df3[df3.index.date == today]
print(f"\nONDS today has {len(today_bars)} 3m bars")
print(f"Price range: ${today_bars['low'].min():.2f} - ${today_bars['high'].max():.2f}")
print(f"Open: ${today_bars['open'].iloc[0]:.2f}  Close: ${today_bars['close'].iloc[-1]:.2f}")
print(f"\nFirst 10 bars (with EMAs):")
print(today_bars[["open","high","low","close","ema5","ema12"]].head(10).to_string())

# Show bars around the 9:30 open and look for flips
import pandas as pd
print(f"\n9:30-10:30 bars (RTH morning):")
mask = (today_bars.index.time >= pd.Timestamp("09:30").time()) & \
       (today_bars.index.time <= pd.Timestamp("10:30").time())
morning = today_bars[mask].copy()
morning["c2_state"] = (morning["ema5"] > morning["ema12"]).map({True: "GRN", False: "RED"})
morning["c3_state"] = (morning["ema34"] > morning["ema50"]).map({True: "GRN", False: "RED"})
print(morning[["close","ema5","ema12","ema34","ema50","c2_state","c3_state","volume"]].to_string())

# Detect any 5/12 flips
print("\nFLIPS DETECTED:")
for i in range(1, len(morning)):
    p = morning.iloc[i-1]
    c = morning.iloc[i]
    if p.ema5 <= p.ema12 and c.ema5 > c.ema12:
        print(f"  {morning.index[i]}  LONG FLIP   px={c.close:.2f}  ema5={c.ema5:.3f}  ema12={c.ema12:.3f}  stop_dist={c.close-c.ema12:.3f}  pct={(c.close-c.ema12)/c.close*100:.2f}%")
    if p.ema5 >= p.ema12 and c.ema5 < c.ema12:
        print(f"  {morning.index[i]}  SHORT FLIP  px={c.close:.2f}  ema5={c.ema5:.3f}  ema12={c.ema12:.3f}")
