"""Diagnose ONDS — check avg_range_by computation and DTR ratio."""
from week_backtest import load_symbols
from ema_engine import compute_dtr_atr_ratio

sym_data = load_symbols(["ONDS"])
df10 = sym_data["ONDS"]["df_10m"]
all_dates = sorted({d for d in df10.index.date})
today = all_dates[-1]

print(f"\nONDS — today = {today}")

# Replicate avg_range_by computation
recent_ranges = [
    float(g["high"].max() - g["low"].min())
    for d, g in df10.groupby(df10.index.date)
    if d < today
]
print(f"\nPrior-day ranges ({len(recent_ranges)} days, last 10):")
for r in recent_ranges[-10:]:
    print(f"  ${r:.3f}")

last5 = recent_ranges[-5:]
avg_range = sum(last5) / len(last5) if last5 else 0.0
print(f"\n5-day avg range: ${avg_range:.3f}")

# Today's price
df3 = sym_data["ONDS"]["df_3m"]
today_bars = df3[df3.index.date == today]
price = float(today_bars["close"].iloc[-1])
print(f"Latest close: ${price:.2f}")
print(f"Range as % of price: {avg_range/price*100:.2f}%")

# DTR check
import datetime
import pandas as pd
df10_today = df10[df10.index <= pd.Timestamp(f"{today} 09:36").tz_localize("US/Eastern")]
dtr_ratio = compute_dtr_atr_ratio(df10_today, today, bar_time=df10_today.index[-1])
print(f"\nDTR ratio at 09:36: {dtr_ratio:.0%}")

# Exemption logic
from config import DTR_EXEMPT_ATR, MIN_DAILY_RANGE, MIN_DAILY_RANGE_PCT
vol_pct = avg_range / price
dtr_exempt = (avg_range >= DTR_EXEMPT_ATR) or (vol_pct >= 0.05)
print(f"\nDTR_EXEMPT_ATR = ${DTR_EXEMPT_ATR}  →  avg_range {avg_range:.2f} >= {DTR_EXEMPT_ATR}? {avg_range >= DTR_EXEMPT_ATR}")
print(f"vol_pct {vol_pct*100:.1f}% >= 5.0%? {vol_pct >= 0.05}")
print(f"DTR EXEMPT? {dtr_exempt}")

print(f"\nMIN_DAILY_RANGE = ${MIN_DAILY_RANGE}  →  avg_range >= floor? {avg_range >= MIN_DAILY_RANGE}")
print(f"MIN_DAILY_RANGE_PCT = {MIN_DAILY_RANGE_PCT*100}%  →  vol_pct {vol_pct*100:.2f}% >= pct? {vol_pct >= MIN_DAILY_RANGE_PCT}")

# Final verdict
print("\n--- WOULD ONDS PASS THE FILTERS? ---")
if avg_range < MIN_DAILY_RANGE:
    print(f"BLOCKED by penny-stock floor: avg_range ${avg_range:.2f} < ${MIN_DAILY_RANGE}")
elif vol_pct < MIN_DAILY_RANGE_PCT:
    print(f"BLOCKED by % range filter: {vol_pct*100:.2f}% < {MIN_DAILY_RANGE_PCT*100}%")
elif dtr_ratio > 0.75 and not dtr_exempt:
    print(f"BLOCKED by DTR filter: {dtr_ratio:.0%} > 75% and not exempt")
else:
    print("PASSES all range / DTR filters")
