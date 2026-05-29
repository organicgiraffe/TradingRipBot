"""Why isn't PLTR triggering?"""
import pandas as pd
from week_backtest import load_symbols
from ema_engine import get_entry_signal_3m, get_open_cloud_break_signal_3m, get_gap_signal_3m, get_trend_10m, compute_dtr_atr_ratio

sym_data = load_symbols(["PLTR"])
df3 = sym_data["PLTR"]["df_3m"]
df10 = sym_data["PLTR"]["df_10m"]
today = sorted({d for d in df3.index.date})[-1]
pmh = sym_data["PLTR"]["pmh_by"].get(today)
pml = sym_data["PLTR"]["pml_by"].get(today)

print(f"PLTR  today={today}  PMH=${pmh}  PML=${pml}")

# Check signal at each early bar
times_to_check = [
    pd.Timestamp(f"{today} 09:33").tz_localize("US/Eastern"),
    pd.Timestamp(f"{today} 09:36").tz_localize("US/Eastern"),
    pd.Timestamp(f"{today} 09:39").tz_localize("US/Eastern"),
    pd.Timestamp(f"{today} 09:42").tz_localize("US/Eastern"),
    pd.Timestamp(f"{today} 09:45").tz_localize("US/Eastern"),
    pd.Timestamp(f"{today} 09:48").tz_localize("US/Eastern"),
    pd.Timestamp(f"{today} 09:51").tz_localize("US/Eastern"),
    pd.Timestamp(f"{today} 10:00").tz_localize("US/Eastern"),
    pd.Timestamp(f"{today} 10:15").tz_localize("US/Eastern"),
    pd.Timestamp(f"{today} 10:30").tz_localize("US/Eastern"),
]

for bt in times_to_check:
    df3_now = df3[df3.index <= bt]
    df10_now = df10[df10.index <= bt]
    if df3_now.empty or df10_now.empty:
        continue
    cur = df3_now.iloc[-1]
    prev = df3_now.iloc[-2] if len(df3_now) >= 2 else cur
    trend = get_trend_10m(df10_now)
    dtr = compute_dtr_atr_ratio(df10_now, today, bar_time=bt)
    c2 = "GRN" if cur.ema5 > cur.ema12 else "RED"
    c3 = "GRN" if cur.ema34 > cur.ema50 else "RED"

    # Try each signal
    s1, _, r1 = get_open_cloud_break_signal_3m(df3_now, bar_time=bt)
    s2, _, r2 = get_gap_signal_3m(df3_now, bar_time=bt, pmh=pmh)
    s3, _, r3 = get_entry_signal_3m(df3_now, trend, bar_time=bt, pmh=pmh, pml=pml)

    print(f"  {bt.time()}  px={cur.close:.2f}  C2:{c2} C3:{c3}  trend={trend}  DTR={dtr:.0%}  "
          f"ocb={s1}({r1})  gap={s2}({r2})  flip={s3}({r3})")
