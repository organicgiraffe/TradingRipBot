"""
_orb_oos.py — legacy coarse-data ORB-10 validation script.

This script downloads Yahoo 5-minute / 60-day data. That is NOT an exact
1-minute opening-range source, so velez_experiment blocks ORB by default.
Use only as a labelled approximation by setting
ve.ALLOW_COARSE_ORB_SOURCE = True, or rebuild with real 1-minute data.

Legacy split, only valid if deliberately run as a labelled approximation:
  Days 1-30  = in-sample  (tune/observe)
  Days 31-60 = out-of-sample

Also sweeps MAX_SIMULTANEOUS_POSITIONS: 2 vs 3
"""
import warnings; warnings.filterwarnings("ignore")
import yfinance as yf, pandas as pd
import velez_experiment as ve
from collections import defaultdict

BASKET = [
    "NVDA","AMD","ARM","AVGO","MU","SNDK","MRVL",
    "AAPL","MSFT","GOOGL","META","AMZN","TSLA",
    "CRM","NOW","PLTR","CRWD","NET","ZS",
    "LLY","MRNA","ABBV",
    "JPM","GS",
    "XOM","CAT",
    "NFLX","COST",
    "HOOD","COIN",
]

def prep_all(symbols, interval, period):
    data = {}
    for s in symbols:
        raw = yf.download(s, period=period, interval=interval,
                          progress=False, auto_adjust=True, prepost=True)
        if raw.empty: continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            raw = raw.tz_convert("US/Eastern")
        raw = raw.between_time("04:00","20:00")

        agg = raw.resample("5min", label="right", closed="right").agg(
            {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
        ).dropna(subset=["Close"])
        agg.columns = [c.lower() for c in agg.columns]
        agg["ema20"]  = agg["close"].ewm(span=20, adjust=False).mean()
        agg["ema200"] = agg["close"].ewm(span=200, adjust=False).mean()
        agg["hl2"]    = (agg["high"] + agg["low"]) / 2
        for p in (5,12,34,50,200):
            agg[f"rip_ema{p}"] = agg["hl2"].ewm(span=p, adjust=False).mean()
        agg["vol_ma20"] = agg["volume"].rolling(20).mean()
        data[s] = agg
    return data

def run_days(data, days, max_pos):
    """Run ORB-10 on a list of days with given slot cap."""
    ve.BAR_SIZE      = "5min"
    ve.ENTRY_MODE    = "orb10"
    ve.ORB_STOP_MODE = "third_from_break"
    all_trades = []; day_results = defaultdict(float)
    for d in days:
        t = ve.velez_multi(data, d, max_positions=max_pos)
        for tr in t: day_results[d] += tr["pnl"]
        all_trades.extend(t)
    if not all_trades:
        return 0, 0, 0.0, 0.0, 0, 0
    wins   = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    net    = sum(t["pnl"] for t in all_trades)
    pf     = sum(t["pnl"] for t in wins)/abs(sum(t["pnl"] for t in losses)) if losses else 999
    dw     = sum(1 for v in day_results.values() if v > 0)
    dl     = sum(1 for v in day_results.values() if v <= 0)
    return len(all_trades), len(wins), net, pf, dw, dl

# ── Download ───────────────────────────────────────────────────────────────
print("Downloading 5-min 60d data for 30 stocks...")
data = prep_all(BASKET, "5m", "60d")
avail = list(data.keys())
print(f"  {len(avail)}/30 loaded\n")

# ── Get sorted trading days ────────────────────────────────────────────────
all_days = ve.latest_common_days(data, lookback=60)
N = len(all_days)
mid = N // 2
days_IS  = all_days[:mid]   # days 1-30 (in-sample)
days_OOS = all_days[mid:]   # days 31-60 (out-of-sample)
print(f"  Total days: {N}  |  IS: {len(days_IS)} ({days_IS[0]} to {days_IS[-1]})")
print(f"                      OOS: {len(days_OOS)} ({days_OOS[0]} to {days_OOS[-1]})\n")

# ── Main table ─────────────────────────────────────────────────────────────
print("="*74)
print("  ORB-10  |  5-min coarse-data approximation  |  30 stocks")
print("="*74)
hdr = "  {:22} {:>5} {:>6} {:>5} {:>10} {:>5} {:>8}"
row = "  {:22} {:>5} {:>6} {:>4}% {:>+10,.0f} {:>5.2f} {:>4}W/{}L"
print(hdr.format("Config","Days","Trades","WinR","Net","PF","DayW/L"))
print("  "+"-"*68)

for slots in (2, 3):
    print(f"\n  -- Slot cap = {slots} ------------------------------------------")
    # Full 60 days
    nt,nw,net,pf,dw,dl = run_days(data, all_days, slots)
    wr = 100*nw//nt if nt else 0
    print(row.format(f"ALL 60 days", N, nt, wr, net, pf, dw, dl))

    # In-sample (days 1-30)
    nt,nw,net,pf,dw,dl = run_days(data, days_IS, slots)
    wr = 100*nw//nt if nt else 0
    print(row.format(f"IS  days 1-{mid}", len(days_IS), nt, wr, net, pf, dw, dl))

    # Out-of-sample (days 31-60) — approximate if coarse source is enabled
    nt,nw,net,pf,dw,dl = run_days(data, days_OOS, slots)
    wr = 100*nw//nt if nt else 0
    print(row.format(f"OOS days {mid+1}-{N}  <<<", len(days_OOS), nt, wr, net, pf, dw, dl))

# ── Per-day OOS breakdown (2-slot and 3-slot side by side) ───────────────
print()
print("  OOS per-day breakdown (days 31-60):")
print("  {:12}  {:>14}  {:>14}".format("Day","2-slot NET","3-slot NET"))
print("  "+"-"*44)

ve.BAR_SIZE="5min"; ve.ENTRY_MODE="orb10"; ve.ORB_STOP_MODE="third_from_break"
for d in days_OOS:
    t2 = ve.velez_multi(data, d, max_positions=2)
    t3 = ve.velez_multi(data, d, max_positions=3)
    n2 = sum(t["pnl"] for t in t2)
    n3 = sum(t["pnl"] for t in t3)
    flag = "  <-- LOSING" if n2 < 0 and n3 < 0 else ("  <-- loss(2slot)" if n2 < 0 else "")
    print("  {:12}  {:>+14,.0f}  {:>+14,.0f}{}".format(d, n2, n3, flag))
