"""
_orb_rip_week.py — legacy coarse-data ORB-10 on this week's Rip picks.

This downloads Yahoo 5-minute / 60-day data. That is NOT an exact 1-minute
opening-range source, so velez_experiment blocks ORB by default. Use only as a
labelled approximation by setting ve.ALLOW_COARSE_ORB_SOURCE = True, or rebuild
with real 1-minute data.

Stocks sourced from:
  - Jun 01 daily plan (user's 8 picks):     LLY AVGO ARM NET NVDA CRM ZS CRWV
  - Jun 02 daily plan (user's 8 picks):     MRVL DELL HPE CRDO DG NVDA MCHP IBM
  - Jun 01 Day2/Day3 highlighted:           PLTR MU SNDK COIN NOW HOOD APP RDDT
                                            OKTA AAOI IONQ RGTI ONDS ALAB NBIS
  - Jun 01 News Play highlighted:           LLY CRWV MRVL NVDA TSLA ARM AMZN NET ZS
  - Jun 02 Day2/Day3 highlighted:           ORCL RKLB ASTS CRM PLTR ESTC
  - Jun 02 News Play highlighted:           MRVL AVGO ARM NVDA TSLA DELL HPE CRDO IBM

All unique tickers = 34 stocks, all from Rip's actual sheets this week.
"""
import warnings; warnings.filterwarnings("ignore")
import yfinance as yf, pandas as pd
import velez_experiment as ve
from collections import defaultdict

RIP_WEEK = [
    # User's actual 8-stock picks (Jun 01)
    "LLY", "AVGO", "ARM", "NET", "NVDA", "CRM", "ZS", "CRWV",
    # User's actual 8-stock picks (Jun 02 — new names only)
    "MRVL", "DELL", "HPE", "CRDO", "DG", "MCHP", "IBM",
    # Day2/Day3 catalyst highlights (both days)
    "PLTR", "MU", "SNDK", "COIN", "NOW", "HOOD", "APP", "RDDT",
    "OKTA", "AAOI", "IONQ", "RGTI", "ONDS", "NBIS",
    # News Play highlights (both days — new names only)
    "TSLA", "AMZN", "ORCL", "RKLB", "ASTS",
]

# Remove any duplicates while preserving order
seen = set()
RIP_WEEK = [s for s in RIP_WEEK if not (s in seen or seen.add(s))]
print(f"Rip's week picks: {len(RIP_WEEK)} unique stocks")
print(f"  {' '.join(RIP_WEEK)}\n")

# ── Download 5-min 60d ─────────────────────────────────────────────────────
print("Downloading 5-min 60d data...")
data = {}
for s in RIP_WEEK:
    raw = yf.download(s, period="60d", interval="5m",
                      progress=False, auto_adjust=True, prepost=True)
    if raw.empty: print(f"  {s} SKIPPED"); continue
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

print(f"  {len(data)}/{len(RIP_WEEK)} loaded\n")

# ── Days ───────────────────────────────────────────────────────────────────
all_days = ve.latest_common_days(data, lookback=60)
N   = len(all_days)
mid = N // 2
days_IS  = all_days[:mid]
days_OOS = all_days[mid:]
print(f"  Total: {N} days  |  IS: {len(days_IS)} ({days_IS[0]} to {days_IS[-1]})")
print(f"                      OOS: {len(days_OOS)} ({days_OOS[0]} to {days_OOS[-1]})\n")

# ── Runner ─────────────────────────────────────────────────────────────────
def run_days(days, max_pos):
    ve.BAR_SIZE      = "5min"
    ve.ENTRY_MODE    = "orb10"
    ve.ORB_STOP_MODE = "third_from_break"
    all_t = []; day_nets = defaultdict(float)
    for d in days:
        t = ve.velez_multi(data, d, max_positions=max_pos)
        for tr in t: day_nets[d] += tr["pnl"]
        all_t.extend(t)
    if not all_t:
        return 0, 0, 0.0, 0.0, 0, 0
    wins   = [t for t in all_t if t["pnl"] > 0]
    losses = [t for t in all_t if t["pnl"] <= 0]
    net    = sum(t["pnl"] for t in all_t)
    pf     = sum(t["pnl"] for t in wins)/abs(sum(t["pnl"] for t in losses)) if losses else 999
    dw     = sum(1 for v in day_nets.values() if v > 0)
    dl     = sum(1 for v in day_nets.values() if v <= 0)
    return len(all_t), len(wins), net, pf, dw, dl

# ── Results ────────────────────────────────────────────────────────────────
print("="*72)
print("  ORB-10 | 5-min coarse-data approximation | Rip's weekly picks | 60-day IS/OOS")
print("="*72)
hdr = "  {:26} {:>5} {:>6} {:>5} {:>10} {:>5} {:>8}"
row = "  {:26} {:>5} {:>6} {:>4}% {:>+10,.0f} {:>5.2f} {:>4}W/{}L"
print(hdr.format("Config","Days","Trades","WinR","Net","PF","DayW/L"))
print("  "+"-"*70)

for slots in (2, 3):
    print(f"\n  -- Slot cap = {slots} --")
    for label, days in [
        (f"ALL {N} days",            all_days),
        (f"IS  days 1-{mid}",        days_IS),
        (f"OOS days {mid+1}-{N} <<<",days_OOS),
    ]:
        nt, nw, net, pf, dw, dl = run_days(days, slots)
        wr = 100*nw//nt if nt else 0
        print(row.format(label, len(days), nt, wr, net, pf, dw, dl))

# ── Per-day OOS ────────────────────────────────────────────────────────────
print()
print("  OOS per-day (2-slot vs 3-slot):")
print("  {:12}  {:>12}  {:>12}".format("Day","2-slot","3-slot"))
print("  "+"-"*42)
ve.BAR_SIZE="5min"; ve.ENTRY_MODE="orb10"; ve.ORB_STOP_MODE="third_from_break"
for d in days_OOS:
    t2 = ve.velez_multi(data, d, max_positions=2)
    t3 = ve.velez_multi(data, d, max_positions=3)
    n2 = sum(t["pnl"] for t in t2)
    n3 = sum(t["pnl"] for t in t3)
    flag = "  LOSING" if n2<0 and n3<0 else ("  -2slot" if n2<0 else "")
    print("  {:12}  {:>+12,.0f}  {:>+12,.0f}{}".format(d, n2, n3, flag))

# ── Per-stock OOS top/bottom ───────────────────────────────────────────────
print()
print("  Per-stock OOS (3-slot) — top 10 and bottom 5:")
by_sym = defaultdict(list)
ve.BAR_SIZE="5min"; ve.ENTRY_MODE="orb10"; ve.ORB_STOP_MODE="third_from_break"
for d in days_OOS:
    for x in ve.velez_multi(data, d, max_positions=3):
        by_sym[x["symbol"]].append(x["pnl"])

sym_stats = []
for s, pnls in by_sym.items():
    net = sum(pnls); w = sum(1 for p in pnls if p>0)
    sym_stats.append((s, len(pnls), w, net))
sym_stats.sort(key=lambda x: -x[3])

print("  {:6} {:>4} {:>4} {:>10}".format("Sym","N","W","Net"))
print("  "+"-"*28)
for s,n,w,net in sym_stats[:10]:
    print("  {:6} {:>4} {:>3}% {:>+10,.0f}".format(s,n,100*w//n if n else 0,net))
if len(sym_stats) > 10:
    print("  ...")
    for s,n,w,net in sym_stats[-5:]:
        print("  {:6} {:>4} {:>3}% {:>+10,.0f}".format(s,n,100*w//n if n else 0,net))

# ── Summary vs other baskets ───────────────────────────────────────────────
print()
print("  COMPARISON vs other baskets (OOS, 3-slot):")
print("  Random 30 (no overlap):    PF 1.90  +$26,790  22W/8L")
print("  Original coarse-data numbers invalidated; rerun with exact 1-min OR source.")
print("  Rip's actual picks:        see above ^")
