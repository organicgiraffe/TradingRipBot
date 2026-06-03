"""
_orb_random.py — legacy coarse-data ORB-10 random-basket script.

This downloads Yahoo 5-minute / 60-day data. That is NOT an exact 1-minute
opening-range source, so velez_experiment blocks ORB by default. Use only as a
labelled approximation by setting ve.ALLOW_COARSE_ORB_SOURCE = True, or rebuild
with real 1-minute data.

Original basket: NVDA AMD ARM AVGO MU SNDK MRVL AAPL MSFT GOOGL META AMZN
                 TSLA CRM NOW PLTR CRWD NET ZS LLY MRNA ABBV JPM GS XOM
                 CAT NFLX COST HOOD COIN

New basket (zero overlap):
  Semis/HW:   INTC QCOM TSM AMAT KLAC
  Software:   SNOW DDOG UBER SHOP OKTA
  Biotech:    PFE GILD AMGN VRTX BMY
  Finance:    BAC MS V MA BLK
  Consumer:   WMT HD NKE SBUX MCD
  Industrial: BA GE HON RTX
  Energy:     CVX COP
  ETFs:       SPY QQQ        <- stress test: does ORB work on the index itself?

60-day test: full window, IS/OOS split, 2-slot and 3-slot.
"""
import warnings; warnings.filterwarnings("ignore")
import yfinance as yf, pandas as pd
import velez_experiment as ve
from collections import defaultdict

RANDOM_BASKET = [
    # Semis / Hardware (NOT in original)
    "INTC", "QCOM", "TSM", "AMAT", "KLAC",
    # Software / Cloud (NOT in original)
    "SNOW", "DDOG", "UBER", "SHOP", "OKTA",
    # Biotech / Pharma (NOT in original)
    "PFE", "GILD", "AMGN", "VRTX", "BMY",
    # Finance (NOT in original)
    "BAC", "MS", "V", "MA", "BLK",
    # Consumer (NOT in original)
    "WMT", "HD", "NKE", "SBUX", "MCD",
    # Industrial (NOT in original)
    "BA", "GE", "HON", "RTX",
    # Energy (NOT in original)
    "CVX", "COP",
    # ETFs — stress test
    "SPY", "QQQ",
]

print(f"RANDOM BASKET ({len(RANDOM_BASKET)} stocks — zero overlap with original 30)\n")
print("  Semis:    INTC QCOM TSM AMAT KLAC")
print("  Software: SNOW DDOG UBER SHOP OKTA")
print("  Biotech:  PFE GILD AMGN VRTX BMY")
print("  Finance:  BAC MS V MA BLK")
print("  Consumer: WMT HD NKE SBUX MCD")
print("  Industr:  BA GE HON RTX")
print("  Energy:   CVX COP")
print("  ETFs:     SPY QQQ")
print()

# ── Download 5-min 60d ─────────────────────────────────────────────────────
print("Downloading 5-min 60d data...")
data = {}
for s in RANDOM_BASKET:
    raw = yf.download(s, period="60d", interval="5m",
                      progress=False, auto_adjust=True, prepost=True)
    if raw.empty: print(f"  {s} SKIPPED"); continue
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        raw = raw.tz_convert("US/Eastern")
    raw = raw.between_time("04:00", "20:00")
    agg = raw.resample("5min", label="right", closed="right").agg(
        {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    ).dropna(subset=["Close"])
    agg.columns = [c.lower() for c in agg.columns]
    agg["ema20"]  = agg["close"].ewm(span=20, adjust=False).mean()
    agg["ema200"] = agg["close"].ewm(span=200, adjust=False).mean()
    agg["hl2"]    = (agg["high"] + agg["low"]) / 2
    for p in (5, 12, 34, 50, 200):
        agg[f"rip_ema{p}"] = agg["hl2"].ewm(span=p, adjust=False).mean()
    agg["vol_ma20"] = agg["volume"].rolling(20).mean()
    data[s] = agg

print(f"  {len(data)}/{len(RANDOM_BASKET)} loaded\n")

# ── Days ───────────────────────────────────────────────────────────────────
all_days = ve.latest_common_days(data, lookback=60)
N   = len(all_days)
mid = N // 2
days_IS  = all_days[:mid]
days_OOS = all_days[mid:]
print(f"  Total days: {N}")
print(f"  IS  (1-{mid}):  {days_IS[0]}  to  {days_IS[-1]}")
print(f"  OOS ({mid+1}-{N}): {days_OOS[0]}  to  {days_OOS[-1]}\n")

# ── Runner ─────────────────────────────────────────────────────────────────
def run_days(days, max_pos):
    ve.BAR_SIZE      = "5min"
    ve.ENTRY_MODE    = "orb10"
    ve.ORB_STOP_MODE = "third_from_break"
    all_t = []; day_results = defaultdict(float)
    for d in days:
        t = ve.velez_multi(data, d, max_positions=max_pos)
        for tr in t: day_results[d] += tr["pnl"]
        all_t.extend(t)
    if not all_t:
        return 0, 0, 0.0, 0.0, 0, 0
    wins   = [t for t in all_t if t["pnl"] > 0]
    losses = [t for t in all_t if t["pnl"] <= 0]
    net    = sum(t["pnl"] for t in all_t)
    pf     = sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses)) if losses else 999
    dw     = sum(1 for v in day_results.values() if v > 0)
    dl     = sum(1 for v in day_results.values() if v <= 0)
    return len(all_t), len(wins), net, pf, dw, dl

# ── Results ────────────────────────────────────────────────────────────────
print("="*72)
print("  ORB-10 | 5-min coarse-data approximation | RANDOM basket | 60-day IS/OOS")
print("  (Zero overlap with original basket)")
print("="*72)
hdr = "  {:24} {:>5} {:>6} {:>5} {:>10} {:>5} {:>8}"
row = "  {:24} {:>5} {:>6} {:>4}% {:>+10,.0f} {:>5.2f} {:>4}W/{}L"
print(hdr.format("Config","Days","Trades","WinR","Net","PF","DayW/L"))
print("  "+"-"*68)

for slots in (2, 3):
    print(f"\n  -- Slot cap = {slots} --")
    for label, days in [
        (f"ALL {N} days",          all_days),
        (f"IS  days 1-{mid}",      days_IS),
        (f"OOS days {mid+1}-{N} <<<", days_OOS),
    ]:
        nt, nw, net, pf, dw, dl = run_days(days, slots)
        wr = 100*nw//nt if nt else 0
        print(row.format(label, len(days), nt, wr, net, pf, dw, dl))

# ── Per-day OOS breakdown ──────────────────────────────────────────────────
print()
print("  OOS per-day (2-slot vs 3-slot):")
print("  {:12}  {:>12}  {:>12}".format("Day", "2-slot", "3-slot"))
print("  "+"-"*40)
ve.BAR_SIZE="5min"; ve.ENTRY_MODE="orb10"; ve.ORB_STOP_MODE="third_from_break"
for d in days_OOS:
    t2 = ve.velez_multi(data, d, max_positions=2)
    t3 = ve.velez_multi(data, d, max_positions=3)
    n2 = sum(t["pnl"] for t in t2)
    n3 = sum(t["pnl"] for t in t3)
    flag = "  LOSING" if n2 < 0 and n3 < 0 else ("  -2slot" if n2 < 0 else "")
    print("  {:12}  {:>+12,.0f}  {:>+12,.0f}{}".format(d, n2, n3, flag))

# ── Per-sector breakdown ───────────────────────────────────────────────────
print()
print("  Sector breakdown (OOS, 3-slot):")
sectors = {
    "Semis/HW":   ["INTC","QCOM","TSM","AMAT","KLAC"],
    "Software":   ["SNOW","DDOG","UBER","SHOP","OKTA"],
    "Biotech":    ["PFE","GILD","AMGN","VRTX","BMY"],
    "Finance":    ["BAC","MS","V","MA","BLK"],
    "Consumer":   ["WMT","HD","NKE","SBUX","MCD"],
    "Industrial": ["BA","GE","HON","RTX"],
    "Energy":     ["CVX","COP"],
    "ETFs":       ["SPY","QQQ"],
}
by_sym = defaultdict(list)
for d in days_OOS:
    t3 = ve.velez_multi(data, d, max_positions=3)
    for x in t3: by_sym[x["symbol"]].append(x["pnl"])
for sec, syms in sectors.items():
    s_pnl = sum(p for s in syms for p in by_sym.get(s, []))
    s_n   = sum(len(by_sym.get(s, [])) for s in syms)
    s_w   = sum(1 for s in syms for p in by_sym.get(s, []) if p > 0)
    if s_n:
        print("  {:12} {:>3}t  {:>3}%w  ${:>+8,.0f}".format(
            sec+":", s_n, 100*s_w//s_n, s_pnl))
    else:
        print(f"  {sec}: no trades")

# ── Head-to-head vs original basket ───────────────────────────────────────
print()
print("  HEAD-TO-HEAD vs original 30-stock basket (OOS, 3-slot):")
print("  Original coarse-data numbers invalidated; rerun with exact 1-min OR source.")
