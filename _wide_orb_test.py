"""
_wide_orb_test.py — Wide-basket ORB validation.

Tests ORB-10 across 30 diverse stocks (momentum, mega-cap, mid-cap, sector ETFs)
to validate the edge is stock-agnostic, not just a catalyst-watchlist artifact.

Runs two comparisons:
  1. Legacy velez_experiment 60-day coarse-data ORB section is blocked by
     default unless explicitly labelled approximate.
  2. Rip vs ORB vs Combined — on 7-day 1-min OR data (Yahoo limit for 1-min)
"""
import warnings; warnings.filterwarnings("ignore")
import io, contextlib, yfinance as yf, pandas as pd
import velez_experiment as ve
from today_backtest import run_multi_today, prepare_sym_data
from collections import defaultdict

# ── BASKET: wide variety across sectors ───────────────────────────────────
BASKET = [
    # Momentum / AI / semis
    "NVDA", "AMD", "ARM", "AVGO", "MU", "SNDK", "MRVL",
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
    # Software / cloud
    "CRM", "NOW", "PLTR", "CRWD", "NET", "ZS",
    # Healthcare / biotech
    "LLY", "MRNA", "ABBV",
    # Finance
    "JPM", "GS",
    # Energy / industrial
    "XOM", "CAT",
    # Consumer
    "NFLX", "COST",
    # Mid-cap momentum
    "HOOD", "COIN",
]

print(f"BASKET: {len(BASKET)} stocks across 9 sectors\n")

# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — velez_experiment ORB standalone (5-min & 10-min, 60 days)
# ═══════════════════════════════════════════════════════════════════════════
print("="*70)
print("  PART 1 — ORB-10 standalone (velez_experiment engine)")
print("  Management: rip_cloud  |  Stop: third_from_break")
print("="*70)

def prep_multi(symbols, interval, period):
    """Download and resample once for all symbols."""
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
        data[s] = raw.between_time("04:00", "20:00")
    return data

def build_velez_frame(raw1m, bar_size):
    agg = raw1m.resample(bar_size, label="right", closed="right").agg(
        {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    ).dropna(subset=["Close"])
    agg.columns = [c.lower() for c in agg.columns]
    agg["ema20"]  = agg["close"].ewm(span=20, adjust=False).mean()
    agg["ema200"] = agg["close"].ewm(span=200, adjust=False).mean()
    agg["hl2"]    = (agg["high"] + agg["low"]) / 2
    for p in (5, 12, 34, 50, 200):
        agg[f"rip_ema{p}"] = agg["hl2"].ewm(span=p, adjust=False).mean()
    agg["vol_ma20"] = agg["volume"].rolling(20).mean()
    return agg

def run_velez_orb(raw_data, bar_size, entry_mode, stop_mode, lookback=60):
    ve.BAR_SIZE      = bar_size
    ve.ENTRY_MODE    = entry_mode
    ve.ORB_STOP_MODE = stop_mode
    data = {s: build_velez_frame(r, bar_size) for s, r in raw_data.items()}
    days = ve.latest_common_days(data, lookback=lookback)
    all_trades = []
    day_results = defaultdict(float)
    for d in days:
        try:
            t = ve.velez_multi(data, d)
        except ValueError as e:
            print(f"  SKIPPED {bar_size} {entry_mode}: {e}")
            return 0, 0, 0, 0.0, 0.0, 0, 0
        for tr in t: day_results[d] += tr["pnl"]
        all_trades.extend(t)
    if not all_trades:
        return 0, 0, 0, 0.0, 0.0, 0, 0
    wins   = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    net    = sum(t["pnl"] for t in all_trades)
    pf     = sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses)) if losses else 999
    dw     = sum(1 for v in day_results.values() if v > 0)
    dl     = sum(1 for v in day_results.values() if v <= 0)
    return len(days), len(all_trades), len(wins), net, pf, dw, dl

print("  Downloading 5-min + 10-min data (60d)...")
raw_5m  = prep_multi(BASKET, "5m",  "60d")
raw_10m = prep_multi(BASKET, "10m", "60d")
avail_5  = list(raw_5m.keys())
avail_10 = list(raw_10m.keys())
print(f"  5-min: {len(avail_5)}/{len(BASKET)} loaded")
print(f"  10-min: {len(avail_10)}/{len(BASKET)} loaded\n")

print("  {:22} {:>5} {:>6} {:>5} {:>10} {:>5} {:>8}".format(
    "Config", "Days", "Trades", "WinR", "Net", "PF", "DayW/L"))
print("  "+"-"*64)

for label, raw, bs, em, sm in [
    ("5-min  ORB-10",    raw_5m,  "5min",  "orb10", "third_from_break"),
    ("5-min  cls+orb",   raw_5m,  "5min",  "classic_or_orb10", "third_from_break"),
    ("10-min ORB-10",    raw_10m, "10min", "orb10", "third_from_break"),
    ("10-min cls+orb",   raw_10m, "10min", "classic_or_orb10", "third_from_break"),
]:
    nd, nt, nw, net, pf, dw, dl = run_velez_orb(raw, bs, em, sm)
    wr = 100*nw//nt if nt else 0
    print("  {:22} {:>5} {:>6} {:>4}% {:>+10,.0f} {:>5.2f} {:>4}W/{}L".format(
        label, nd, nt, wr, net, pf, dw, dl))

# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Rip vs ORB vs Combined on wide basket (7-day 1-min)
# ═══════════════════════════════════════════════════════════════════════════
print()
print("="*70)
print("  PART 2 — Rip vs ORB-10 vs Combined (5-min ORB / 7-day wide basket)")
print("="*70)

print("  Downloading 1-min data (7d)...")
sym_data = prepare_sym_data(BASKET, interval="1m", period="7d",
                             entry_freq="5min", quiet=True)
avail = list(sym_data.keys())
print(f"  {len(avail)}/{len(BASKET)} loaded")

# Build setups (no Rip levels — let cloud decide direction)
setups = {s: {"support": None, "resistance": None} for s in avail}

# Find common trading days
all_dates = set()
for sd in sym_data.values():
    for d in sd["df_5m"].index.date:
        all_dates.add(str(d))
days_1m = sorted(all_dates)
print(f"  Days: {len(days_1m)} ({days_1m[0]} -> {days_1m[-1]})\n")

def run_rip(d, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        t = run_multi_today(setups, date_str=d, interval="1m", period="7d",
                            entry_freq="5min", stale_secs=360, sizing="live",
                            prefetched=sym_data, quiet=True, **kw)
    return t

configs = [
    ("Rip (baseline)",  {}),
    ("ORB-10 only",     {"orb_entry": True, "use_lost_dir": False}),
    ("Rip + ORB-10",    {"orb_entry": True}),
]

print("  {:18} {:>6} {:>5} {:>10} {:>5} {:>8}".format(
    "Config", "Trades", "WinR", "Net", "PF", "DayW/L"))
print("  "+"-"*58)

day_results_all = {}
for label, kw in configs:
    all_t = []; day_nets = defaultdict(float)
    for d in days_1m:
        t = run_rip(d, **kw)
        for x in t: day_nets[d] += x["pnl"]
        all_t.extend(t)
    wins   = [x for x in all_t if x["pnl"] > 0]
    losses = [x for x in all_t if x["pnl"] <= 0]
    net    = sum(x["pnl"] for x in all_t)
    pf     = (sum(x["pnl"] for x in wins) /
               abs(sum(x["pnl"] for x in losses))) if losses else 999
    wr     = 100*len(wins)//len(all_t) if all_t else 0
    dw     = sum(1 for v in day_nets.values() if v > 0)
    dl     = sum(1 for v in day_nets.values() if v <= 0)
    day_results_all[label] = day_nets
    print("  {:18} {:>6} {:>4}% {:>+10,.0f} {:>5.2f} {:>4}W/{}L".format(
        label, len(all_t), wr, net, pf, dw, dl))

print()
print("  Per-day:")
hdrs = [l[:14] for l, _ in configs]
print("  {:12}  {:>14}  {:>14}  {:>14}".format("Day", *hdrs))
print("  "+"-"*60)
for d in days_1m:
    vals = ["${:+,.0f}".format(day_results_all[l][d]) for l, _ in configs]
    print("  {:12}  {:>14}  {:>14}  {:>14}".format(d, *vals))

# ── Per-sector summary (rough grouping) ──────────────────────────────────
print()
print("  Per-sector (ORB-10 only, 7-day):")
sectors = {
    "AI/Semis":    ["NVDA","AMD","ARM","AVGO","MU","SNDK","MRVL"],
    "Mega-Tech":   ["AAPL","MSFT","GOOGL","META","AMZN","TSLA"],
    "Software":    ["CRM","NOW","PLTR","CRWD","NET","ZS"],
    "Healthcare":  ["LLY","MRNA","ABBV"],
    "Finance":     ["JPM","GS"],
    "Energy/Ind":  ["XOM","CAT"],
    "Consumer":    ["NFLX","COST"],
    "Mid-cap":     ["HOOD","COIN"],
}
orb_trades_by_sym = defaultdict(list)
for d in days_1m:
    t = run_rip(d, orb_entry=True, use_lost_dir=False)
    for x in t:
        orb_trades_by_sym[x["symbol"]].append(x["pnl"])

for sector, syms in sectors.items():
    s_pnl = sum(p for s in syms for p in orb_trades_by_sym.get(s, []))
    s_n   = sum(len(orb_trades_by_sym.get(s, [])) for s in syms)
    s_w   = sum(1 for s in syms for p in orb_trades_by_sym.get(s, []) if p > 0)
    if s_n:
        print("  {:12} {:>2}t  {:>3}%w  ${:+,.0f}".format(
            sector+":", s_n, 100*s_w//s_n, s_pnl))
