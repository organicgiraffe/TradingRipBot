"""
month_backtest.py — 30-stock, 22-day synthetic backtest of the HOD Breakout strategy.

NOTE: yfinance / Alpaca data endpoints are blocked in this environment.
      This uses calibrated synthetic price data (GBM + realistic intraday patterns)
      so the strategy signal logic is fully exercised.  Replace _generate_data()
      with real Alpaca bars once credentials are available.

Run:  python month_backtest.py
"""
import sys, warnings, datetime, random
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, ".")

from config import (EMA_PERIODS, MIN_STOP_DIST, MAX_STOP_PCT,
                    MAX_TRADES_PER_DAY, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
                    MAX_RISK_PER_TRADE, MIN_SHARES)
from ema_engine import should_exit_3m, compute_trailing_stop
from hod_breakout import get_hod_signal, _print_trade

# ── Universe ───────────────────────────────────────────────────────────────────
# (price, daily_vol_pct, trend)  trend: 1=above SMA200, -1=below, 0=choppy
UNIVERSE = {
    # Index ETFs
    "SPY":  (752,  0.0065, 1),
    "QQQ":  (737,  0.0080, 1),
    "IWM":  (288,  0.0090, 0),
    # Mag7
    "NVDA": (214,  0.0280, 1),
    "TSLA": (421,  0.0310, 1),
    "AAPL": (312,  0.0120, 1),
    "META": (618,  0.0170, 1),
    "MSFT": (430,  0.0110, 1),
    "AMZN": (252,  0.0150, 1),
    "GOOGL":(359,  0.0130, 1),
    # Semis / AI
    "AVGO": (408,  0.0220, 1),
    "AMD":  (527,  0.0250, 0),
    "ARM":  (382,  0.0290, 0),
    "SMCI": (44,   0.0420, -1),
    # High-vol day-trading names
    "MSTR": (127,  0.0500, -1),
    "COIN": (163,  0.0460, -1),
    "PLTR": (145,  0.0240, 1),
    "HOOD": (62,   0.0380, 1),
    "RKLB": (23,   0.0450, -1),
    # Earnings-catalyst plays (from Rip sheet)
    "COST": (980,  0.0085, 1),
    "PANW": (277,  0.0180, 0),
    "CRWD": (667,  0.0210, 0),
    "AVGO": (408,  0.0220, 1),   # duplicate key — last wins, fine
    "NET":  (262,  0.0200, 1),
    "ROKU": (124,  0.0310, 0),
    # Misc movers
    "BA":   (212,  0.0160, -1),
    "UNH":  (386,  0.0130, 1),
    "AFRM": (61,   0.0490, -1),
    "SNAP": (12,   0.0520, -1),
    "SOFI": (14,   0.0410, 0),
}

# ── Trading calendar: 22 business days ending today ───────────────────────────
def _trading_days(n: int = 22) -> list:
    today = datetime.date(2026, 6, 4)
    days  = []
    d     = today
    while len(days) < n:
        if d.weekday() < 5:   # Mon–Fri
            days.append(d)
        d -= datetime.timedelta(days=1)
    return sorted(days)


# ── Synthetic data generator ───────────────────────────────────────────────────

def _generate_data(symbol: str, base_price: float, daily_vol: float,
                   trend: int, trading_days: list, seed: int = 0):
    """
    Returns (df_5m_all, daily_ohlc, pmh_by, pml_by).

    Price path: geometric Brownian motion with a slight trend drift.
    Intraday vol: 1.8× at the open (9:30–10:00), 0.6× at mid-day, 1.2× at close.
    Overnight gap: small most days (~N(0, daily_vol)), occasionally large (5% prob).
    Pre-market (4:00–9:30): random walk with 40% of daily vol.
    """
    rng   = np.random.default_rng(seed + abs(hash(symbol)) % 9999)
    dt    = 5 / (60 * 6.5)        # 5-min bar as fraction of trading day
    drift = trend * 0.0002         # slight directional drift per bar

    bars_per_rth   = 78            # 9:30–16:00  = 6.5 h × 12 bars/h
    bars_per_pm    = 66            # 4:00–9:29   = ~5.5 h × 12 bars/h (sparse, we use ~30)
    pm_bars        = 30

    all_5m_rows  = []
    daily_rows   = []
    pmh_by, pml_by = {}, {}

    px = base_price
    # 200D SMA: set 3% above (bullish), 3% below (bearish), or at price (choppy)
    sma200_offset = {1: -0.03, -1: 0.03, 0: 0.0}[trend]
    sma200 = px * (1 + sma200_offset)

    for day_idx, day in enumerate(trading_days):
        # ── Overnight gap ─────────────────────────────────────────────
        gap_big   = rng.random() < 0.08          # 8% chance of large gap
        gap_sigma = daily_vol * 1.5 if gap_big else daily_vol * 0.4
        gap_dir   = trend if gap_big else 0       # large gaps tend to trend direction
        gap       = rng.normal(gap_dir * daily_vol * 0.5, gap_sigma)
        px_open   = max(0.10, px * (1 + gap))

        # ── Pre-market (sparse, 30 bars from 04:00) ───────────────────
        pm_times = pd.date_range(
            datetime.datetime.combine(day, datetime.time(4, 0)),
            periods=pm_bars, freq="10min", tz="America/New_York")
        pm_px = px_open
        pm_rows = []
        for pt in pm_times:
            pm_ret = rng.normal(0, daily_vol * 0.35 * np.sqrt(dt * 2))
            pm_px  = max(0.10, pm_px * (1 + pm_ret))
            half   = pm_px * daily_vol * 0.2
            bar_h  = pm_px + abs(rng.normal(0, half))
            bar_l  = pm_px - abs(rng.normal(0, half))
            pm_rows.append({"time": pt, "Open": pm_px, "High": bar_h,
                             "Low": bar_l, "Close": pm_px, "Volume": int(rng.integers(500, 3000))})

        pm_df = pd.DataFrame(pm_rows).set_index("time")
        pmh_by[day] = float(pm_df["High"].max())
        pml_by[day] = float(pm_df["Low"].min())
        all_5m_rows.extend(pm_rows)

        # ── Regular session (78 bars, 9:30–16:00) ─────────────────────
        rth_times = pd.date_range(
            datetime.datetime.combine(day, datetime.time(9, 30)),
            periods=bars_per_rth, freq="5min", tz="America/New_York")

        rth_px = px_open
        day_high = rth_px; day_low = rth_px; day_open = rth_px
        rth_rows = []
        for b_idx, bt in enumerate(rth_times):
            # Intraday vol multiplier: high at open & close, low midday
            rel = b_idx / bars_per_rth
            vol_mult = (1.8 if rel < 0.08 else          # first 30 min
                        0.6 if 0.15 < rel < 0.70 else   # midday
                        1.2)                              # last hour
            bar_vol  = daily_vol * vol_mult * np.sqrt(dt)
            ret      = rng.normal(drift, bar_vol)
            rth_px   = max(0.10, rth_px * (1 + ret))

            spread   = rth_px * daily_vol * 0.15 * vol_mult
            bar_h    = rth_px + abs(rng.normal(0, spread))
            bar_l    = rth_px - abs(rng.normal(0, spread))
            bar_o    = rth_px * (1 + rng.normal(0, bar_vol * 0.3))
            day_high = max(day_high, bar_h)
            day_low  = min(day_low,  bar_l)
            vol      = int(rng.integers(200_000, 5_000_000) * (2.0 if rel < 0.08 else
                                                                0.5 if 0.2 < rel < 0.7 else 1.0))
            rth_rows.append({"time": bt, "Open": bar_o, "High": bar_h,
                              "Low": bar_l, "Close": rth_px, "Volume": vol})

        all_5m_rows.extend(rth_rows)
        day_close = rth_px
        px = day_close   # carry forward to next day

        daily_rows.append({"date": day, "open": day_open, "high": day_high,
                            "low": day_low, "close": day_close, "sma200": sma200})

    # ── Build DataFrames ───────────────────────────────────────────────────────
    df_5m = pd.DataFrame(all_5m_rows).set_index("time")
    df_5m.columns = [c.lower() for c in df_5m.columns]
    df_5m["hl2"] = (df_5m["high"] + df_5m["low"]) / 2

    # EMA columns — computed on RTH bars only, then merged back
    rth_mask = ((df_5m.index.time >= datetime.time(9, 30)) &
                (df_5m.index.time <= datetime.time(16, 0)))
    rth_df   = df_5m[rth_mask].copy()
    for p in EMA_PERIODS:
        rth_df[f"ema{p}"] = rth_df["hl2"].ewm(span=p, adjust=False).mean()
    rth_df["vol_ma20"] = rth_df["volume"].rolling(20).mean()
    df_5m = df_5m.join(rth_df[[f"ema{p}" for p in EMA_PERIODS] + ["vol_ma20"]])
    df_5m[df_5m.filter(like="ema").columns] = (
        df_5m.filter(like="ema").ffill())
    df_5m["vol_ma20"] = df_5m["vol_ma20"].ffill()

    daily_df = pd.DataFrame(daily_rows).set_index("date")
    return df_5m, daily_df, pmh_by, pml_by


# ── Single-symbol multi-day runner ─────────────────────────────────────────────

def _run_symbol(symbol: str, base_price: float, daily_vol: float,
                trend: int, trading_days: list, seed: int) -> list:

    df_5m, daily_df, pmh_by, pml_by = _generate_data(
        symbol, base_price, daily_vol, trend, trading_days, seed)

    trades       = []
    position     = None
    trades_today = 0
    last_date    = None

    rth = df_5m[(df_5m.index.time >= datetime.time(9, 30)) &
                (df_5m.index.time <= datetime.time(16, 0))].copy()

    for i, bar_time in enumerate(rth.index):
        bar_date = bar_time.date()

        # Reset daily counters
        if bar_date != last_date:
            trades_today = 0
            last_date    = bar_date

        t      = bar_time.time()
        is_eod = (t.hour == MARKET_CLOSE_HOUR and t.minute >= MARKET_CLOSE_MINUTE)
        no_new = (t.hour > MARKET_CLOSE_HOUR or
                  (t.hour == MARKET_CLOSE_HOUR and t.minute >= MARKET_CLOSE_MINUTE))

        df_today = rth[(rth.index.date == bar_date) & (rth.index <= bar_time)]
        cur      = df_today.iloc[-1]

        # ── Manage position ────────────────────────────────────────────
        if position:
            if is_eod:
                ep  = float(cur["close"])
                sh  = position["shares"]
                pnl = ((ep - position["entry"]) * sh if position["dir"] == "long"
                       else (position["entry"] - ep) * sh)
                trades.append({**position, "exit": ep, "exit_time": bar_time,
                               "pnl": pnl, "reason": "EOD"})
                position = None; continue

            # Use full RTH slice up to bar_time for trailing stop / cloud exit
            df_rth_now = rth[rth.index <= bar_time]
            new_stop   = compute_trailing_stop(
                df_rth_now, position["dir"], position["stop"], position["entry"])
            position["stop"] = new_stop
            sh = position["shares"]

            if position["dir"] == "long" and float(cur["low"]) <= position["stop"]:
                pnl = (position["stop"] - position["entry"]) * sh
                trades.append({**position, "exit": position["stop"],
                               "exit_time": bar_time, "pnl": pnl, "reason": "stop"})
                trades_today += 1; position = None; continue

            if position["dir"] == "short" and float(cur["high"]) >= position["stop"]:
                pnl = (position["entry"] - position["stop"]) * sh
                trades.append({**position, "exit": position["stop"],
                               "exit_time": bar_time, "pnl": pnl, "reason": "stop"})
                trades_today += 1; position = None; continue

            df_rth_now = rth[rth.index <= bar_time]
            if should_exit_3m(df_rth_now, position["dir"]):
                ep  = float(cur["close"])
                pnl = ((ep - position["entry"]) * sh if position["dir"] == "long"
                       else (position["entry"] - ep) * sh)
                trades.append({**position, "exit": ep, "exit_time": bar_time,
                               "pnl": pnl, "reason": "cloud"})
                trades_today += 1; position = None
            continue

        if no_new or trades_today >= MAX_TRADES_PER_DAY:
            continue

        # ── Entry check ────────────────────────────────────────────────
        prev_days = [d for d in daily_df.index if d < bar_date]
        if not prev_days:
            continue
        prev  = daily_df.loc[prev_days[-1]]
        pmh   = pmh_by.get(bar_date)
        pml   = pml_by.get(bar_date)

        signal, stop_price = get_hod_signal(
            bar_time, cur, df_today,
            float(prev["high"]), float(prev["low"]),
            float(prev["close"]), float(prev["sma200"]),
            pmh, pml)

        if signal == "none":
            continue

        entry_price = float(cur["close"])
        stop_dist   = abs(entry_price - stop_price)
        n_shares    = max(MIN_SHARES, int(MAX_RISK_PER_TRADE / stop_dist))
        risk        = stop_dist * n_shares

        position = {"symbol": symbol, "dir": signal,
                    "entry": entry_price, "stop": stop_price,
                    "shares": n_shares, "entry_time": bar_time, "risk": risk}

    return trades


# ── Main ───────────────────────────────────────────────────────────────────────

def run_month_backtest():
    trading_days = _trading_days(22)
    print(f"\n{'='*70}")
    print(f"  HOD BREAKOUT / LOD BREAKDOWN  —  30-STOCK  22-DAY BACKTEST")
    print(f"  Period : {trading_days[0]} → {trading_days[-1]}")
    print(f"  NOTE   : Calibrated synthetic data (yfinance blocked in env)")
    print(f"{'='*70}\n")

    all_trades   = []
    sym_results  = {}

    symbols = list(UNIVERSE.keys())[:30]   # cap at 30

    for sym in symbols:
        base_px, vol, trend = UNIVERSE[sym]
        trades = _run_symbol(sym, base_px, vol, trend, trading_days,
                             seed=42 + symbols.index(sym))
        all_trades.extend(trades)
        sym_results[sym] = trades

    # ── Per-symbol table ────────────────────────────────────────────────────────
    print(f"  {'SYM':<6}  {'TIER':<7}  {'TR':>3}  {'W':>3}  {'L':>3}  "
          f"{'WR%':>5}  {'TOT P&L':>10}  {'AVG W':>8}  {'AVG L':>8}  {'BEST':>8}")
    print(f"  {'-'*75}")

    tier_map = {1: "BULL", -1: "BEAR", 0: "CHOP"}
    for sym in symbols:
        t_list = sym_results[sym]
        if not t_list:
            print(f"  {sym:<6}  {tier_map[UNIVERSE[sym][2]]:<7}  "
                  f"{'--':>3}  --  --  {'--':>5}  {'--':>10}")
            continue
        wins   = [t for t in t_list if t["pnl"] > 0]
        losses = [t for t in t_list if t["pnl"] <= 0]
        total  = sum(t["pnl"] for t in t_list)
        wr     = len(wins) / len(t_list) * 100
        avg_w  = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_l  = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        best   = max(t["pnl"] for t in t_list)
        tier   = tier_map[UNIVERSE[sym][2]]
        print(f"  {sym:<6}  {tier:<7}  {len(t_list):>3}  "
              f"{len(wins):>3}  {len(losses):>3}  "
              f"{wr:>5.1f}%  ${total:>9.2f}  "
              f"${avg_w:>7.2f}  ${avg_l:>7.2f}  ${best:>7.2f}")

    # ── Aggregate stats ─────────────────────────────────────────────────────────
    wins_all   = [t for t in all_trades if t["pnl"] > 0]
    losses_all = [t for t in all_trades if t["pnl"] <= 0]
    total_all  = sum(t["pnl"] for t in all_trades)

    print(f"\n  {'='*70}")
    print(f"  AGGREGATE  —  {len(all_trades)} trades over {len(symbols)} symbols × 22 days")
    print(f"  {'='*70}")
    print(f"  Total P&L   : ${total_all:+,.2f}")
    print(f"  Win rate    : {len(wins_all)}/{len(all_trades)} "
          f"({len(wins_all)/len(all_trades)*100:.1f}%)" if all_trades else "  No trades")
    if wins_all:
        print(f"  Avg win     : ${sum(t['pnl'] for t in wins_all)/len(wins_all):+.2f}")
    if losses_all:
        print(f"  Avg loss    : ${sum(t['pnl'] for t in losses_all)/len(losses_all):+.2f}")
    if wins_all and losses_all:
        avg_w = sum(t["pnl"] for t in wins_all) / len(wins_all)
        avg_l = abs(sum(t["pnl"] for t in losses_all) / len(losses_all))
        print(f"  Profit factor: {sum(t['pnl'] for t in wins_all) / abs(sum(t['pnl'] for t in losses_all)):.2f}")
        print(f"  R:R ratio   : {avg_w / avg_l:.2f} : 1")

    # ── By exit reason ──────────────────────────────────────────────────────────
    print(f"\n  Exit breakdown:")
    for reason in ["stop", "cloud", "EOD"]:
        bucket = [t for t in all_trades if t["reason"] == reason]
        if not bucket:
            continue
        bw = [t for t in bucket if t["pnl"] > 0]
        bt = sum(t["pnl"] for t in bucket)
        print(f"    {reason:<8}: {len(bucket):>3} trades  "
              f"{len(bw)}/{len(bucket)} wins  ${bt:+,.2f}")

    # ── By tier ─────────────────────────────────────────────────────────────────
    print(f"\n  By daily trend tier:")
    for tier_val, tier_name in [(1, "BULL (above SMA200)"),
                                 (-1, "BEAR (below SMA200)"),
                                 (0,  "CHOP (at SMA200)")]:
        tier_syms  = [s for s in symbols if UNIVERSE[s][2] == tier_val]
        tier_trades = [t for t in all_trades if t["symbol"] in tier_syms]
        if not tier_trades:
            continue
        tw    = [t for t in tier_trades if t["pnl"] > 0]
        ttot  = sum(t["pnl"] for t in tier_trades)
        wr    = len(tw) / len(tier_trades) * 100
        print(f"    {tier_name:<26}: {len(tier_trades):>3} trades  "
              f"{wr:.1f}% WR  ${ttot:+,.2f}")

    # ── By direction ────────────────────────────────────────────────────────────
    print(f"\n  By direction:")
    for d in ["long", "short"]:
        dt = [t for t in all_trades if t["dir"] == d]
        if not dt:
            continue
        dw = [t for t in dt if t["pnl"] > 0]
        dtot = sum(t["pnl"] for t in dt)
        print(f"    {d.upper():<6}: {len(dt):>3} trades  "
              f"{len(dw)/len(dt)*100:.1f}% WR  ${dtot:+,.2f}")

    # ── Signal frequency ────────────────────────────────────────────────────────
    stock_days    = len(symbols) * 22
    signal_rate   = len(all_trades) / stock_days * 100
    print(f"\n  Signal frequency: {len(all_trades)} signals / {stock_days} "
          f"stock-days = {signal_rate:.1f}%")
    print(f"  (avg {len(all_trades)/len(symbols):.1f} trades per symbol over 22 days)")
    print(f"{'='*70}\n")

    return all_trades


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_month_backtest()
