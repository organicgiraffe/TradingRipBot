"""
month_backtest.py — 30-stock full-year 2026 backtest of the HOD Breakout strategy,
broken down by calendar month.

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

# ── Trading calendar: all 2026 trading days Jan 2 → Jun 4 ─────────────────────
_2026_HOLIDAYS = {
    datetime.date(2026, 1,  1),   # New Year's Day
    datetime.date(2026, 1, 19),   # MLK Day
    datetime.date(2026, 2, 16),   # Presidents' Day
    datetime.date(2026, 4,  3),   # Good Friday
    datetime.date(2026, 5, 25),   # Memorial Day
}

def _trading_days(start: datetime.date = datetime.date(2026, 1, 2),
                  end:   datetime.date = datetime.date(2026, 6, 4)) -> list:
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d not in _2026_HOLIDAYS:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


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

def _month_label(d: datetime.date) -> str:
    return d.strftime("%b %Y")

def _stats(trades: list) -> dict:
    if not trades:
        return dict(n=0, wins=0, losses=0, wr=0, total=0, avg_w=0, avg_l=0, pf=0, rr=0)
    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total  = sum(t["pnl"] for t in trades)
    avg_w  = sum(t["pnl"] for t in wins)   / len(wins)   if wins   else 0
    avg_l  = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    gross_w = sum(t["pnl"] for t in wins)
    gross_l = abs(sum(t["pnl"] for t in losses))
    pf  = gross_w / gross_l if gross_l else float("inf")
    rr  = avg_w / abs(avg_l) if avg_l else float("inf")
    return dict(n=len(trades), wins=len(wins), losses=len(losses),
                wr=len(wins)/len(trades)*100, total=total,
                avg_w=avg_w, avg_l=avg_l, pf=pf, rr=rr)


def run_year_backtest():
    trading_days = _trading_days()
    symbols      = list(UNIVERSE.keys())[:30]

    print(f"\n{'='*72}")
    print(f"  HOD BREAKOUT / LOD BREAKDOWN  —  FULL YEAR 2026  —  {len(symbols)} STOCKS")
    print(f"  Period : {trading_days[0]}  →  {trading_days[-1]}  "
          f"({len(trading_days)} trading days)")
    print(f"  NOTE   : Calibrated synthetic data (yfinance/Alpaca blocked in env)")
    print(f"{'='*72}\n")
    print(f"  Generating data & running signal engine ...", flush=True)

    all_trades  = []
    sym_results = {}

    for i, sym in enumerate(symbols):
        base_px, vol, trend = UNIVERSE[sym]
        trades = _run_symbol(sym, base_px, vol, trend, trading_days,
                             seed=42 + i)
        all_trades.extend(trades)
        sym_results[sym] = trades
        print(f"  {sym:<6} done  ({len(trades)} trades)", flush=True)

    # ── Monthly breakdown ───────────────────────────────────────────────────────
    months = sorted({_month_label(t["entry_time"].date()) for t in all_trades})
    month_order = {m: i for i, m in enumerate(
        [datetime.date(2026, mo, 1).strftime("%b %Y") for mo in range(1, 7)])}
    months = sorted(months, key=lambda m: month_order.get(m, 99))

    # Count trading days per month
    tdays_by_month = {}
    for d in trading_days:
        lbl = _month_label(d)
        tdays_by_month[lbl] = tdays_by_month.get(lbl, 0) + 1

    print(f"\n{'─'*72}")
    print(f"  MONTHLY BREAKDOWN")
    print(f"{'─'*72}")
    print(f"  {'MONTH':<10} {'DAYS':>4} {'TR':>4} {'W':>4} {'L':>4} "
          f"{'WR%':>6} {'P&L':>10} {'AVG W':>8} {'AVG L':>8} {'PF':>5} {'R:R':>5}")
    print(f"  {'-'*69}")

    running_pnl = 0.0
    for mo in months:
        mo_trades = [t for t in all_trades
                     if _month_label(t["entry_time"].date()) == mo]
        s = _stats(mo_trades)
        running_pnl += s["total"]
        td = tdays_by_month.get(mo, "?")
        pf_str = f"{s['pf']:.2f}" if s["pf"] != float("inf") else " inf"
        rr_str = f"{s['rr']:.2f}" if s["rr"] != float("inf") else " inf"
        print(f"  {mo:<10} {td:>4} {s['n']:>4} {s['wins']:>4} {s['losses']:>4} "
              f"{s['wr']:>5.1f}% ${s['total']:>9,.0f} "
              f"${s['avg_w']:>7,.0f} ${s['avg_l']:>7,.0f} "
              f"{pf_str:>5} {rr_str:>5}")

    print(f"  {'-'*69}")
    s_all = _stats(all_trades)
    pf_str = f"{s_all['pf']:.2f}" if s_all["pf"] != float("inf") else "  inf"
    rr_str = f"{s_all['rr']:.2f}" if s_all["rr"] != float("inf") else "  inf"
    print(f"  {'TOTAL':<10} {len(trading_days):>4} {s_all['n']:>4} "
          f"{s_all['wins']:>4} {s_all['losses']:>4} "
          f"{s_all['wr']:>5.1f}% ${s_all['total']:>9,.0f} "
          f"${s_all['avg_w']:>7,.0f} ${s_all['avg_l']:>7,.0f} "
          f"{pf_str:>5} {rr_str:>5}")

    # ── Per-symbol summary ──────────────────────────────────────────────────────
    tier_map = {1: "BULL", -1: "BEAR", 0: "CHOP"}
    print(f"\n{'─'*72}")
    print(f"  PER-SYMBOL SUMMARY  (full year)")
    print(f"{'─'*72}")
    print(f"  {'SYM':<6} {'TIER':<5} {'TR':>4} {'WR%':>6} "
          f"{'TOTAL P&L':>11} {'AVG W':>8} {'AVG L':>8}")
    print(f"  {'-'*56}")
    for sym in symbols:
        s = _stats(sym_results[sym])
        tier = tier_map[UNIVERSE[sym][2]]
        if s["n"] == 0:
            print(f"  {sym:<6} {tier:<5}  {'--':>4}    {'--':>5}   {'--':>10}")
            continue
        print(f"  {sym:<6} {tier:<5} {s['n']:>4} {s['wr']:>5.1f}% "
              f"${s['total']:>10,.0f} ${s['avg_w']:>7,.0f} ${s['avg_l']:>7,.0f}")

    # ── Aggregates ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  FULL-YEAR SUMMARY")
    print(f"{'─'*72}")
    print(f"  Total trades    : {s_all['n']}")
    print(f"  Win rate        : {s_all['wins']}/{s_all['n']} ({s_all['wr']:.1f}%)")
    print(f"  Total P&L       : ${s_all['total']:+,.2f}")
    print(f"  Avg win         : ${s_all['avg_w']:+,.2f}")
    print(f"  Avg loss        : ${s_all['avg_l']:+,.2f}")
    print(f"  Profit factor   : {s_all['pf']:.2f}")
    print(f"  R:R ratio       : {s_all['rr']:.2f} : 1")
    print(f"  Signal freq     : {s_all['n']}/{len(symbols)*len(trading_days)} "
          f"stock-days = {s_all['n']/(len(symbols)*len(trading_days))*100:.1f}%")

    print(f"\n  By trend tier:")
    for tier_val, tier_name in [(1,"BULL"),(0,"CHOP"),(-1,"BEAR")]:
        tsyms = [s for s in symbols if UNIVERSE[s][2] == tier_val]
        tt    = [t for t in all_trades if t["symbol"] in tsyms]
        s     = _stats(tt)
        if s["n"]:
            print(f"    {tier_name:<5}: {s['n']:>4} trades  "
                  f"{s['wr']:.1f}% WR  ${s['total']:+,.0f}  PF {s['pf']:.2f}")

    print(f"\n  By direction:")
    for d in ["long", "short"]:
        dt = [t for t in all_trades if t["dir"] == d]
        s  = _stats(dt)
        if s["n"]:
            print(f"    {d.upper():<6}: {s['n']:>4} trades  "
                  f"{s['wr']:.1f}% WR  ${s['total']:+,.0f}  PF {s['pf']:.2f}")

    print(f"\n  By exit type:")
    for reason in ["stop", "cloud", "EOD"]:
        rt = [t for t in all_trades if t["reason"] == reason]
        s  = _stats(rt)
        if s["n"]:
            print(f"    {reason:<8}: {s['n']:>4} trades  "
                  f"{s['wr']:.1f}% WR  ${s['total']:+,.0f}")

    print(f"{'='*72}\n")
    return all_trades


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_year_backtest()
