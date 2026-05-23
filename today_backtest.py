"""
today_backtest.py — run today's intraday simulation on the stocks
from Rip's Daily Levels sheet.

Usage:  python today_backtest.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf
sys.path.insert(0, ".")

from config import (EMA_PERIODS, MIN_BARS_3M, MIN_BARS_10M,
                    MAX_TRADES_PER_DAY, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
                    MAX_RISK_PER_TRADE, MIN_SHARES, MIN_STOP_DIST)
from ema_engine import (get_trend_10m, get_entry_signal_3m,
                        should_exit_3m, compute_trailing_stop)


def _add_emas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    out["hl2"] = (out["high"] + out["low"]) / 2
    for p in EMA_PERIODS:
        out[f"ema{p}"] = out["hl2"].ewm(span=p, adjust=False).mean()
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    return out


def _resample(raw, freq):
    return _add_emas(
        raw.resample(freq, label="right", closed="right").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna(subset=["Close"])
    )


def run_today(symbol: str, shares: int = None,
              rip_levels: dict = None, date_str: str = None):
    """
    Download 60 days of 5-min data for EMA warm-up, simulate entries on today only.

    shares:     if None (default), sized dynamically from MAX_RISK_PER_TRADE / stop_distance
    rip_levels: optional dict with keys 'support', 'resistance', 'bias'
                e.g. {"support": 219.20, "resistance": 220.46, "bias": "long"}
    """
    print(f"\n{'='*55}")
    label = f"  {symbol}  |  {'auto-sized' if shares is None else str(shares) + ' shares'}"
    print(label, end="")
    if rip_levels:
        print(f"  |  Rip sup={rip_levels.get('support')}  res={rip_levels.get('resistance')}  bias={rip_levels.get('bias','any')}", end="")
    print(f"\n{'='*55}")

    # Pull 60 days of 5-min data (including pre-market) for EMA warm-up and
    # pre-market high/low levels.  Regular-session bars only are used for the sim.
    raw = yf.download(symbol, period="60d", interval="5m",
                      progress=False, auto_adjust=True, prepost=True)
    if raw.empty:
        print("  No data returned.")
        return []
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.tz_convert("US/Eastern")

    # Determine trading date
    all_dates = sorted(set(
        raw.between_time("09:30", "16:00").index.date
    ))
    if date_str:
        import datetime
        target = datetime.date.fromisoformat(date_str)
    else:
        target = all_dates[-1]   # most recent day

    # Pre-market high / low per date (04:00–09:29 ET).
    # Used by TYPE 4 signal: first regular-session close above PMH / below PML.
    _pre = raw.between_time("04:00", "09:29")
    pmh_by_date = {}
    pml_by_date = {}
    for _dt, _grp in _pre.groupby(_pre.index.date):
        if not _grp.empty:
            pmh_by_date[_dt] = float(_grp["High"].max())
            pml_by_date[_dt] = float(_grp["Low"].min())

    pmh_today = pmh_by_date.get(target)
    pml_today = pml_by_date.get(target)
    if pmh_today and pml_today:
        print(f"  Pre-market:  H=${pmh_today:.2f}  L=${pml_today:.2f}"
              f"  range=${pmh_today - pml_today:.2f}")
    else:
        print("  Pre-market:  no data")

    print(f"  Simulating: {target}")

    # Regular-session bars only for the simulation loop
    raw = raw.between_time("09:30", "16:00")

    # Build 5-min (entry proxy) and 10-min (trend) from full 60-day history
    df_5m  = _resample(raw, "5min")
    df_10m = _resample(raw, "10min")

    print(f"  5-min bars: {len(df_5m)}  |  10-min bars: {len(df_10m)}")

    position      = None
    trades        = []
    trades_today  = 0
    lost_dir_today = None

    for i in range(MIN_BARS_3M, len(df_5m)):
        bar_time = df_5m.index[i]
        if bar_time.date() != target:
            continue

        df_3m_now  = df_5m.iloc[:i + 1]
        cur        = df_3m_now.iloc[-1]
        df_10m_now = df_10m[df_10m.index <= bar_time]
        if df_10m_now.empty:
            continue

        t      = bar_time.time()
        is_eod = (t.hour == MARKET_CLOSE_HOUR and t.minute >= MARKET_CLOSE_MINUTE)
        no_new = (t.hour > MARKET_CLOSE_HOUR or
                  (t.hour == MARKET_CLOSE_HOUR and t.minute >= MARKET_CLOSE_MINUTE))

        trend = get_trend_10m(df_10m_now)

        # ---- EOD forced close ----
        if position and is_eod:
            ep  = cur["close"]
            sh  = position["shares"]
            pnl = ((ep - position["entry"]) * sh
                   if position["dir"] == "long"
                   else (position["entry"] - ep) * sh)
            trades.append({**position, "exit": ep, "exit_time": bar_time,
                           "pnl": pnl, "reason": "EOD close"})
            _print_trade(trades[-1])
            position = None
            continue

        # ---- Manage open position ----
        if position:
            new_stop = compute_trailing_stop(
                df_3m_now, position["dir"], position["stop"], position["entry"])
            position["stop"] = new_stop
            sh = position["shares"]

            if position["dir"] == "long" and cur["low"] <= position["stop"]:
                pnl = (position["stop"] - position["entry"]) * sh
                trades.append({**position, "exit": position["stop"],
                               "exit_time": bar_time, "pnl": pnl, "reason": "trailing stop"})
                _print_trade(trades[-1])
                if pnl < 0:           # only block re-entry on actual loss, not breakeven
                    lost_dir_today = position["dir"]
                trades_today += 1
                position = None
                continue

            if position["dir"] == "short" and cur["high"] >= position["stop"]:
                pnl = (position["entry"] - position["stop"]) * sh
                trades.append({**position, "exit": position["stop"],
                               "exit_time": bar_time, "pnl": pnl, "reason": "trailing stop"})
                _print_trade(trades[-1])
                if pnl < 0:
                    lost_dir_today = position["dir"]
                trades_today += 1
                position = None
                continue

            if should_exit_3m(df_3m_now, position["dir"]):
                ep  = cur["close"]
                pnl = ((ep - position["entry"]) * sh
                       if position["dir"] == "long"
                       else (position["entry"] - ep) * sh)
                trades.append({**position, "exit": ep, "exit_time": bar_time,
                               "pnl": pnl, "reason": "cloud exit"})
                _print_trade(trades[-1])
                if pnl < 0:
                    lost_dir_today = position["dir"]
                trades_today += 1
                position = None
            continue

        # ---- Entry ----
        if no_new or trades_today >= MAX_TRADES_PER_DAY:
            continue

        signal, stop_price = get_entry_signal_3m(
            df_3m_now, trend, bar_time=bar_time,
            pmh=pmh_today, pml=pml_today)
        if signal == "none":
            continue
        if signal == lost_dir_today:
            continue

        # No bias filter — take what the market gives us.
        # The EMA signal engine decides direction based on cloud alignment alone.

        entry_price  = cur["close"]
        stop_dist    = abs(entry_price - stop_price)
        if stop_dist < MIN_STOP_DIST:
            continue   # degenerate entry — stop is too tight to size sensibly
        if shares is None:
            # Risk-based sizing: risk exactly MAX_RISK_PER_TRADE dollars
            n_shares = max(MIN_SHARES, int(MAX_RISK_PER_TRADE / stop_dist))
        else:
            n_shares = shares
        risk = stop_dist * n_shares
        print(f"  >> ENTRY  {signal.upper():<5} {bar_time.strftime('%H:%M')}  "
              f"@ ${entry_price:.2f}  stop=${stop_price:.2f}  "
              f"shares={n_shares}  risk=${risk:.2f}  trend={trend}")
        position = {"symbol": symbol, "dir": signal,
                    "entry": entry_price, "stop": stop_price,
                    "shares": n_shares,
                    "entry_time": bar_time, "risk": risk}

    # ---- Summary ----
    print(f"  Completed trades: {len(trades)}")
    if trades:
        total = sum(t["pnl"] for t in trades)
        wins  = [t for t in trades if t["pnl"] > 0]
        losses= [t for t in trades if t["pnl"] <= 0]
        print(f"  Win rate : {len(wins)}/{len(trades)}  |  Total P&L: ${total:+.2f}")
        if wins:   print(f"  Avg win  : ${sum(t['pnl'] for t in wins)/len(wins):+.2f}")
        if losses: print(f"  Avg loss : ${sum(t['pnl'] for t in losses)/len(losses):+.2f}")
    return trades


def _print_trade(t):
    sign = "+" if t["pnl"] >= 0 else ""
    sh   = t.get("shares", "?")
    print(f"     CLOSE {t['dir'].upper():<5} "
          f"{t['entry_time'].strftime('%H:%M')} -> {t['exit_time'].strftime('%H:%M')}  "
          f"${t['entry']:.2f} -> ${t['exit']:.2f}  "
          f"x{sh}sh  pnl=${sign}{t['pnl']:.2f}  [{t['reason']}]")


# ------------------------------------------------------------------ #
# Today's picks — read directly from Rip's Daily Levels sheet
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("\nRIP'S DAILY PLAYS  —  May 22, 2026  (Lotto Friday)")
    print("=" * 65)
    print("  SOURCE: Rip's Day2/Day3 Play + News Play + Wakeup Summary")
    print()
    print("  DAY2/DAY3 CONTINUATION PLAYS:")
    print("  TSLA  — Daily looking good, long over PMH, watch flow       bias=LONG")
    print("  AAOI  — Bullish bias long over YH or 34/50 EMA curl         bias=LONG")
    print("  INTU  — Earnings Day2, bearish under 305/EMA break          bias=SHORT")
    print("  DELL  — Evercore ISI Tactical Outperform add, PT $270       bias=LONG")
    print()
    print("  TIER 1 BANK CATALYST PLAYS:")
    print("  SPOT  — JPMorgan raises PT  (Tier 1)                        bias=LONG")
    print("  DKLD  — BofA initiates Buy, PT $93  (Tier 1)               bias=LONG")
    print("  INSP  — BofA downgrade  (Tier 1)                            bias=SHORT")
    print()
    print("  SKIP:  AMD (wide $16 range, no clean bias)")
    print("  SKIP:  NVDA (Inside Day, 'no go under 220')")
    print("=" * 65)

    # Rip's sheet: symbol -> {support, resistance, bias, catalyst}
    setups = {
        # Day2/Day3 continuation plays
        "TSLA": {"support": 449.45, "resistance": 452.00, "bias": "long",
                 "note": "Daily looking good, long over PMH"},
        "AAOI": {"support": 59.50,  "resistance": 63.90,  "bias": "long",
                 "note": "Bullish long over YH or 34/50 EMA curl"},
        "INTU": {"support": 302.40, "resistance": 309.00, "bias": "short",
                 "note": "Earnings Day2 — bearish under 305"},
        "DELL": {"support": None,   "resistance": None,   "bias": "long",
                 "note": "Evercore ISI Tactical Outperform, PT $270"},
        # Tier 1 bank catalyst plays
        "SPOT": {"support": None,   "resistance": None,   "bias": "long",
                 "note": "JPMorgan raises PT (Tier 1 bank)"},
        "DKLD": {"support": None,   "resistance": None,   "bias": "long",
                 "note": "BofA initiates Buy PT $93 (Tier 1 bank)"},
        "INSP": {"support": None,   "resistance": None,   "bias": "short",
                 "note": "BofA downgrade (Tier 1 bank)"},
    }

    all_trades = []
    for sym, levels in setups.items():
        t = run_today(sym, shares=100, rip_levels=levels)
        all_trades.extend(t)

    wins   = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    total  = sum(t["pnl"] for t in all_trades)
    print(f"\n{'='*65}")
    print(f"  TOTAL TODAY  |  {len(all_trades)} trades  "
          f"|  {len(wins)}W / {len(losses)}L  |  ${total:+.2f}")
    if wins:   print(f"  Avg win  : ${sum(t['pnl'] for t in wins)/len(wins):+.2f}")
    if losses: print(f"  Avg loss : ${sum(t['pnl'] for t in losses)/len(losses):+.2f}")
    print(f"{'='*65}")
