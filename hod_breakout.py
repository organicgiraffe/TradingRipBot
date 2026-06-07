"""
hod_breakout.py — HOD Breakout (long) / LOD Breakdown (short) strategy.

Scanner criteria (Humbled Trader, via Claude + TradingView video):

  LONG  — all conditions must be true after 10:00 AM ET:
    1. price > yesterday's daily high
    2. yesterday's close > 200D SMA
    3. price > today's premarket high
    4. price is making a new intraday HOD (close > all prior RTH highs today)

  SHORT — full inverse:
    1. price < yesterday's daily low
    2. yesterday's close < 200D SMA
    3. price < today's premarket low
    4. price is making a new intraday LOD

  Entry: close of the first 5-min bar that satisfies all conditions.
  Stop : the breakout level (prev_day_high for longs, prev_day_low for shorts).
         If ema50 is tighter (closer to entry), use that instead.
  Exit : trailing ema50 stop + fast-cloud (ema5/ema12) flip exit — same as
         the rest of the bot.

Usage:
    python hod_breakout.py
"""
import sys, warnings, datetime
warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf
sys.path.insert(0, ".")

from config import (EMA_PERIODS, MIN_STOP_DIST, MAX_STOP_PCT,
                    MAX_TRADES_PER_DAY, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
                    MAX_SIMULTANEOUS_POSITIONS, MAX_RISK_PER_TRADE, MIN_SHARES)
from ema_engine import should_exit_3m, compute_trailing_stop


# ── HOD entry time gate ────────────────────────────────────────────────────────
HOD_ENTRY_HOUR   = 10   # no entries before 10:00 AM ET
HOD_ENTRY_MINUTE = 0


# ------------------------------------------------------------------ #
# Data helpers
# ------------------------------------------------------------------ #

def _add_emas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    out["hl2"] = (out["high"] + out["low"]) / 2
    for p in EMA_PERIODS:
        out[f"ema{p}"] = out["hl2"].ewm(span=p, adjust=False).mean()
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    return out


def _resample(raw: pd.DataFrame, freq: str) -> pd.DataFrame:
    return _add_emas(
        raw.resample(freq, label="right", closed="right").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna(subset=["Close"])
    )


def _load_symbol(symbol: str):
    """
    Download ~65 days of 5-min data (enough for EMA warm-up + daily SMA200).
    Returns (df_5m, daily_df, pmh_by_date, pml_by_date) or None on failure.

    daily_df index = date, columns include 'high', 'low', 'close', 'sma200'.
    """
    raw = yf.download(symbol, period="65d", interval="5m",
                      progress=False, auto_adjust=True, prepost=True)
    if raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.tz_convert("US/Eastern")

    # Pre-market high / low per date (04:00–09:29)
    pre = raw.between_time("04:00", "09:29")
    pmh_by, pml_by = {}, {}
    for d, grp in pre.groupby(pre.index.date):
        if not grp.empty:
            pmh_by[d] = float(grp["High"].max())
            pml_by[d] = float(grp["Low"].min())

    rth = raw.between_time("09:30", "16:00")
    df_5m = _resample(rth, "5min")

    # Daily OHLC + 200D SMA from the 5-min data collapsed to daily
    daily = (rth.groupby(rth.index.date)
               .agg(high=("High", "max"), low=("Low", "min"),
                    close=("Close", "last"), open=("Open", "first")))
    daily["sma200"] = daily["close"].rolling(200, min_periods=1).mean()

    return df_5m, daily, pmh_by, pml_by


# ------------------------------------------------------------------ #
# Signal
# ------------------------------------------------------------------ #

def get_hod_signal(bar_time, cur, df_5m_today: pd.DataFrame,
                   prev_day_high: float, prev_day_low: float,
                   prev_day_close: float, sma200: float,
                   pmh: float, pml: float) -> tuple[str, float]:
    """
    Check HOD breakout / LOD breakdown conditions at bar_time.

    df_5m_today : all RTH bars for today UP TO AND INCLUDING bar_time.

    Returns (signal, stop_price)  where signal in {'long','short','none'}.
    """
    # ── Time gate: no entries before 10:00 AM ─────────────────────────
    t = bar_time.time()
    if t.hour < HOD_ENTRY_HOUR or (t.hour == HOD_ENTRY_HOUR and t.minute < HOD_ENTRY_MINUTE):
        return "none", 0.0

    close = float(cur["close"])

    # ── Running HOD / LOD: max/min of all bars BEFORE the current bar ─
    prior = df_5m_today.iloc[:-1]   # exclude the bar we're evaluating
    if prior.empty:
        return "none", 0.0

    running_hod = float(prior["high"].max())
    running_lod = float(prior["low"].min())

    # ── LONG: price > prev_day_high, prev_close > SMA200,
    #         price > PMH, price makes new HOD ──────────────────────────
    if (prev_day_high is not None and pmh is not None and
            close > prev_day_high and
            prev_day_close > sma200 and
            close > pmh and
            close > running_hod):
        # Stop = prev_day_high (breakout level); tighten to ema50 if closer
        stop = prev_day_high
        if cur["ema50"] > stop and cur["ema50"] < close:
            stop = float(cur["ema50"])
        dist = close - stop
        if dist < MIN_STOP_DIST or dist / close > MAX_STOP_PCT:
            return "none", 0.0
        return "long", stop

    # ── SHORT: price < prev_day_low, prev_close < SMA200,
    #          price < PML, price makes new LOD ──────────────────────────
    if (prev_day_low is not None and pml is not None and
            close < prev_day_low and
            prev_day_close < sma200 and
            close < pml and
            close < running_lod):
        # Stop = prev_day_low (breakdown level); tighten to ema50 if closer
        stop = prev_day_low
        if cur["ema50"] < stop and cur["ema50"] > close:
            stop = float(cur["ema50"])
        dist = stop - close
        if dist < MIN_STOP_DIST or dist / close > MAX_STOP_PCT:
            return "none", 0.0
        return "short", stop

    return "none", 0.0


# ------------------------------------------------------------------ #
# Single-symbol backtest
# ------------------------------------------------------------------ #

def run_hod_today(symbol: str, date_str: str = None) -> list:
    print(f"\n{'='*55}")
    print(f"  {symbol}  |  HOD Breakout / LOD Breakdown")
    print(f"{'='*55}")

    result = _load_symbol(symbol)
    if result is None:
        print("  No data.")
        return []
    df_5m, daily, pmh_by, pml_by = result

    all_dates = sorted(set(df_5m.index.date))
    target = (datetime.date.fromisoformat(date_str) if date_str
              else all_dates[-1])

    # ── Previous trading day ────────────────────────────────────────────
    prev_days = [d for d in daily.index if d < target]
    if not prev_days:
        print("  No prior-day data.")
        return []
    prev_day = prev_days[-1]
    prev_row       = daily.loc[prev_day]
    prev_day_high  = float(prev_row["high"])
    prev_day_low   = float(prev_row["low"])
    prev_day_close = float(prev_row["close"])
    sma200         = float(prev_row["sma200"])
    pmh = pmh_by.get(target)
    pml = pml_by.get(target)

    print(f"  Date      : {target}")
    print(f"  Prev high : ${prev_day_high:.2f}   Prev low: ${prev_day_low:.2f}   "
          f"Prev close: ${prev_day_close:.2f}")
    print(f"  SMA200    : ${sma200:.2f}   "
          f"{'ABOVE' if prev_day_close > sma200 else 'BELOW'} (daily trend "
          f"{'bullish' if prev_day_close > sma200 else 'bearish'})")
    print(f"  Pre-market: H=${pmh:.2f}  L=${pml:.2f}" if pmh else
          "  Pre-market: no data")
    print()

    today_5m = df_5m[df_5m.index.date == target]
    if today_5m.empty:
        print("  No intraday bars.")
        return []

    position = None
    trades   = []
    trades_today = 0

    for i, bar_time in enumerate(today_5m.index):
        t      = bar_time.time()
        is_eod = (t.hour == MARKET_CLOSE_HOUR and t.minute >= MARKET_CLOSE_MINUTE)
        no_new = (t.hour > MARKET_CLOSE_HOUR or
                  (t.hour == MARKET_CLOSE_HOUR and t.minute >= MARKET_CLOSE_MINUTE))

        df_now = today_5m.iloc[:i + 1]
        cur    = df_now.iloc[-1]

        # ── Manage open position ────────────────────────────────────────
        if position:
            if is_eod:
                ep  = float(cur["close"])
                sh  = position["shares"]
                pnl = ((ep - position["entry"]) * sh if position["dir"] == "long"
                       else (position["entry"] - ep) * sh)
                trades.append({**position, "exit": ep, "exit_time": bar_time,
                               "pnl": pnl, "reason": "EOD close"})
                _print_trade(trades[-1])
                position = None
                continue

            new_stop = compute_trailing_stop(
                df_now, position["dir"], position["stop"], position["entry"])
            position["stop"] = new_stop
            sh = position["shares"]

            if position["dir"] == "long" and float(cur["low"]) <= position["stop"]:
                pnl = (position["stop"] - position["entry"]) * sh
                trades.append({**position, "exit": position["stop"],
                               "exit_time": bar_time, "pnl": pnl,
                               "reason": "trailing stop"})
                _print_trade(trades[-1])
                trades_today += 1; position = None; continue

            if position["dir"] == "short" and float(cur["high"]) >= position["stop"]:
                pnl = (position["entry"] - position["stop"]) * sh
                trades.append({**position, "exit": position["stop"],
                               "exit_time": bar_time, "pnl": pnl,
                               "reason": "trailing stop"})
                _print_trade(trades[-1])
                trades_today += 1; position = None; continue

            if should_exit_3m(df_now, position["dir"]):
                ep  = float(cur["close"])
                pnl = ((ep - position["entry"]) * sh if position["dir"] == "long"
                       else (position["entry"] - ep) * sh)
                trades.append({**position, "exit": ep, "exit_time": bar_time,
                               "pnl": pnl, "reason": "cloud exit"})
                _print_trade(trades[-1])
                trades_today += 1; position = None
            continue

        # ── Entry ────────────────────────────────────────────────────────
        if no_new or trades_today >= MAX_TRADES_PER_DAY:
            continue

        signal, stop_price = get_hod_signal(
            bar_time, cur, df_now,
            prev_day_high, prev_day_low, prev_day_close, sma200,
            pmh, pml)

        if signal == "none":
            continue

        entry_price = float(cur["close"])
        stop_dist   = abs(entry_price - stop_price)
        n_shares    = max(MIN_SHARES, int(MAX_RISK_PER_TRADE / stop_dist))
        risk        = stop_dist * n_shares

        print(f"  >> ENTRY  {signal.upper():<5} {bar_time.strftime('%H:%M')}  "
              f"@ ${entry_price:.2f}  stop=${stop_price:.2f}  "
              f"shares={n_shares}  risk=${risk:.2f}")
        position = {"symbol": symbol, "dir": signal,
                    "entry": entry_price, "stop": stop_price,
                    "shares": n_shares, "entry_time": bar_time, "risk": risk}

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n  Trades: {len(trades)}")
    if trades:
        total  = sum(t["pnl"] for t in trades)
        wins   = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        print(f"  Win rate : {len(wins)}/{len(trades)}  |  Total P&L: ${total:+.2f}")
        if wins:   print(f"  Avg win  : ${sum(t['pnl'] for t in wins)/len(wins):+.2f}")
        if losses: print(f"  Avg loss : ${sum(t['pnl'] for t in losses)/len(losses):+.2f}")
    return trades


# ------------------------------------------------------------------ #
# Multi-symbol runner
# ------------------------------------------------------------------ #

def run_hod_multi(symbols: list, date_str: str = None) -> list:
    """Run HOD breakout on multiple symbols with MAX_SIMULTANEOUS_POSITIONS limit."""
    print(f"\n{'='*65}")
    print(f"  HOD BREAKOUT / LOD BREAKDOWN  |  {len(symbols)} symbols  "
          f"|  max {MAX_SIMULTANEOUS_POSITIONS} simultaneous position(s)")
    print(f"{'='*65}")

    # ── Load data ────────────────────────────────────────────────────────
    sym_data = {}
    for sym in symbols:
        result = _load_symbol(sym)
        if result is None:
            print(f"  {sym}: no data — skipped")
            continue
        df_5m, daily, pmh_by, pml_by = result
        sym_data[sym] = {"df_5m": df_5m, "daily": daily,
                         "pmh_by": pmh_by, "pml_by": pml_by}
        print(f"  {sym:6s} loaded  {len(df_5m)} bars")

    if not sym_data:
        print("  No data for any symbol.")
        return []

    # ── Target date ────────────────────────────────────────────────────
    if date_str:
        target = datetime.date.fromisoformat(date_str)
    else:
        target = sorted(d for sd in sym_data.values()
                        for d in sd["df_5m"].index.date)[-1]

    print(f"\n  Simulating: {target}\n")

    # ── Per-symbol prior-day levels ────────────────────────────────────
    sym_levels = {}
    for sym, sd in sym_data.items():
        daily = sd["daily"]
        prev_days = [d for d in daily.index if d < target]
        if not prev_days:
            continue
        prev_row = daily.loc[prev_days[-1]]
        sym_levels[sym] = {
            "prev_high":  float(prev_row["high"]),
            "prev_low":   float(prev_row["low"]),
            "prev_close": float(prev_row["close"]),
            "sma200":     float(prev_row["sma200"]),
            "pmh":        sd["pmh_by"].get(target),
            "pml":        sd["pml_by"].get(target),
        }
        lvl = sym_levels[sym]
        bias = "BULLISH" if lvl["prev_close"] > lvl["sma200"] else "BEARISH"
        pm = (f"PM H=${lvl['pmh']:.2f}  L=${lvl['pml']:.2f}"
              if lvl["pmh"] else "no PM data")
        print(f"  {sym:6s}  PrevH=${lvl['prev_high']:.2f}  "
              f"PrevL=${lvl['prev_low']:.2f}  SMA200=${lvl['sma200']:.2f}  "
              f"{bias}  |  {pm}")

    # ── Merged timeline ────────────────────────────────────────────────
    all_times = sorted({
        t for sd in sym_data.values()
        for t in sd["df_5m"].index
        if t.date() == target
    })

    # ── Simulation state ──────────────────────────────────────────────
    positions      = {}
    all_trades     = []
    trades_today   = {sym: 0 for sym in sym_data}
    print()

    for bar_time in all_times:
        t      = bar_time.time()
        is_eod = (t.hour == MARKET_CLOSE_HOUR and t.minute >= MARKET_CLOSE_MINUTE)
        no_new = (t.hour > MARKET_CLOSE_HOUR or
                  (t.hour == MARKET_CLOSE_HOUR and t.minute >= MARKET_CLOSE_MINUTE))

        # ── Manage open positions ────────────────────────────────────────
        for sym in list(positions.keys()):
            pos    = positions[sym]
            df5    = sym_data[sym]["df_5m"]
            df_now = df5[df5.index <= bar_time]
            if df_now.empty:
                continue
            cur = df_now.iloc[-1]
            sh  = pos["shares"]

            if is_eod:
                ep  = float(cur["close"])
                pnl = ((ep - pos["entry"]) * sh if pos["dir"] == "long"
                       else (pos["entry"] - ep) * sh)
                all_trades.append({**pos, "exit": ep, "exit_time": bar_time,
                                   "pnl": pnl, "reason": "EOD close"})
                _print_trade(all_trades[-1])
                del positions[sym]; continue

            new_stop = compute_trailing_stop(
                df_now, pos["dir"], pos["stop"], pos["entry"])
            pos["stop"] = new_stop

            if pos["dir"] == "long" and float(cur["low"]) <= pos["stop"]:
                pnl = (pos["stop"] - pos["entry"]) * sh
                all_trades.append({**pos, "exit": pos["stop"],
                                   "exit_time": bar_time, "pnl": pnl,
                                   "reason": "trailing stop"})
                _print_trade(all_trades[-1])
                trades_today[sym] += 1; del positions[sym]; continue

            if pos["dir"] == "short" and float(cur["high"]) >= pos["stop"]:
                pnl = (pos["entry"] - pos["stop"]) * sh
                all_trades.append({**pos, "exit": pos["stop"],
                                   "exit_time": bar_time, "pnl": pnl,
                                   "reason": "trailing stop"})
                _print_trade(all_trades[-1])
                trades_today[sym] += 1; del positions[sym]; continue

            if should_exit_3m(df_now, pos["dir"]):
                ep  = float(cur["close"])
                pnl = ((ep - pos["entry"]) * sh if pos["dir"] == "long"
                       else (pos["entry"] - ep) * sh)
                all_trades.append({**pos, "exit": ep, "exit_time": bar_time,
                                   "pnl": pnl, "reason": "cloud exit"})
                _print_trade(all_trades[-1])
                trades_today[sym] += 1; del positions[sym]

        # ── Entry ────────────────────────────────────────────────────────
        if no_new or len(positions) >= MAX_SIMULTANEOUS_POSITIONS:
            continue

        for sym in symbols:
            if len(positions) >= MAX_SIMULTANEOUS_POSITIONS:
                break
            if sym not in sym_data or sym not in sym_levels:
                continue
            if sym in positions or trades_today[sym] >= MAX_TRADES_PER_DAY:
                continue

            df5    = sym_data[sym]["df_5m"]
            df_now = df5[(df5.index.date == target) & (df5.index <= bar_time)]
            if df_now.empty:
                continue
            cur  = df_now.iloc[-1]
            lvl  = sym_levels[sym]

            signal, stop_price = get_hod_signal(
                bar_time, cur, df_now,
                lvl["prev_high"], lvl["prev_low"], lvl["prev_close"], lvl["sma200"],
                lvl["pmh"], lvl["pml"])

            if signal == "none":
                continue

            entry_price = float(cur["close"])
            stop_dist   = abs(entry_price - stop_price)
            n_shares    = max(MIN_SHARES, int(MAX_RISK_PER_TRADE / stop_dist))
            risk        = stop_dist * n_shares
            slot        = len(positions) + 1

            print(f"  >> ENTRY  {signal.upper():<5} {sym:6s} "
                  f"{bar_time.strftime('%H:%M')}  "
                  f"@ ${entry_price:.2f}  stop=${stop_price:.2f}  "
                  f"shares={n_shares}  risk=${risk:.2f}  "
                  f"[{slot}/{MAX_SIMULTANEOUS_POSITIONS}]")

            positions[sym] = {
                "symbol": sym, "dir": signal,
                "entry": entry_price, "stop": stop_price,
                "shares": n_shares, "entry_time": bar_time, "risk": risk,
            }

    # ── Summary ────────────────────────────────────────────────────────
    wins   = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    total  = sum(t["pnl"] for t in all_trades)
    print(f"\n  {'-'*55}")
    print(f"  Completed: {len(all_trades)} trades  "
          f"|  {len(wins)}W / {len(losses)}L  |  ${total:+.2f}")
    if wins:   print(f"  Avg win  : ${sum(t['pnl'] for t in wins)/len(wins):+.2f}")
    if losses: print(f"  Avg loss : ${sum(t['pnl'] for t in losses)/len(losses):+.2f}")

    syms_traded = sorted({t["symbol"] for t in all_trades})
    if len(syms_traded) > 1:
        print()
        for sym in syms_traded:
            sym_t = [t for t in all_trades if t["symbol"] == sym]
            sym_w = sum(1 for t in sym_t if t["pnl"] > 0)
            print(f"  {sym:6s}: {len(sym_t)} trade(s)  "
                  f"{sym_w}W/{len(sym_t)-sym_w}L  "
                  f"${sum(t['pnl'] for t in sym_t):+.2f}")

    return all_trades


def _print_trade(t):
    sign = "+" if t["pnl"] >= 0 else ""
    print(f"     CLOSE {t['symbol']:6s} {t['dir'].upper():<5} "
          f"{t['entry_time'].strftime('%H:%M')} -> {t['exit_time'].strftime('%H:%M')}  "
          f"${t['entry']:.2f} -> ${t['exit']:.2f}  "
          f"x{t['shares']}sh  pnl=${sign}{t['pnl']:.2f}  [{t['reason']}]")


# ------------------------------------------------------------------ #
# Run today's watchlist
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("\nHOD BREAKOUT / LOD BREAKDOWN  —  June 4, 2026")
    print("=" * 65)
    print("  Strategy: Humbled Trader scanner criteria")
    print("  LONG  after 10am: price > prev_high, prev_close > SMA200,")
    print("        price > PMH, price making new HOD")
    print("  SHORT (inverse): price < prev_low, prev_close < SMA200,")
    print("        price < PML, price making new LOD")
    print("=" * 65)

    # Today's universe — same as Rip's sheet (SPY/QQQ + top movers)
    symbols = ["SPY", "QQQ", "NVDA", "TSLA", "AVGO", "COST", "PANW"]

    all_trades = run_hod_multi(symbols)

    wins   = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    total  = sum(t["pnl"] for t in all_trades)
    print(f"\n{'='*65}")
    print(f"  TOTAL  |  {len(all_trades)} trades  "
          f"|  {len(wins)}W / {len(losses)}L  |  ${total:+.2f}")
    if wins:   print(f"  Avg win  : ${sum(t['pnl'] for t in wins)/len(wins):+.2f}")
    if losses: print(f"  Avg loss : ${sum(t['pnl'] for t in losses)/len(losses):+.2f}")
    print(f"{'='*65}")
