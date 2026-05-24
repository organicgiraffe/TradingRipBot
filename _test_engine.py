"""Quick engine test — run after changes to ema_engine.py to verify results."""
import sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd, yfinance as yf, datetime as dt
sys.path.insert(0, ".")

from config import (EMA_PERIODS, MIN_BARS_3M, MIN_BARS_10M,
                    MAX_TRADES_PER_DAY, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
                    MAX_RISK_PER_TRADE, MIN_SHARES, MIN_STOP_DIST)
from ema_engine import (get_trend_10m, get_entry_signal_3m,
                        should_exit_3m, compute_trailing_stop)


def _add_emas(df):
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    out["hl2"] = (out["high"] + out["low"]) / 2
    for p in EMA_PERIODS:
        out[f"ema{p}"] = out["hl2"].ewm(span=p, adjust=False).mean()
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    return out


def _resample(raw, freq):
    return _add_emas(
        raw.resample(freq, label="right", closed="right").agg(
            {"Open": "first", "High": "max", "Low": "min",
             "Close": "last", "Volume": "sum"}
        ).dropna(subset=["Close"])
    )


def run_sym(symbol, target_dates, shares=None, support=None, resistance=None):
    raw = yf.download(symbol, period="60d", interval="5m",
                      progress=False, auto_adjust=True, prepost=True)
    if raw.empty:
        return []
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.tz_convert("US/Eastern")

    _pre = raw.between_time("04:00", "09:29")
    pmh_by = {}; pml_by = {}
    for d, grp in _pre.groupby(_pre.index.date):
        if not grp.empty:
            pmh_by[d] = float(grp["High"].max())
            pml_by[d] = float(grp["Low"].min())

    raw = raw.between_time("09:30", "16:00")
    df_5m  = _resample(raw, "5min")
    df_10m = _resample(raw, "10min")

    position = None; trades = []; trades_today = 0
    lost_dir_today = None; last_date = None

    for i in range(MIN_BARS_3M, len(df_5m)):
        bar_time = df_5m.index[i]
        if bar_time.date() not in target_dates:
            continue

        df_3m_now = df_5m.iloc[:i + 1]
        cur       = df_3m_now.iloc[-1]
        df_10m_now = df_10m[df_10m.index <= bar_time]
        if df_10m_now.empty:
            continue

        bar_date = bar_time.date()
        if bar_date != last_date:
            trades_today = 0; lost_dir_today = None; last_date = bar_date

        t      = bar_time.time()
        is_eod = (t.hour == MARKET_CLOSE_HOUR and t.minute >= MARKET_CLOSE_MINUTE)
        no_new = (t.hour > MARKET_CLOSE_HOUR or
                  (t.hour == MARKET_CLOSE_HOUR and t.minute >= MARKET_CLOSE_MINUTE))
        trend  = get_trend_10m(df_10m_now)

        if position and is_eod:
            ep  = cur["close"]; sh = position["shares"]
            pnl = ((ep - position["entry"]) * sh if position["dir"] == "long"
                   else (position["entry"] - ep) * sh)
            trades.append({**position, "exit": ep, "exit_time": bar_time,
                           "pnl": pnl, "reason": "EOD"})
            position = None
            continue

        if position:
            ns = compute_trailing_stop(df_3m_now, position["dir"],
                                       position["stop"], position["entry"])
            position["stop"] = ns
            sh = position["shares"]

            if position["dir"] == "long" and cur["low"] <= position["stop"]:
                pnl = (position["stop"] - position["entry"]) * sh
                trades.append({**position, "exit": position["stop"],
                               "exit_time": bar_time, "pnl": pnl, "reason": "stop"})
                if pnl < 0: lost_dir_today = position["dir"]
                trades_today += 1; position = None; continue

            if position["dir"] == "short" and cur["high"] >= position["stop"]:
                pnl = (position["entry"] - position["stop"]) * sh
                trades.append({**position, "exit": position["stop"],
                               "exit_time": bar_time, "pnl": pnl, "reason": "stop"})
                if pnl < 0: lost_dir_today = position["dir"]
                trades_today += 1; position = None; continue

            if should_exit_3m(df_3m_now, position["dir"]):
                ep  = cur["close"]
                pnl = ((ep - position["entry"]) * sh if position["dir"] == "long"
                       else (position["entry"] - ep) * sh)
                trades.append({**position, "exit": ep, "exit_time": bar_time,
                               "pnl": pnl, "reason": "cloud exit"})
                if pnl < 0: lost_dir_today = position["dir"]
                trades_today += 1; position = None
            continue

        if no_new or trades_today >= MAX_TRADES_PER_DAY:
            continue

        pmh    = pmh_by.get(bar_date)
        pml    = pml_by.get(bar_date)
        signal, stop_price = get_entry_signal_3m(
            df_3m_now, trend, bar_time=bar_time,
            pmh=pmh, pml=pml, support=support, resistance=resistance)
        if signal == "none" or signal == lost_dir_today:
            continue

        entry_price = cur["close"]
        stop_dist   = abs(entry_price - stop_price)
        if stop_dist < MIN_STOP_DIST:
            continue

        n = (max(MIN_SHARES, int(MAX_RISK_PER_TRADE / stop_dist))
             if shares is None else shares)
        position = {
            "symbol": symbol, "dir": signal,
            "entry": entry_price, "stop": stop_price,
            "shares": n, "entry_time": bar_time, "risk": stop_dist * n,
        }
    return trades


def _banner(trades, label):
    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total  = sum(t["pnl"] for t in trades)
    rate   = f"{len(wins)/len(trades)*100:.0f}%" if trades else "n/a"
    print(f"  {label}: {len(trades)} trades  {len(wins)}W/{len(losses)}L  "
          f"win={rate}  ${total:+.0f}")
    if wins:   print(f"    avg win : ${sum(t['pnl'] for t in wins)/len(wins):+.0f}")
    if losses: print(f"    avg loss: ${sum(t['pnl'] for t in losses)/len(losses):+.0f}")


if __name__ == "__main__":
    # ── Generic 5-day ─────────────────────────────────────────────────────
    print("\n=== GENERIC 5-DAY ===")
    spy = yf.download("SPY", period="10d", interval="1d", progress=False)
    last5 = set(sorted(spy.index.date)[-5:])
    all_g = []
    for sym in ["TSLA", "NVDA", "AAPL", "META", "AMD", "MSFT"]:
        tr = run_sym(sym, last5)
        all_g.extend(tr)
        wins = sum(1 for t in tr if t["pnl"] > 0)
        total = sum(t["pnl"] for t in tr)
        print(f"  {sym:5s}: {len(tr)} trades  {wins}W/{len(tr)-wins}L  ${total:+.0f}")
        for t in tr:
            sign = "+" if t["pnl"] >= 0 else ""
            print(f"    {t['dir'].upper():<5} {t['entry_time'].strftime('%m/%d %H:%M')}  "
                  f"${t['entry']:.2f}->${t['exit']:.2f}  x{t['shares']}sh  "
                  f"${sign}{t['pnl']:.0f}  [{t['reason']}]")
    _banner(all_g, "GENERIC TOTAL")

    # ── Broader Rip-style 15-day test ─────────────────────────────────────
    print("\n=== BROADER 15-COMBO TEST ===")
    combos = [
        ("NVDA", "2026-05-19"), ("NVDA", "2026-05-20"),
        ("NVDA", "2026-05-21"), ("NVDA", "2026-05-22"),
        ("TSLA", "2026-05-19"), ("TSLA", "2026-05-20"), ("TSLA", "2026-05-21"),
        ("AAPL", "2026-05-20"), ("AAPL", "2026-05-21"), ("AAPL", "2026-05-22"),
        ("META", "2026-05-20"), ("META", "2026-05-21"), ("META", "2026-05-22"),
        ("AMD",  "2026-05-20"), ("AMD",  "2026-05-21"),
    ]
    all_b = []
    for sym, ds in combos:
        tr = run_sym(sym, {dt.date.fromisoformat(ds)})
        all_b.extend(tr)
        for t in tr:
            sign = "+" if t["pnl"] >= 0 else ""
            w    = "W" if t["pnl"] > 0 else "L"
            print(f"  {sym:5s} {ds}  {t['dir'].upper():<5} "
                  f"{t['entry_time'].strftime('%H:%M')}  "
                  f"${t['entry']:.2f}->${t['exit']:.2f}  x{t['shares']}sh  "
                  f"{w} ${sign}{t['pnl']:.0f}  [{t['reason']}]")
    _banner(all_b, "BROADER TOTAL")
