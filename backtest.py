"""
Backtest using Yahoo Finance historical data — no IBKR required.
Downloads the last 5 trading days, resamples to 3-min and 10-min,
then walks forward bar-by-bar simulating the live bot's logic.

Run:  python backtest.py
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
sys.path.insert(0, ".")

from config import (EMA_PERIODS, MIN_BARS_10M, MIN_BARS_3M,
                    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
                    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
                    MAX_TRADES_PER_DAY, BREAKEVEN_TRIGGER)
from ema_engine import (get_trend_10m, get_entry_signal_3m,
                        should_exit_3m, compute_trailing_stop)


# ------------------------------------------------------------------ #
# Helper: compute EMAs on a plain DataFrame (yfinance format)
# ------------------------------------------------------------------ #

def _add_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Add all EMA columns and vol_ma20.
    Source = hl2 = (high+low)/2, matching Ripster's PineScript exactly."""
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    out["hl2"] = (out["high"] + out["low"]) / 2   # Ripster's EMA source
    for p in EMA_PERIODS:
        out[f"ema{p}"] = out["hl2"].ewm(span=p, adjust=False).mean()
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    return out


def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = df.resample(freq, label="right", closed="right").agg({
        "Open":  "first",
        "High":  "max",
        "Low":   "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna(subset=["Close"])
    return _add_emas(agg)


# ------------------------------------------------------------------ #
# Main backtest
# ------------------------------------------------------------------ #

def run_backtest(symbol: str, shares: int):
    print(f"\n{'='*55}")
    print(f"  Backtest: {symbol}  |  {shares} shares  |  Last 60 trading days")
    print(f"  (5-min bars used as proxy for 3-min entry timing)")
    print(f"{'='*55}")

    # 5-min data (including pre-market) for 60 days.
    # Pre-market used for PMH/PML levels; regular-session used for EMA + signals.
    print("  Downloading 60 days of 5-min data (incl. pre-market)...")
    raw = yf.download(symbol, period="60d", interval="5m",
                      progress=False, auto_adjust=True, prepost=True)

    if raw.empty:
        print("  ERROR: No data returned. Check the symbol and try again.")
        return

    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.tz_convert("US/Eastern")

    # Pre-market high / low per date (04:00–09:29 ET) for TYPE 4 breakout signal
    _pre = raw.between_time("04:00", "09:29")
    pmh_by_date = {}
    pml_by_date = {}
    for _dt, _grp in _pre.groupby(_pre.index.date):
        if not _grp.empty:
            pmh_by_date[_dt] = float(_grp["High"].max())
            pml_by_date[_dt] = float(_grp["Low"].min())

    # Regular-session bars only for the walk-forward simulation
    raw = raw.between_time("09:30", "16:00")

    # Build 10-min (trend) and 5-min (entry proxy for 3-min) DataFrames
    df_10m = _resample(raw, "10min")
    df_3m  = _resample(raw, "5min")   # 5-min used as 3-min proxy

    print(f"  10-min bars: {len(df_10m)}  |  5-min bars (3-min proxy): {len(df_3m)}")

    if len(df_10m) < MIN_BARS_10M:
        print(f"  WARNING: Only {len(df_10m)} 10-min bars — need {MIN_BARS_10M} for a "
              f"reliable 200 EMA. Signals may be inaccurate early on.")

    # ---- Walk-forward simulation ---------------------------------- #
    position   = None
    trades     = []

    last_date      = None
    trades_today   = 0
    lost_dir_today = None   # direction of last losing trade this day
    pmh            = None   # pre-market high for current date (TYPE 4)
    pml            = None   # pre-market low  for current date (TYPE 4)

    for i in range(MIN_BARS_3M, len(df_3m)):
        bar_time = df_3m.index[i]
        df_3m_now = df_3m.iloc[: i + 1]
        cur_3m    = df_3m_now.iloc[-1]

        # Latest 10-min bars up to this moment
        df_10m_now = df_10m[df_10m.index <= bar_time]
        if df_10m_now.empty:
            continue

        bar_date = bar_time.date()
        if bar_date != last_date:
            trades_today   = 0
            lost_dir_today = None
            last_date      = bar_date
            pmh = pmh_by_date.get(bar_date)
            pml = pml_by_date.get(bar_date)

        t = bar_time.time()
        is_eod   = (t.hour == MARKET_CLOSE_HOUR and
                    t.minute >= MARKET_CLOSE_MINUTE)

        if position and is_eod:
            exit_price = cur_3m["close"]
            pnl = ((exit_price - position["entry"]) * shares
                   if position["dir"] == "long"
                   else (position["entry"] - exit_price) * shares)
            trades.append({**position,
                           "exit": exit_price,
                           "exit_time": bar_time,
                           "pnl": pnl,
                           "reason": "EOD close"})
            _print_trade(trades[-1])
            if pnl <= 0:
                lost_dir_today = position["dir"]
            position = None

        # Stop taking new entries after the end-of-day cutoff
        no_new_entries = (t.hour > MARKET_CLOSE_HOUR or
                          (t.hour == MARKET_CLOSE_HOUR and
                           t.minute >= MARKET_CLOSE_MINUTE))

        # Update trend on every new 10-min bar
        trend = get_trend_10m(df_10m_now)

        # ---- Manage open position --------------------------------- #
        if position:
            # Update trailing stop
            new_stop = compute_trailing_stop(
                df_3m_now, position["dir"],
                position["stop"], position["entry"]
            )
            position["stop"] = new_stop

            # Trailing stop hit
            if position["dir"] == "long" and cur_3m["low"] <= position["stop"]:
                pnl = (position["stop"] - position["entry"]) * shares
                trades.append({**position,
                               "exit": position["stop"],
                               "exit_time": bar_time,
                               "pnl": pnl,
                               "reason": "trailing stop"})
                _print_trade(trades[-1])
                if pnl <= 0:
                    lost_dir_today = position["dir"]
                trades_today += 1
                position = None
                continue

            if position["dir"] == "short" and cur_3m["high"] >= position["stop"]:
                pnl = (position["entry"] - position["stop"]) * shares
                trades.append({**position,
                               "exit": position["stop"],
                               "exit_time": bar_time,
                               "pnl": pnl,
                               "reason": "trailing stop"})
                _print_trade(trades[-1])
                if pnl <= 0:
                    lost_dir_today = position["dir"]
                trades_today += 1
                position = None
                continue

            # 3-min cloud flip exit
            if should_exit_3m(df_3m_now, position["dir"]):
                exit_price = cur_3m["close"]
                pnl = ((exit_price - position["entry"]) * shares
                       if position["dir"] == "long"
                       else (position["entry"] - exit_price) * shares)
                trades.append({**position,
                               "exit": exit_price,
                               "exit_time": bar_time,
                               "pnl": pnl,
                               "reason": "3m cloud flip"})
                _print_trade(trades[-1])
                if pnl <= 0:
                    lost_dir_today = position["dir"]
                trades_today += 1
                position = None
            continue

        # ---- Entry check ----------------------------------------- #
        if no_new_entries or trades_today >= MAX_TRADES_PER_DAY:
            continue

        signal, stop_price = get_entry_signal_3m(
            df_3m_now, trend, bar_time=bar_time,
            pmh=pmh, pml=pml)
        if signal == "none":
            continue

        # Block same-direction re-entry after a losing trade today
        if signal == lost_dir_today:
            continue

        entry_price = cur_3m["close"]
        risk        = abs(entry_price - stop_price) * shares
        position = {
            "symbol":     symbol,
            "dir":        signal,
            "entry":      entry_price,
            "stop":       stop_price,
            "entry_time": bar_time,
            "risk":       risk,
        }
        print(f"\n  >> ENTRY  {signal.upper():<5} {bar_time.strftime('%a %m/%d %H:%M')}  "
              f"@ ${entry_price:.2f}  stop=${stop_price:.2f}  risk=${risk:.2f}  "
              f"[trade {trades_today + 1}/{MAX_TRADES_PER_DAY}]")

    # Close any position still open at end of data
    if position:
        last_close = df_3m.iloc[-1]["close"]
        pnl = ((last_close - position["entry"]) * shares
               if position["dir"] == "long"
               else (position["entry"] - last_close) * shares)
        trades.append({**position,
                       "exit": last_close,
                       "exit_time": df_3m.index[-1],
                       "pnl": pnl,
                       "reason": "end of data"})
        _print_trade(trades[-1])

    # ---- Summary -------------------------------------------------- #
    print(f"\n{'='*55}")
    print(f"  COMPLETED TRADES: {len(trades)}")
    if trades:
        total_pnl = sum(t["pnl"] for t in trades)
        winners   = [t for t in trades if t["pnl"] > 0]
        losers    = [t for t in trades if t["pnl"] <= 0]
        win_rate  = len(winners) / len(trades) * 100
        print(f"  Win rate : {win_rate:.0f}%  ({len(winners)}W / {len(losers)}L)")
        print(f"  Total P&L: ${total_pnl:+.2f}")
        if winners:
            print(f"  Avg win  : ${sum(t['pnl'] for t in winners)/len(winners):+.2f}")
        if losers:
            print(f"  Avg loss : ${sum(t['pnl'] for t in losers)/len(losers):+.2f}")
    print("="*55)


def _print_trade(t: dict):
    entry_ts = t["entry_time"].strftime("%a %m/%d %H:%M")
    exit_ts  = t["exit_time"].strftime("%H:%M")
    print(f"     CLOSE {t['dir'].upper():<5} {entry_ts} -> {exit_ts}  "
          f"${t['entry']:.2f} -> ${t['exit']:.2f}  "
          f"pnl=${t['pnl']:+.2f}  [{t['reason']}]")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("\nRIPSTER CLOUD BACKTEST")
    raw_symbols = input("Symbols to backtest (comma-separated, e.g. MU, NVDA): ")
    symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
    if not symbols:
        symbols = ["MU"]

    while True:
        try:
            shares = int(input("Shares per trade: "))
            if shares > 0:
                break
        except ValueError:
            pass
        print("  Enter a whole number greater than 0.")

    for sym in symbols:
        run_backtest(sym, shares)
