"""
manual_backtest.py — replay today's session with yfinance data + Rip's levels.
Run: python manual_backtest.py
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import yfinance as yf
import pandas as pd

# Add bot directory to path so we can import ema_engine
sys.path.insert(0, r"C:\Users\nicol\OneDrive\Documents\Tradingbot")

from ema_engine import compute_emas, get_trend_10m, get_entry_signal_3m
from config import (
    VOLUME_CONFIRM_MULT, MIN_BARS_10M, MIN_BARS_3M,
    RATCHET_START, RATCHET_GIVEBACK, PROFIT_TARGET_SHARE,
    FIXED_SHARES, FIXED_SHARES_HIGH, HIGH_PRICE_THRESHOLD,
    MAX_RISK_DOLLARS, FIRST_ENTRY_MINUTE,
    LAST_ENTRY_HOUR, LAST_ENTRY_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
    RVOL_EXIT_MULT, LEVEL_PROX_LONG, LEVEL_PROX_SHORT,
)

# ── Today's plan from the log ──────────────────────────────────────────────
PLAN = {
    "TSLA":  {"support": 428.63, "resistance": 429.80},
    "NVDA":  {"support": 217.41, "resistance": 218.52},
    "AMD":   {"support": 475.80, "resistance": 480.00},
    "META":  {"support": 610.00, "resistance": 612.00},
    "GOOGL": {"support": 384.30, "resistance": 385.20},
    "MU":    {"support": 800.00, "resistance": None},
    "ARM":   {"support": 309.75, "resistance": 314.50},
    "IONQ":  {"support":  63.80, "resistance":  65.00},
    "OKLO":  {"support":  71.00, "resistance":  73.00},
}

TODAY = pd.Timestamp("2026-05-26").date()


class Bar:
    def __init__(self, date, open_, high, low, close, volume):
        self.date   = date
        self.open   = open_
        self.high   = high
        self.low    = low
        self.close  = close
        self.volume = int(volume)


def download_1m(sym):
    """Download 1-minute bars (yfinance supports 1m; 3m/10m are not available)."""
    df = yf.download(sym, period="5d", interval="1m",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")
    # RTH only: 09:30–16:00
    df = df.between_time("09:30", "15:59")
    return df


def resample_bars(df1m, minutes):
    """Aggregate 1-min bars into N-minute bars (OHLCV)."""
    rule = f"{minutes}min"
    agg = df1m.resample(rule, closed="left", label="left").agg(
        open=("open", "first"),
        high=("high",  "max"),
        low=("low",   "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])
    # Align bar labels to bar CLOSE time (matching IBKR convention)
    agg.index = agg.index + pd.Timedelta(minutes=minutes)
    return agg


def df_to_bars(df):
    bars = []
    for row in df.itertuples():
        bars.append(Bar(row.Index, row.open, row.high, row.low, row.close, row.volume))
    return bars


def compute_ratchet_stop(signal, stop_cur, entry, best_unreal, ema50):
    trail = ema50
    if signal == "long":
        if best_unreal >= RATCHET_START:
            floor = entry + max(0.0, best_unreal - RATCHET_GIVEBACK)
            trail = max(trail, floor)
        return max(stop_cur, trail)
    else:
        if best_unreal >= RATCHET_START:
            floor = entry - max(0.0, best_unreal - RATCHET_GIVEBACK)
            trail = min(trail, floor)
        return min(stop_cur, trail)


# ── Main ──────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print(f"  MANUAL BACKTEST  —  {TODAY}  —  Ripster Cloud rules (perfect data)")
print("=" * 72)
print()

all_results = []

for sym, levels in PLAN.items():
    sup = levels["support"]
    res = levels["resistance"]

    try:
        raw1m = download_1m(sym)
        if raw1m.empty:
            print(f"  {sym}: no data")
            continue
        raw3  = resample_bars(raw1m, 3)
        raw10 = resample_bars(raw1m, 10)
    except Exception as e:
        print(f"  {sym}: download error — {e}")
        continue

    bars10_all = df_to_bars(raw10)
    bars3_all  = df_to_bars(raw3)

    # Today's 3-min bars only
    today3 = [b for b in bars3_all if pd.Timestamp(b.date).date() == TODAY]
    if not today3:
        print(f"  {sym}: no bars for today")
        continue

    # Determine share size from typical price today
    typical_px = today3[len(today3)//2].close
    shares = FIXED_SHARES_HIGH if typical_px >= HIGH_PRICE_THRESHOLD else FIXED_SHARES

    # Scan for entry signals bar by bar
    signals_found = []
    for i, bar in enumerate(today3):
        bar_time = pd.Timestamp(bar.date)

        # Entry time gates
        if bar_time.hour == 9 and bar_time.minute < FIRST_ENTRY_MINUTE:
            continue
        if (bar_time.hour > LAST_ENTRY_HOUR or
                (bar_time.hour == LAST_ENTRY_HOUR and bar_time.minute >= LAST_ENTRY_MINUTE)):
            break

        # Build lookback slices up to this bar
        bars10_so_far = [b for b in bars10_all if pd.Timestamp(b.date) <= bar_time]
        bars3_so_far  = bars3_all[:bars3_all.index(bar) + 1]

        if len(bars10_so_far) < 5 or len(bars3_so_far) < MIN_BARS_3M:
            continue

        df10 = compute_emas(bars10_so_far)
        df3  = compute_emas(bars3_so_far)
        trend = get_trend_10m(df10)

        signal, stop_px, reason = get_entry_signal_3m(
            df3, trend,
            bar_time=bar_time.to_pydatetime(),
            support=sup, resistance=res)

        if signal == "none":
            continue

        entry   = df3.iloc[-1].close
        stop_dist = abs(entry - stop_px)
        risk    = stop_dist * shares

        if risk > MAX_RISK_DOLLARS:
            continue

        # Level proximity gate
        if signal == "long" and res is not None:
            if entry > res * (1 + LEVEL_PROX_LONG):
                continue
        if signal == "short" and sup is not None:
            if entry < sup * (1 - LEVEL_PROX_SHORT):
                continue

        signals_found.append((i, bar_time, signal, entry, stop_px, stop_dist, reason, trend))

    # For each signal, simulate the trade forward
    for entry_i, bar_time, signal, entry, initial_stop, stop_dist, reason, trend in signals_found:
        stop_cur     = initial_stop
        best_unreal  = 0.0
        half_done    = False
        half_pnl     = 0.0
        remaining    = shares
        exit_price   = None
        exit_time    = None
        exit_rsn     = None
        peak_unreal  = 0.0

        future = today3[entry_i + 1:]

        for fb in future:
            fb_time = pd.Timestamp(fb.date)

            # EOD forced close
            if (fb_time.hour > MARKET_CLOSE_HOUR or
                    (fb_time.hour == MARKET_CLOSE_HOUR and
                     fb_time.minute >= MARKET_CLOSE_MINUTE)):
                exit_price = fb.close
                exit_time  = fb_time
                exit_rsn   = "eod_close"
                break

            # Rebuild df for EMA computation
            bars3_fwd = bars3_all[:bars3_all.index(fb) + 1]
            df3f  = compute_emas(bars3_fwd)
            cur   = df3f.iloc[-1]

            unreal = (cur.close - entry) if signal == "long" else (entry - cur.close)
            best_unreal = max(best_unreal, unreal)
            peak_unreal = max(peak_unreal, unreal)

            # Ratchet trailing stop
            stop_cur = compute_ratchet_stop(signal, stop_cur, entry, best_unreal, cur.ema50)

            # Half-exit at level OR profit target
            if not half_done:
                hpx  = None
                hrsn = "half@level"
                if signal == "long" and res is not None and fb.high >= res:
                    hpx = res
                elif signal == "short" and sup is not None and fb.low <= sup:
                    hpx = sup
                has_level = (res is not None) if signal == "long" else (sup is not None)
                if hpx is None and not has_level and unreal >= PROFIT_TARGET_SHARE:
                    hpx  = cur.close
                    hrsn = "half@target"
                if hpx is not None:
                    hp = ((hpx - entry) if signal == "long" else (entry - hpx)) * (shares // 2)
                    if hp > 0:
                        half_pnl  = hp
                        remaining = shares - shares // 2
                    half_done = True

            # Hard stop
            if signal == "long" and fb.low <= stop_cur:
                exit_price = stop_cur
                exit_time  = fb_time
                exit_rsn   = "stop"
                break
            if signal == "short" and fb.high >= stop_cur:
                exit_price = stop_cur
                exit_time  = fb_time
                exit_rsn   = "stop"
                break

            # RVOL + C2 against (new rule)
            stop_locked = (stop_cur > entry) if signal == "long" else (stop_cur < entry)
            c2_against  = (cur.ema5 < cur.ema12) if signal == "long" else (cur.ema5 > cur.ema12)
            rvol_ok = cur.vol_ma20 > 0 and (cur.volume / cur.vol_ma20) < RVOL_EXIT_MULT
            if rvol_ok and not stop_locked and c2_against:
                exit_price = cur.close
                exit_time  = fb_time
                exit_rsn   = "rvol+C2"
                break

        if exit_price is None:
            exit_price = today3[-1].close
            exit_time  = pd.Timestamp(today3[-1].date)
            exit_rsn   = "still_open"

        runner_pnl = ((exit_price - entry) if signal == "long" else (entry - exit_price)) * remaining
        total_pnl  = half_pnl + runner_pnl
        held_min   = int((exit_time - bar_time).total_seconds() // 60) if exit_time else 0
        tag = "WIN " if total_pnl > 0 else ("EVEN" if total_pnl == 0 else "LOSS")

        all_results.append({
            "sym": sym, "time": bar_time, "signal": signal.upper(),
            "entry": entry, "initial_stop": initial_stop,
            "dist": stop_dist, "shares": shares, "risk": stop_dist * shares,
            "reason": reason, "trend": trend,
            "half_pnl": half_pnl, "runner_pnl": runner_pnl, "total_pnl": total_pnl,
            "exit_px": exit_price, "exit_time": exit_time,
            "exit_rsn": exit_rsn, "held": held_min, "tag": tag,
            "peak": peak_unreal,
        })

# ── Print summary ──────────────────────────────────────────────────────────
print(f"  {'TIME':<5}  {'SYM':<5}  {'DIR':<5}  {'ENTRY':>7}  {'STOP':>7}  {'RISK':>5}  "
      f"{'PNL':>7}  {'PEAK':>7}  {'EXIT':>13}  HELD   SIGNAL")
print(f"  {'-'*110}")

for r in all_results:
    sign = "+" if r["total_pnl"] >= 0 else ""
    pk   = f"+${r['peak']:.2f}" if r["peak"] >= 0 else f"-${abs(r['peak']):.2f}"
    print(
        f"  {r['time'].strftime('%H:%M')}  {r['sym']:<5}  {r['signal']:<5}  "
        f"${r['entry']:>6.2f}  ${r['initial_stop']:>6.2f}  ${r['risk']:>4.0f}  "
        f"  {sign}${abs(r['total_pnl']):>5.0f}  "
        f"{pk:>7}  {r['exit_rsn']:<14}  {r['held']:>3}m  "
        f"[{r['reason']}] trend={r['trend']}"
        + (f"  (half+${r['half_pnl']:.0f} runner {'+' if r['runner_pnl']>=0 else ''}"
           f"${r['runner_pnl']:.0f})" if r["half_pnl"] else "")
    )

if all_results:
    total = sum(r["total_pnl"] for r in all_results)
    wins  = sum(1 for r in all_results if r["total_pnl"] > 0)
    loss  = sum(1 for r in all_results if r["total_pnl"] <= 0)
    print(f"\n  {'-'*110}")
    print(f"  {len(all_results)} signal(s)   {wins}W / {loss}L   TOTAL P&L: ${total:+.0f}")
    print()
    # Per-symbol breakdown
    print("  PER SYMBOL:")
    syms_seen = dict()
    for r in all_results:
        syms_seen.setdefault(r["sym"], []).append(r["total_pnl"])
    for s, pnls in syms_seen.items():
        sign = "+" if sum(pnls) >= 0 else ""
        print(f"    {s:<6}  {sign}${sum(pnls):.0f}  ({len(pnls)} trade(s))")
else:
    print("  No valid signals fired today with the current rules.")

print()
