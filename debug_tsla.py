"""
debug_tsla.py -- bar-by-bar trace of TSLA May 22 to show why signals fire or miss.

Run:  python debug_tsla.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf
sys.path.insert(0, ".")

from config import (EMA_PERIODS, MIN_BARS_3M, MIN_BARS_10M,
                    MAX_TRADES_PER_DAY, MAX_STOP_PCT, MAX_STOP_DISTANCE,
                    BREAKEVEN_TRIGGER, VOLUME_CONFIRM_MULT)
from ema_engine import _taking_off, _stop_ok, compute_stop


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


def main():
    import datetime
    target = datetime.date(2026, 5, 22)

    print("Downloading TSLA 5-min data (60 days)...")
    raw = yf.download("TSLA", period="60d", interval="5m",
                      progress=False, auto_adjust=True)
    if raw.empty:
        print("No data returned."); return
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.tz_convert("US/Eastern").between_time("09:30", "16:00")

    df_3m  = _resample(raw, "5min")   # 5-min proxy for 3-min bars (yfinance limit)
    df_10m = _resample(raw, "10min")

    print(f"5-min bars total: {len(df_3m)}")
    print(f"10-min bars total: {len(df_10m)}")

    # Find bars for today only
    today_3m = df_3m[df_3m.index.date == target]
    if today_3m.empty:
        print(f"No 5-min bars for {target}"); return
    print(f"Today's 5-min bars: {len(today_3m)}")

    print(f"\n{'-'*110}")
    print(f"{'TIME':<6}  {'CLOSE':>7}  {'ema5':>7}  {'ema12':>7}  {'ema34':>7}  {'ema50':>7}  "
          f"{'TREND':>8}  {'TAKOFF':>6}  {'VOL_OK':>6}  {'C2_DIR':>6}  {'STOP_OK':>7}  NOTE")
    print(f"{'-'*110}")

    from ema_engine import get_trend_10m

    prev_trend = "none"

    for i in range(len(df_3m)):
        bar_time = df_3m.index[i]
        if bar_time.date() != target:
            continue

        df_3m_now  = df_3m.iloc[:i + 1]
        df_10m_now = df_10m[df_10m.index <= bar_time]
        cur        = df_3m_now.iloc[-1]
        prev_bar   = df_3m_now.iloc[-2] if len(df_3m_now) >= 2 else cur

        if df_10m_now.empty or len(df_3m_now) < MIN_BARS_3M:
            bar_count = len(df_3m_now)
            if bar_count >= MIN_BARS_3M - 5:   # only print near-threshold
                print(f"{bar_time.strftime('%H:%M')}  (warming up -- {bar_count}/{MIN_BARS_3M} bars)")
            continue

        trend = get_trend_10m(df_10m_now)

        # TYPE 1 checks
        cur_all_bull  = cur.ema5  > cur.ema12 and cur.ema34 > cur.ema50
        prev_all_bull = prev_bar.ema5 > prev_bar.ema12 and prev_bar.ema34 > prev_bar.ema50
        cur_all_bear  = cur.ema5  < cur.ema12 and cur.ema34 < cur.ema50
        prev_all_bear = prev_bar.ema5 < prev_bar.ema12 and prev_bar.ema34 < prev_bar.ema50

        vol_ok = (cur.vol_ma20 <= 0 or cur.volume >= VOLUME_CONFIRM_MULT * cur.vol_ma20)

        # TYPE 2 checks
        taking_off_long  = _taking_off(cur, "long")
        taking_off_short = _taking_off(cur, "short")
        cloud2_bull      = cur.ema5 > cur.ema12
        cloud2_bear      = cur.ema5 < cur.ema12

        stop  = cur.ema50
        entry = cur.close
        stop_ok_long  = _stop_ok(entry, stop, "long")

        dist_pct = abs(entry - stop) / entry * 100

        # Build note
        note = ""
        if cur_all_bull and not prev_all_bull and cur.hl2 > cur.ema200 and vol_ok:
            note = "** TYPE1 LONG (cloud flip)"
        elif cur_all_bear and not prev_all_bear and cur.hl2 < cur.ema200 and vol_ok:
            note = "** TYPE1 SHORT (cloud flip)"
        elif trend == "bullish" and taking_off_long and cloud2_bull:
            if stop_ok_long:
                note = "** TYPE2 LONG (taking off)"
            else:
                note = f"MISS TYPE2 LONG -- stop too far ({dist_pct:.1f}% > {MAX_STOP_PCT*100:.1f}%)"
        elif trend == "bullish" and taking_off_long and not cloud2_bull:
            note = f"MISS TYPE2 LONG -- C2 bear (ema5={cur.ema5:.2f} < ema12={cur.ema12:.2f})"
        elif trend == "bullish":
            levels = [cur.ema9, cur.ema12, cur.ema34, cur.ema50]
            names  = ["ema9", "ema12", "ema34", "ema50"]
            dists  = [f"{n}:{abs(cur.low - lvl):.2f}" for n, lvl in zip(names, levels)]
            low_to_levels = f"low={cur.low:.2f} [{','.join(dists)}]"
            note = f"trend=bull no_touch {low_to_levels}"

        if trend != prev_trend:
            note = f"[TREND->{trend.upper()}]  " + note
            prev_trend = trend

        # Also build TYPE1 summary even when trend is none
        t1_long  = cur_all_bull and not prev_all_bull and cur.hl2 > cur.ema200
        t1_short = cur_all_bear and not prev_all_bear and cur.hl2 < cur.ema200

        # Always show morning bars + anything interesting
        morning = bar_time.hour < 13 or (bar_time.hour == 13 and bar_time.minute < 30)
        if morning and not note:
            if t1_long:
                note = f"** TYPE1 LONG (flip) vol_ok={vol_ok}"
            elif t1_short:
                note = f"** TYPE1 SHORT (flip) vol_ok={vol_ok}"
            elif cur_all_bull:
                note = f"C1+C3=BULL no_flip prev_bull={prev_all_bull}"
            elif cur_all_bear:
                note = f"C1+C3=BEAR no_flip prev_bear={prev_all_bear}"
            else:
                note = f"mixed C2={'bull' if cloud2_bull else 'bear'} C3={'bull' if cur.ema34>cur.ema50 else 'bear'}"

        interesting = (note.startswith("**") or note.startswith("MISS") or
                       "[TREND->" in note or trend == "bullish" or morning)

        if interesting:
            print(
                f"{bar_time.strftime('%H:%M')}  "
                f"{cur.close:>7.2f}  {cur.ema5:>7.2f}  {cur.ema12:>7.2f}  "
                f"{cur.ema34:>7.2f}  {cur.ema50:>7.2f}  "
                f"{trend:>8}  {'Y' if taking_off_long else 'n':>6}  "
                f"{'Y' if vol_ok else 'n':>6}  "
                f"{'bull' if cloud2_bull else 'bear':>6}  "
                f"{'Y' if stop_ok_long else 'n':>7}  {note}"
            )


if __name__ == "__main__":
    main()
