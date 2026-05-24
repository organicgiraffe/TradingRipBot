import pandas as pd
from config import (EMA_PERIODS, MIN_BARS_10M, MIN_BARS_3M,
                    MAX_STOP_DISTANCE, MAX_STOP_PCT,
                    BREAKEVEN_TRIGGER, VOLUME_CONFIRM_MULT,
                    CLOUD_EXIT_BUFFER,
                    LAST_ENTRY_HOUR, LAST_ENTRY_MINUTE)


# ------------------------------------------------------------------ #
# EMA calculation — SOURCE IS hl2 = (high+low)/2
# This matches Ripster's PineScript exactly:
#   src = input(title="Source", type=input.source, defval=hl2)
# ------------------------------------------------------------------ #

def compute_emas(bars: list) -> pd.DataFrame:
    """Convert IBKR bar objects to a DataFrame with all EMA columns.
    All EMAs computed on hl2 = (high+low)/2, matching Ripster's indicator."""
    df = pd.DataFrame({
        "time":   [b.date for b in bars],
        "open":   [b.open for b in bars],
        "high":   [b.high for b in bars],
        "low":    [b.low for b in bars],
        "close":  [b.close for b in bars],
        "volume": [b.volume for b in bars],
    })
    df.set_index("time", inplace=True)
    df["hl2"] = (df["high"] + df["low"]) / 2   # Ripster's EMA source
    for p in EMA_PERIODS:
        df[f"ema{p}"] = df["hl2"].ewm(span=p, adjust=False).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df


# ------------------------------------------------------------------ #
# Stop quality gate — percentage-based so it works across all price levels
# ------------------------------------------------------------------ #

def _stop_ok(entry_price: float, stop_price: float, direction: str) -> bool:
    """
    Accept the trade only if the stop is within MAX_STOP_PCT of entry price.
    2.5% of a $400 stock = $10 max stop — scales correctly for any price.
    Also enforces a hard MAX_STOP_DISTANCE dollar floor as a secondary cap.
    """
    dist = (entry_price - stop_price if direction == "long"
            else stop_price - entry_price)
    if dist <= 0:
        return False
    return (dist / entry_price <= MAX_STOP_PCT and
            dist <= MAX_STOP_DISTANCE * 3)   # hard cap = 3x legacy limit


# ------------------------------------------------------------------ #
# Stop loss — ema50 is Ripster's defined risk level
# "Whenever you long or short, that 34-50 cloud is your risk level."
# ------------------------------------------------------------------ #

def compute_stop(df_3m: pd.DataFrame, direction: str, entry_price: float) -> float:
    """
    Stop = ema50 (the far edge of the slow 34/50 cloud).

    Bullish: ema34 > ema50  → ema50 is BELOW price  → long stop
    Bearish: ema34 < ema50  → ema50 is ABOVE price  → short stop
    """
    cur = df_3m.iloc[-1]
    return cur.ema50


# ------------------------------------------------------------------ #
# Trailing stop — follows ema50 as it moves with price
# ------------------------------------------------------------------ #

def compute_trailing_stop(df_3m: pd.DataFrame, direction: str,
                          current_stop: float, entry_price: float) -> float:
    """
    Trail the stop to the current ema50 (the slow cloud's far edge).
    As price rises (longs), ema50 rises with it — we trail up.
    As price falls (shorts), ema50 falls — we trail down.

    Rules:
      - Stop never moves against the position.
      - Once BREAKEVEN_TRIGGER ($5) profitable, stop locks at entry minimum.
    """
    cur = df_3m.iloc[-1]
    trail_to = cur.ema50

    if direction == "long":
        unrealised = cur.close - entry_price
        if unrealised >= BREAKEVEN_TRIGGER:
            trail_to = max(trail_to, entry_price)   # never below entry
        return max(current_stop, trail_to)          # never move stop down

    else:  # short
        unrealised = entry_price - cur.close
        if unrealised >= BREAKEVEN_TRIGGER:
            trail_to = min(trail_to, entry_price)   # never above entry
        return min(current_stop, trail_to)          # never move stop up


# ------------------------------------------------------------------ #
# 10-min trend — direction filter only (established 2-bar alignment)
# ------------------------------------------------------------------ #

def get_trend_10m(df_10m: pd.DataFrame) -> str:
    """
    Returns 'bullish', 'bearish', or 'none'.

    Requires BOTH current AND previous 10-min bar to agree on:
      - Cloud 2 direction (ema5 vs ema12)
      - Cloud 3 direction (ema34 vs ema50)
      - Price above/below 200 EMA (all computed on hl2)
    """
    if len(df_10m) < MIN_BARS_10M:
        return "none"

    cur  = df_10m.iloc[-1]
    prev = df_10m.iloc[-2]

    def _both_bull(r): return r.ema5 > r.ema12 and r.ema34 > r.ema50
    def _both_bear(r): return r.ema5 < r.ema12 and r.ema34 < r.ema50

    if _both_bull(cur) and _both_bull(prev) and cur.hl2 > cur.ema200:
        return "bullish"
    if _both_bear(cur) and _both_bear(prev) and cur.hl2 < cur.ema200:
        return "bearish"
    return "none"


# ------------------------------------------------------------------ #
# 3-min entry signal
# ------------------------------------------------------------------ #

def get_entry_signal_3m(df_3m: pd.DataFrame, trend: str = None,
                        bar_time=None,
                        pmh: float = None, pml: float = None,
                        support: float = None,
                        resistance: float = None) -> tuple[str, float]:
    """
    Ripster cloud flip — simple as it gets.

    LONG  when Cloud 2 (ema5/ema12) flips GREEN  and Cloud 3 (ema34/ema50) is GREEN.
    SHORT when Cloud 2 (ema5/ema12) flips RED    and Cloud 3 (ema34/ema50) is RED.

    Stop = ema50 (the far edge of the slow cloud — Ripster's defined risk level).
    If Rip's support / resistance levels are supplied, the tighter of ema50 vs
    that key level is used as the stop (whichever is closer to entry price).

    Volume must be above average.  No entries before 09:40 ET.

    Live-trading note — early entry on volume:
      Don't wait for the bar to close.  As soon as ema5 crosses ema12 on the
      live 3-min bar AND volume is already tracking above average mid-candle,
      that IS the signal.  Enter immediately; every second of delay costs slippage
      on a momentum move.
    """
    if len(df_3m) < MIN_BARS_3M:
        return "none", 0.0

    cur  = df_3m.iloc[-1]
    prev = df_3m.iloc[-2]
    entry_price = cur.close

    # No entries in the first 10 minutes — let the open settle
    if bar_time is not None and bar_time.hour == 9 and bar_time.minute < 40:
        return "none", 0.0

    # No new entries after 15:00 — not enough time for trade to develop before close
    if bar_time is not None:
        if (bar_time.hour > LAST_ENTRY_HOUR or
                (bar_time.hour == LAST_ENTRY_HOUR
                 and bar_time.minute >= LAST_ENTRY_MINUTE)):
            return "none", 0.0

    # Volume gate — above-average participation confirms the move is real
    if cur.vol_ma20 > 0 and cur.volume < VOLUME_CONFIRM_MULT * cur.vol_ma20:
        return "none", 0.0

    # Cloud 2 flip: ema5 crosses ema12
    c2_flip_long  = prev.ema5 <= prev.ema12 and cur.ema5 > cur.ema12
    c2_flip_short = prev.ema5 >= prev.ema12 and cur.ema5 < cur.ema12

    # Cloud 3 direction: ema34 vs ema50
    c3_green = cur.ema34 > cur.ema50
    c3_red   = cur.ema34 < cur.ema50

    # 10-min trend filter — don't fight the established macro trend.
    # 'none' means unclear → allow both directions.
    # 'bullish' → skip shorts.  'bearish' → skip longs.
    if trend == "bearish" and c2_flip_long:
        return "none", 0.0
    if trend == "bullish" and c2_flip_short:
        return "none", 0.0

    # ---- LONG: C2 just flipped green, C3 is green ----
    if c2_flip_long and c3_green:
        stop = cur.ema50
        # Rip's support level: if it's higher than ema50, use it — tighter and
        # more meaningful (break of support = trade is wrong)
        if support is not None and support > stop and support < entry_price:
            stop = support
        if _stop_ok(entry_price, stop, "long"):
            return "long", stop

    # ---- SHORT: C2 just flipped red, C3 is red ----
    if c2_flip_short and c3_red:
        stop = cur.ema50
        # Rip's resistance level: if it's lower than ema50, use it — tighter
        # (break back above resistance = trade is wrong)
        if resistance is not None and resistance < stop and resistance > entry_price:
            stop = resistance
        if _stop_ok(entry_price, stop, "short"):
            return "short", stop

    return "none", 0.0


# ------------------------------------------------------------------ #
# 3-min exit — slow cloud (34/50) violation
# ------------------------------------------------------------------ #

def should_exit_3m(df_3m: pd.DataFrame, direction: str) -> bool:
    """
    Exit long:  3-min candle closes below ema34 (price enters slow cloud from above)
    Exit short: 3-min candle closes above ema34 (price enters slow cloud from below)

    Ripster: "that 34-50 cloud is your risk level."
    We hold through 5/12 wiggles (normal trend pullbacks) and only exit
    when price actually breaks INTO the slow cloud.
    The trailing stop at ema50 acts as the absolute floor below this.
    """
    if len(df_3m) < 1:
        return False

    cur = df_3m.iloc[-1]

    if direction == "long":
        # Must close CLEARLY below ema34, not just graze the edge.
        # CLOUD_EXIT_BUFFER ($0.10) prevents false exits on consolidation bars
        # where close ≈ ema34 by a fraction of a cent.
        return cur.close < cur.ema34 - CLOUD_EXIT_BUFFER
    if direction == "short":
        return cur.close > cur.ema34 + CLOUD_EXIT_BUFFER
    return False
