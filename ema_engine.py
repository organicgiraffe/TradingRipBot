import pandas as pd
from config import (EMA_PERIODS, MIN_BARS_10M, MIN_BARS_3M,
                    MAX_STOP_DISTANCE, MAX_STOP_PCT, MIN_STOP_PCT_LOWER,
                    BREAKEVEN_TRIGGER, RATCHET_START, RATCHET_GIVEBACK,
                    VOLUME_CONFIRM_MULT,
                    CLOUD_EXIT_BUFFER, GAP_THRESHOLD,
                    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
                    LAST_ENTRY_HOUR, LAST_ENTRY_MINUTE,
                    FRIDAY_OPEN_MINUTE,
                    RVOL_EXIT_MULT,
                    ATR_PERIODS, DTR_MAX_PCT)


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
# DTR / ATR ratio — Rip's range-exhaustion filter
# "DTR: 6.21 vs ATR: 7.59  82%" — if today's range has already used up
# most of the average daily range, the move is largely done.  Don't enter.
# ------------------------------------------------------------------ #

def compute_dtr_atr_ratio(df_10m: pd.DataFrame, target_date,
                          bar_time=None,
                          atr_periods: int = ATR_PERIODS) -> float:
    """
    Returns today's DTR/ATR ratio as a fraction (e.g. 0.82 = 82%).

    DTR = today's high-low range up to bar_time (from 10-min bars).
    ATR = simple average True Range of the past atr_periods trading days.
          Strictly uses data BEFORE target_date — no lookahead.

    Returns 0.0 when data is insufficient (caller treats 0 as "no filter").
    """
    if df_10m is None or df_10m.empty or len(df_10m) < 2:
        return 0.0

    # Collapse 10-min bars to daily OHLC
    df_daily = (df_10m.groupby(df_10m.index.date)
                      .agg(high=("high", "max"),
                           low=("low",  "min"),
                           close=("close", "last")))

    if len(df_daily) < atr_periods + 1:
        return 0.0

    # True Range (vectorised: max of H-L, |H-prevC|, |L-prevC|)
    hl = df_daily["high"] - df_daily["low"]
    hc = (df_daily["high"] - df_daily["close"].shift(1)).abs()
    lc = (df_daily["low"]  - df_daily["close"].shift(1)).abs()
    df_daily["tr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # ATR from the most-recent atr_periods days BEFORE today (no lookahead)
    past = df_daily[df_daily.index < target_date].dropna(subset=["tr"])
    if len(past) < atr_periods:
        return 0.0

    atr = float(past["tr"].tail(atr_periods).mean())
    if atr <= 0:
        return 0.0

    # DTR: today's range so far (up to bar_time, or full day if None)
    if bar_time is not None:
        today_bars = df_10m[
            (df_10m.index.date == target_date) & (df_10m.index <= bar_time)
        ]
    else:
        today_bars = df_10m[df_10m.index.date == target_date]

    if today_bars.empty:
        return 0.0

    dtr = float(today_bars["high"].max() - today_bars["low"].min())
    return dtr / atr


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
    pct = dist / entry_price
    return (pct >= MIN_STOP_PCT_LOWER and      # floor: tighter than 0.25% = noise
            pct <= MAX_STOP_PCT and            # ceiling: wider than 2.5% = too much risk
            dist <= MAX_STOP_DISTANCE * 3)     # hard dollar cap


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
                          current_stop: float, entry_price: float,
                          best_unrealised: float = 0.0) -> float:
    """
    Trail the stop to the current ema50 (the slow cloud's far edge).
    As price rises (longs), ema50 rises with it — we trail up.
    As price falls (shorts), ema50 falls — we trail down.

    Ratchet rule (replaces the old flat breakeven trigger):
      Pass best_unrealised = highest per-share profit seen so far.
      Once best_unrealised >= RATCHET_START ($3), the stop floor rises:
        floor = entry + max(0, best_unrealised - RATCHET_GIVEBACK)
      e.g. best +$6  → floor = entry + $3  (locks $300 on 100 shares)
           best +$10 → floor = entry + $7  (locks $700 on 100 shares)

      Using the high-water-mark (not current close) prevents the intrabar
      phantom stop where a single bar's high triggers the ratchet then the
      same bar's low immediately hits it.
    """
    cur = df_3m.iloc[-1]
    trail_to = cur.ema50

    if direction == "long":
        if best_unrealised >= RATCHET_START:
            floor = entry_price + max(0.0, best_unrealised - RATCHET_GIVEBACK)
            trail_to = max(trail_to, floor)
        return max(current_stop, trail_to)          # never move stop down

    else:  # short
        if best_unrealised >= RATCHET_START:
            floor = entry_price - max(0.0, best_unrealised - RATCHET_GIVEBACK)
            trail_to = min(trail_to, floor)
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
                        resistance: float = None) -> tuple[str, float, str]:
    """
    Ripster cloud flip — simple as it gets.

    LONG  when Cloud 2 (ema5/ema12) flips GREEN  and Cloud 3 (ema34/ema50) is GREEN.
    SHORT when Cloud 2 (ema5/ema12) flips RED    and Cloud 3 (ema34/ema50) is RED.

    Stop = ema50 (the far edge of the slow cloud — Ripster's defined risk level).
    If Rip's support / resistance levels are supplied, the tighter of ema50 vs
    that key level is used as the stop (whichever is closer to entry price).

    Volume must be above average.  No entries before 09:40 ET.

    Returns: (signal, stop_price, entry_reason)
      signal       - 'long' | 'short' | 'none'
      stop_price   - initial stop level (0.0 when signal is 'none')
      entry_reason - 'cloud_flip' | 'pmh_breakout' | 'pml_breakdown' | ''

    Live-trading note — early entry on volume:
      Don't wait for the bar to close.  As soon as ema5 crosses ema12 on the
      live 3-min bar AND volume is already tracking above average mid-candle,
      that IS the signal.  Enter immediately; every second of delay costs slippage
      on a momentum move.
    """
    if len(df_3m) < MIN_BARS_3M:
        return "none", 0.0, ""

    cur  = df_3m.iloc[-1]
    prev = df_3m.iloc[-2]
    entry_price = cur.close

    # No entries before the first 3-min bar closes (09:33 ET).
    # On Fridays (options expiry / Lotto Friday) push to 09:45 — the open
    # bar is hit by violent expiry-driven moves that stop out clean setups.
    if bar_time is not None:
        is_friday   = (bar_time.weekday() == 4)
        open_minute = FRIDAY_OPEN_MINUTE if is_friday else MARKET_OPEN_MINUTE
        if (bar_time.hour < MARKET_OPEN_HOUR or
                (bar_time.hour == MARKET_OPEN_HOUR
                 and bar_time.minute < open_minute)):
            return "none", 0.0, ""

    # No new entries after 15:00 — not enough time for trade to develop before close
    if bar_time is not None:
        if (bar_time.hour > LAST_ENTRY_HOUR or
                (bar_time.hour == LAST_ENTRY_HOUR
                 and bar_time.minute >= LAST_ENTRY_MINUTE)):
            return "none", 0.0, ""

    # Volume gate — above-average participation confirms the move is real
    if cur.vol_ma20 > 0 and cur.volume < VOLUME_CONFIRM_MULT * cur.vol_ma20:
        return "none", 0.0, ""

    # Trend gate — 10m must show a clear direction.
    # When trend is 'none', both clouds are mixed or price is straddling ema200.
    # C2 flips in that environment are noise — skip everything.
    if trend == "none":
        return "none", 0.0, ""

    # Cloud 2 flip: ema5 crosses ema12
    c2_flip_long  = prev.ema5 <= prev.ema12 and cur.ema5 > cur.ema12
    c2_flip_short = prev.ema5 >= prev.ema12 and cur.ema5 < cur.ema12

    # Cloud 3 direction: ema34 vs ema50
    c3_green = cur.ema34 > cur.ema50
    c3_red   = cur.ema34 < cur.ema50

    # 10-min trend filter — don't fight the established macro trend.
    # 'bullish' → skip shorts.  'bearish' → skip longs.
    if trend == "bearish" and c2_flip_long:
        return "none", 0.0, ""
    if trend == "bullish" and c2_flip_short:
        return "none", 0.0, ""

    # ---- LONG: C2 just flipped green, C3 is green ----
    if c2_flip_long and c3_green:
        stop = cur.ema50
        # Rip's support level: if it's higher than ema50, use it — tighter and
        # more meaningful (break of support = trade is wrong)
        if support is not None and support > stop and support < entry_price:
            stop = support
        if _stop_ok(entry_price, stop, "long"):
            return "long", stop, "cloud_flip"

    # ---- SHORT: C2 just flipped red, C3 is red ----
    if c2_flip_short and c3_red:
        stop = cur.ema50
        # Rip's resistance level: if it's lower than ema50, use it — tighter
        # (break back above resistance = trade is wrong)
        if resistance is not None and resistance < stop and resistance > entry_price:
            stop = resistance
        if _stop_ok(entry_price, stop, "short"):
            return "short", stop, "cloud_flip"

    # ---- PMH breakout: first bar to close above pre-market high, C3 green ----
    if pmh is not None and c3_green and cur.close > pmh and prev.close <= pmh:
        gap_pct = (entry_price - cur.ema50) / entry_price if entry_price > 0 else 0
        stop    = cur.ema12 if gap_pct > GAP_THRESHOLD else cur.ema50
        if support is not None and support > stop and support < entry_price:
            stop = support
        if _stop_ok(entry_price, stop, "long"):
            return "long", stop, "pmh_breakout"

    # ---- PML breakdown: first bar to close below pre-market low, C3 red ----
    if pml is not None and c3_red and cur.close < pml and prev.close >= pml:
        gap_pct = (cur.ema50 - entry_price) / entry_price if entry_price > 0 else 0
        stop    = cur.ema12 if gap_pct > GAP_THRESHOLD else cur.ema50
        if resistance is not None and resistance < stop and resistance > entry_price:
            stop = resistance
        if _stop_ok(entry_price, stop, "short"):
            return "short", stop, "pml_breakdown"

    return "none", 0.0, ""


# ------------------------------------------------------------------ #
# 3-min exit — fast cloud (5/12) flip
# ------------------------------------------------------------------ #

def should_exit_3m(df_3m: pd.DataFrame, direction: str) -> bool:
    """
    Exit long:  fast cloud (ema5/ema12) flips RED  — prev bar ema5>=ema12, cur ema5<ema12
    Exit short: fast cloud (ema5/ema12) flips GREEN — prev bar ema5<=ema12, cur ema5>ema12

    This captures the momentum move and gets out early — exit when the fast cloud
    reverses, not when price has already fallen all the way into the slow cloud.
    The trailing stop at ema50 still acts as a hard floor if price gaps through.
    """
    if len(df_3m) < 2:
        return False

    cur  = df_3m.iloc[-1]
    prev = df_3m.iloc[-2]

    if direction == "long":
        return prev.ema5 >= prev.ema12 and cur.ema5 < cur.ema12
    if direction == "short":
        return prev.ema5 <= prev.ema12 and cur.ema5 > cur.ema12
    return False


# ------------------------------------------------------------------ #
# 10-min exit — fast cloud (5/12) flip on the higher timeframe
# ------------------------------------------------------------------ #

def should_exit_10m(df_10m: pd.DataFrame, direction: str) -> bool:
    """
    Exit when the 10-min fast cloud (ema5/ema12) flips against the position.
    Gives the trade more room than the 3-min exit — only closes when the
    higher-timeframe momentum has genuinely reversed.
    """
    if len(df_10m) < 2:
        return False
    cur  = df_10m.iloc[-1]
    prev = df_10m.iloc[-2]
    if direction == "long":
        return prev.ema5 >= prev.ema12 and cur.ema5 < cur.ema12
    if direction == "short":
        return prev.ema5 <= prev.ema12 and cur.ema5 > cur.ema12
    return False


# ------------------------------------------------------------------ #
# RVOL exit — relative volume dried up, momentum is gone
# ------------------------------------------------------------------ #

def should_exit_rvol(df_3m: pd.DataFrame) -> bool:
    """
    Exit when the current 3-min bar's volume drops below RVOL_EXIT_MULT × average.
    When the crowd stops participating the move is over — don't wait for the cloud.
    """
    if len(df_3m) < 1:
        return False
    cur = df_3m.iloc[-1]
    if cur.vol_ma20 <= 0:
        return False
    return (cur.volume / cur.vol_ma20) < RVOL_EXIT_MULT
