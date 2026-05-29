import pandas as pd
from config import (EMA_PERIODS, MIN_BARS_10M, MIN_BARS_3M,
                    MAX_STOP_DISTANCE, MAX_STOP_PCT, MIN_STOP_PCT_LOWER,
                    BREAKEVEN_TRIGGER, RATCHET_START, RATCHET_GIVEBACK,
                    VOLUME_CONFIRM_MULT,
                    CLOUD_EXIT_BUFFER, GAP_THRESHOLD, CLOUD_CONT_MAX_DIST,
                    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
                    GAP_ENTRY_START_HOUR, GAP_ENTRY_START_MINUTE,
                    GAP_ENTRY_END_HOUR, GAP_ENTRY_END_MINUTE,
                    OPEN_CLOUD_BREAK_BODY_PCT, OPEN_CLOUD_BREAK_RANGE_PCT,
                    OPEN_CLOUD_BREAK_VOL_MULT,
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
        # IBKR returns tz-aware timestamps; datetime.now() is naive — align them.
        bt = pd.Timestamp(bar_time)
        if df_10m.index.tz is not None and bt.tzinfo is None:
            bt = bt.tz_localize(df_10m.index.tz)
        elif df_10m.index.tz is None and bt.tzinfo is not None:
            bt = bt.tz_localize(None)
        today_bars = df_10m[
            (df_10m.index.date == target_date) & (df_10m.index <= bt)
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
    No dollar cap — actual risk is gated downstream by MAX_RISK_DOLLARS /
    MAX_RISK_DOLLARS_HIGH per-trade caps.  A $15 dollar cap here was silently
    blocking every MU/META/SNDK trade because their 2.5% range exceeds $15.
    """
    dist = (entry_price - stop_price if direction == "long"
            else stop_price - entry_price)
    if dist <= 0:
        return False
    pct = dist / entry_price
    return (pct >= MIN_STOP_PCT_LOWER and      # floor: tighter than 0.25% = noise
            pct <= MAX_STOP_PCT)               # ceiling: wider than 2.5% = too much risk


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
        new_stop = max(current_stop, trail_to)      # never move stop down
        # Symmetric guard with the short path below:
        # On cloud_cont_crash longs (rip-and-go entries), ema5 << ema50,
        # so trail_to = ema50 sits ABOVE the initial stop.  Without the
        # guard the next 1-min bar tightens the stop to ema50 and any
        # normal wick stops the trade out.  Keep the initial stop intact
        # until the ratchet has been triggered.
        if best_unrealised < RATCHET_START and new_stop > current_stop:
            return current_stop
        return new_stop

    else:  # short
        if best_unrealised >= RATCHET_START:
            floor = entry_price - max(0.0, best_unrealised - RATCHET_GIVEBACK)
            trail_to = min(trail_to, floor)
        new_stop = min(current_stop, trail_to)
        # Guard: crash entries (cloud_cont_crash) have ema50 well below entry_price.
        # Without this guard, trail_to = ema50 << entry immediately drops the stop
        # into "profit territory" and triggers the short stop on the very next bar.
        # Keep the existing stop unchanged until the ratchet has activated.
        if best_unrealised < RATCHET_START and new_stop < entry_price:
            return current_stop
        return new_stop


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
# Opening-drive gap model
# ------------------------------------------------------------------ #

def _in_gap_entry_window(bar_time) -> bool:
    """True for completed bars from 09:33 through 10:00 ET inclusive."""
    if bar_time is None:
        return False
    minutes = bar_time.hour * 60 + bar_time.minute
    start = GAP_ENTRY_START_HOUR * 60 + GAP_ENTRY_START_MINUTE
    end = GAP_ENTRY_END_HOUR * 60 + GAP_ENTRY_END_MINUTE
    return start <= minutes <= end


def get_gap_signal_3m(df_3m: pd.DataFrame,
                      bar_time=None,
                      pmh: float = None,
                      support: float = None,
                      resistance: float = None) -> tuple[str, float, str]:
    """
    Opening-drive playbook checked before normal cloud/curl entries.

    Gap & Go:
      LONG when the current 3-min close crosses above max(PMH, resistance).
      Initial stop is PMH.

    Gap & Crap:
      SHORT when the current 3-min close crosses below support.
      Initial stop is support.
    """
    if not _in_gap_entry_window(bar_time) or len(df_3m) < 2:
        return "none", 0.0, ""

    cur = df_3m.iloc[-1]
    prev = df_3m.iloc[-2]
    entry_price = cur.close

    if pmh is not None:
        trigger = max(pmh, resistance) if resistance is not None else pmh
        if prev.close <= trigger and entry_price > trigger and pmh < entry_price:
            return "long", pmh, "gap_go_pmh"

    if support is not None:
        if prev.close >= support and entry_price < support:
            return "short", support, "gap_crap_support"

    return "none", 0.0, ""


def get_open_cloud_break_signal_3m(df_3m: pd.DataFrame,
                                   bar_time=None) -> tuple[str, float, str]:
    """
    Opening-drive cloud break playbook.

    This is separate from Rip support/resistance. It catches stocks that gap in
    premarket and then slice through the active 5/12 cloud on the opening drive.
    The first RTH bar is evaluated by its own open/close, not yesterday's close.

    SHORT:
      - 09:33-10:00 window
      - red wide-range candle
      - opens above/inside the fast cloud
      - closes below the fast cloud
      - volume expansion
      - stop = candle high

    LONG is the mirror image with stop = candle low.
    """
    if not _in_gap_entry_window(bar_time) or len(df_3m) < 2:
        return "none", 0.0, ""

    cur = df_3m.iloc[-1]
    entry_price = cur.close
    if entry_price <= 0:
        return "none", 0.0, ""

    fast_top = max(cur.ema5, cur.ema12)
    fast_bot = min(cur.ema5, cur.ema12)
    body = abs(cur.close - cur.open)
    full_range = cur.high - cur.low
    body_pct = body / entry_price
    range_pct = full_range / entry_price
    vol_ok = cur.vol_ma20 <= 0 or cur.volume >= OPEN_CLOUD_BREAK_VOL_MULT * cur.vol_ma20
    wide_enough = (body_pct >= OPEN_CLOUD_BREAK_BODY_PCT and
                   range_pct >= OPEN_CLOUD_BREAK_RANGE_PCT)

    if not vol_ok or not wide_enough:
        return "none", 0.0, ""

    red_slice = (cur.close < cur.open and
                 cur.open >= fast_bot and
                 cur.close < fast_bot)
    if red_slice and _stop_ok(entry_price, cur.high, "short"):
        return "short", cur.high, "open_cloud_break_short"

    green_slice = (cur.close > cur.open and
                   cur.open <= fast_top and
                   cur.close > fast_top)
    if green_slice and _stop_ok(entry_price, cur.low, "long"):
        return "long", cur.low, "open_cloud_break_long"

    return "none", 0.0, ""


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
    # Cloud 2 flip: ema5 crosses ema12
    # Grace window: also fire on the 1-2 bars immediately after a flip while
    # C2 is still aligned — catches cases where the flip bar had low volume
    # but volume confirmed on the next bar.
    c2_flip_long  = prev.ema5 <= prev.ema12 and cur.ema5 > cur.ema12
    c2_flip_short = prev.ema5 >= prev.ema12 and cur.ema5 < cur.ema12

    flip_late = False   # True when entry fires on bar +1 or +2 after the flip
    if not c2_flip_long and cur.ema5 > cur.ema12 and len(df_3m) >= 4:
        p2 = df_3m.iloc[-3]
        p3 = df_3m.iloc[-4]
        # 1 bar ago was the flip (prev=green, p2=red)
        if prev.ema5 > prev.ema12 and p2.ema5 <= p2.ema12:
            c2_flip_long = True
            flip_late = True
        # 2 bars ago was the flip (p2=green, p3=red)
        elif prev.ema5 > prev.ema12 and p2.ema5 > p2.ema12 and p3.ema5 <= p3.ema12:
            c2_flip_long = True
            flip_late = True

    if not c2_flip_short and cur.ema5 < cur.ema12 and len(df_3m) >= 4:
        p2 = df_3m.iloc[-3]
        p3 = df_3m.iloc[-4]
        if prev.ema5 < prev.ema12 and p2.ema5 >= p2.ema12:
            c2_flip_short = True
            flip_late = True
        elif prev.ema5 < prev.ema12 and p2.ema5 < p2.ema12 and p3.ema5 >= p3.ema12:
            c2_flip_short = True
            flip_late = True

    # =================================================================
    # CORE DAY-TRADING TRIGGER: 5/12 cloud flip.
    # No C3 (34/50) filter.  No 10m trend filter.  No volume filter.
    # User decision (2026-05-28): if the lower cloud flips, take the trade.
    # The 34/50 cloud and 10m trend are LOGGED as context only — never block.
    # Stop = ema12 with a safety fallback to the candle high/low so a flat
    # ema12 right at price still produces a usable stop distance.
    # =================================================================
    c3_green = cur.ema34 > cur.ema50    # informational only
    c3_red   = cur.ema34 < cur.ema50    # informational only

    if c2_flip_long:
        # Stop is the LOWER of ema12 vs candle low — whichever gives more room.
        # If ema12 happens to be above entry (rare on a fresh flip), candle low
        # carries the stop below entry so _stop_ok still passes.
        stop = min(cur.ema12, cur.low)
        reason = "cloud_512_flip+1" if flip_late else "cloud_512_flip"
        if _stop_ok(entry_price, stop, "long"):
            return "long", stop, reason

    if c2_flip_short:
        # Stop is the HIGHER of ema12 vs candle high.
        stop = max(cur.ema12, cur.high)
        reason = "cloud_512_flip+1" if flip_late else "cloud_512_flip"
        if _stop_ok(entry_price, stop, "short"):
            return "short", stop, reason

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

    # ---- Cloud continuation — already trending, no flip needed ─────────────
    # Two variants:
    #
    # Variant A  (EMA-crossover, steady trend)
    #   ema5 crossed ema12 some bars ago and both bars stay aligned.
    #   Works well for smooth trends with narrow-range bars.
    #   Requires CONFIRMED 10m trend (reduces pre-market noise).
    #   Stop = ema12 (near cloud edge).
    #
    # Variant B  (close-based, crash / gap-day)
    #   Ripster EMAs use hl2 = (high+low)/2.  Wide crash / rip bars distort
    #   ema5 — it stays ABOVE the close even while price collapses, so the
    #   ema5/ema12 crossover never fires (or fires 30+ min late).
    #   Instead, check whether the CLOSE price itself is below ema12.
    #   Because ema5 (hl2-based) > close on bear bars, it serves as a
    #   natural stop ABOVE the entry — satisfying _stop_ok for a short.
    #   Allows trend == "none" so it fires on crash-open days where the
    #   10-min cloud hasn't had time to confirm.
    #   Stop = ema5 (above close for shorts / below close for longs).

    # Variant A — EMA-crossover continuation
    c2_bear_cont = prev.ema5 < prev.ema12 and cur.ema5 < cur.ema12
    if c2_bear_cont and c3_red and trend == "bearish":
        stop     = cur.ema12
        dist_pct = (cur.ema12 - entry_price) / entry_price if entry_price > 0 else 1.0
        if dist_pct <= CLOUD_CONT_MAX_DIST and _stop_ok(entry_price, stop, "short"):
            return "short", stop, "cloud_cont"

    c2_bull_cont = prev.ema5 > prev.ema12 and cur.ema5 > cur.ema12
    if c2_bull_cont and c3_green and trend in ("bullish", "none"):
        stop     = cur.ema12
        dist_pct = (entry_price - cur.ema12) / entry_price if entry_price > 0 else 1.0
        if dist_pct <= CLOUD_CONT_MAX_DIST and _stop_ok(entry_price, stop, "long"):
            return "long", stop, "cloud_cont"

    # Variant B — close-based crash continuation (hl2-distortion specific)
    #
    # The hl2 distortion signature: wide crash/rip bars cause ema5 (hl2-based)
    # to stay ABOVE ema12 even as the actual close price collapses BELOW ema12.
    # This creates a paradox — the EMA crossover says "bullish" but close says
    # "bearish."  Variant B fires ONLY in this specific situation:
    #
    #   ema5 > ema12      ←  EMA crossover says bullish (hl2 distorted upward)
    #   close < ema12     ←  but actual close is below the fast cloud top
    #   ema5  > close     ←  ema5 is above close → valid stop ABOVE entry for short
    #   price_declining   ←  momentum still negative
    #
    # This precisely targets crash-open scenarios and CANNOT fire on normal
    # bearish continuations (those have ema5 < ema12 and are handled by Variant A).

    c2_ema_says_bull  = cur.ema5   > cur.ema12  # hl2 distortion signature
    c2_close_below    = cur.close  < cur.ema12  # but close is below cloud top
    c2_ema5_above_cls = cur.ema5   > cur.close  # ema5 above close → valid short stop
    price_declining   = cur.close  < prev.close  # momentum still negative

    if c2_ema_says_bull and c2_close_below and c2_ema5_above_cls and price_declining \
            and trend in ("bearish", "none"):
        stop     = cur.ema5
        dist_pct = (stop - entry_price) / entry_price if entry_price > 0 else 1.0
        if dist_pct <= CLOUD_CONT_MAX_DIST and _stop_ok(entry_price, stop, "short"):
            return "short", stop, "cloud_cont_crash"

    # Mirror for longs (gap-up / rip scenario):
    # ema5 < ema12 (hl2 distorted downward), but close > ema12.
    # Allow trend="none" here as an opening/day-trading reclaim. Waiting for
    # 10m bullish confirmation misses the first usable lower-cloud turn.
    c2_ema_says_bear  = cur.ema5   < cur.ema12
    c2_close_above    = cur.close  > cur.ema12
    c2_ema5_below_cls = cur.ema5   < cur.close
    price_rising      = cur.close  > prev.close

    if c2_ema_says_bear and c2_close_above and c2_ema5_below_cls and price_rising \
            and trend in ("bullish", "none"):
        stop     = cur.ema5
        dist_pct = (entry_price - stop) / entry_price if entry_price > 0 else 1.0
        if dist_pct <= CLOUD_CONT_MAX_DIST and _stop_ok(entry_price, stop, "long"):
            return "long", stop, "cloud_cont_crash"

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
