import pandas as pd
from config import (EMA_PERIODS, MIN_BARS_10M, MIN_BARS_3M,
                    MAX_STOP_DISTANCE, MAX_STOP_PCT,
                    BREAKEVEN_TRIGGER, VOLUME_CONFIRM_MULT,
                    CLOUD_EXIT_BUFFER, GAP_THRESHOLD)


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
# "Taking off" — price bounced off a cloud boundary and closed away
# ------------------------------------------------------------------ #

def _taking_off(row, direction: str) -> bool:
    """
    True when the candle touched a cloud boundary AND closed moving away from it.

    Levels checked (all are cloud edges, not noisy fast EMAs):
      ema9  — slow edge of Cloud 1 (micro-trend, 8/9)
      ema12 — slow edge of Cloud 2 (fast cloud,  5/12)
      ema34 — fast edge of Cloud 3 (slow cloud, 34/50) ← most important
      ema50 — slow edge of Cloud 3 (stop level)

    Long:  candle low <= level AND close > level  (bounced up through level)
    Short: candle high >= level AND close < level  (rejected down through level)
    """
    levels = [row.ema9, row.ema12, row.ema34, row.ema50]
    if direction == "long":
        return any(row.low <= lvl <= row.close for lvl in levels)
    return any(row.close <= lvl <= row.high for lvl in levels)


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

def get_entry_signal_3m(df_3m: pd.DataFrame, trend: str,
                        bar_time=None,
                        pmh: float = None, pml: float = None) -> tuple[str, float]:
    """
    Returns (signal, stop_price).  signal = 'long' | 'short' | 'none'

    TYPE 1 — Cloud flip (highest priority)
      Cloud 2 (5/12) AND Cloud 3 (34/50) both simultaneously flip to the
      same direction AND hl2 is on the correct side of ema200.
      Classic "all clouds go green/red at once" setup.
      For Day1 gap-up/gap-down stocks, allow even at trend=none (see TYPE 1b).

    TYPE 1b — Gap continuation (Day1 catalyst plays)
      Stock opens with ALL clouds already aligned (no flip needed) because it
      gapped on a news catalyst.  ema50 is >1.5% from price — the slow cloud
      hasn't caught up.  Entry fires on the first bar(s) above all clouds with
      volume.  Stop: ema12 (fast cloud slow edge, much tighter than ema50).
      Requires yesterday's 10-min bars to already confirm the direction.

    TYPE 2 — Taking off from a cloud boundary in an established trend
      Price touches ema9/ema12/ema34/ema50 and closes away in the trend
      direction. Fast cloud (5/12) must confirm.

    TYPE 3 — 34/50 cloud reversal: C2 flips ON the reversal candle (fires before TYPE 2)
      Price dips into the slow cloud (low <= ema34) and closes BACK above it on
      the same bar, AND the fast cloud (5/12) flips bullish on that exact candle
      (prev ema5 < ema12, cur ema5 > ema12).  No prior 10-min trend required —
      the cloud touch + simultaneous C2 flip IS the Ripster reversal signal.
      Symmetric for shorts: high >= ema34, close < ema34/ema50, C2 flips bear.
      Volume must be above average to confirm genuine participation.
      Stop = ema50 (far edge of slow cloud).

    TYPE 4 — Pre-market high / low breakout or continuation  (pmh / pml required)
      Two sub-cases, both checked after the opening bar:

      4a — Crossing: first regular-session close that moves ABOVE pmh (or BELOW pml).
        prev.close was on the near side, cur.close is on the far side.
        Classic intraday "reclaim of the pre-market level" entry.
        Stop: ema12 — fast cloud slow edge, close to the crossing level.

      4b — Already above/below by gap: price is already >GAP_THRESHOLD (2%) past the
        pre-market level AND ema12 has also cleared the level (whole fast cloud is
        on the correct side).  Catches stocks that gap hard through PMH/PML in
        pre-market and never look back — enter after the opening bar settles.
        Stop: max(ema12, 2% below entry) — when ema12 hasn't yet caught up after a
        large pre-market gap the 2% momentum floor prevents a degenerate $10+ wide
        stop.  Once ema12 rises within 2% of entry it takes over as the mechanical
        stop automatically.

      Fast cloud must confirm direction.  Volume must be above average.
    """
    if len(df_3m) < MIN_BARS_3M:
        return "none", 0.0

    cur  = df_3m.iloc[-1]
    prev = df_3m.iloc[-2]

    entry_price = cur.close

    # Opening-bar gate — no entries at all before 09:40 ET.
    # The first 1-2 candles capture the gap open and are extremely noisy:
    # EMAs lag, clouds are distorted, and nearly any signal type fires spuriously.
    # Rip's own rule: "wait for the open to settle" = let the 09:30-09:39 bars
    # establish a range before committing.  All entry types are blocked here;
    # the first tradeable bar is 09:40.
    is_opening_bar = (bar_time is not None and
                      bar_time.hour == 9 and bar_time.minute < 40)
    if is_opening_bar:
        return "none", 0.0

    # --- Volume gate ---
    vol_above_avg = (cur.vol_ma20 <= 0 or
                     cur.volume >= VOLUME_CONFIRM_MULT * cur.vol_ma20)

    # --- TYPE 1: simultaneous Cloud 2 + Cloud 3 flip ---
    cur_all_bull  = cur.ema5  > cur.ema12 and cur.ema34 > cur.ema50
    prev_all_bull = prev.ema5 > prev.ema12 and prev.ema34 > prev.ema50
    cur_all_bear  = cur.ema5  < cur.ema12 and cur.ema34 < cur.ema50
    prev_all_bear = prev.ema5 < prev.ema12 and prev.ema34 < prev.ema50

    # Gap-up / gap-down detection — Day1 catalyst stocks.
    # ema50 > GAP_THRESHOLD (2.0%) away from price: slow cloud hasn't caught up.
    # Use ema12 (fast cloud slow edge) as the tighter stop instead.
    # These stocks gap on news so the catalyst IS the trend direction signal.
    is_gap_up   = (cur_all_bull and
                   (entry_price - cur.ema50) / entry_price > GAP_THRESHOLD)
    is_gap_down = (cur_all_bear and
                   (cur.ema50 - entry_price) / entry_price > GAP_THRESHOLD)

    # TYPE 1 requires established 10-min trend OR a gap-up/down condition.
    # Non-gap flips at trend=none = morning chop trap (blocked).
    t1_trend_ok_long  = (trend == "bullish") or is_gap_up
    t1_trend_ok_short = (trend == "bearish") or is_gap_down

    if cur_all_bull and not prev_all_bull and cur.hl2 > cur.ema200 and vol_above_avg and t1_trend_ok_long:
        stop = cur.ema12 if is_gap_up else compute_stop(df_3m, "long", entry_price)
        if _stop_ok(entry_price, stop, "long"):
            return "long", stop

    if cur_all_bear and not prev_all_bear and cur.hl2 < cur.ema200 and vol_above_avg and t1_trend_ok_short:
        stop = cur.ema12 if is_gap_down else compute_stop(df_3m, "short", entry_price)
        if _stop_ok(entry_price, stop, "short"):
            return "short", stop

    # --- TYPE 1b: Gap continuation (both cur AND prev already all-aligned) ---
    # For stocks that GAP UP on a catalyst and OPEN above all clouds with no flip.
    # Yesterday's 10-min bars must confirm the direction (trend="bullish"/"bearish").
    # Stop: ema12 (fast cloud slow edge) — tighter than ema50 on a fresh gap.
    # Volume gate confirms genuine participation (opening gap bars usually qualify).
    if (cur_all_bull and prev_all_bull and cur.hl2 > cur.ema200 and
            vol_above_avg and is_gap_up and trend == "bullish"):
        stop = cur.ema12
        if _stop_ok(entry_price, stop, "long"):
            return "long", stop

    if (cur_all_bear and prev_all_bear and cur.hl2 < cur.ema200 and
            vol_above_avg and is_gap_down and trend == "bearish"):
        stop = cur.ema12
        if _stop_ok(entry_price, stop, "short"):
            return "short", stop

    # --- TYPE 3: 34/50 cloud reversal — C2 flips ON the reversal candle ---
    # Long:  low dips into slow cloud (low <= ema34), closes back above it,
    #        AND fast cloud flips bull on this exact bar (prev c2 bear → cur c2 bull).
    #        No prior 10-min trend needed; C2 flip + cloud touch = the reversal.
    # Short: high spikes into slow cloud (high >= ema34), closes back below it,
    #        AND fast cloud flips bear on this exact bar (prev c2 bull → cur c2 bear).
    c2_flip_up   = prev.ema5 < prev.ema12 and cur.ema5 > cur.ema12
    c2_flip_down = prev.ema5 > prev.ema12 and cur.ema5 < cur.ema12

    cloud_bounce_long  = (cur.low  <= cur.ema34 and
                          cur.close >  cur.ema34 and
                          cur.close >  cur.ema50 and
                          c2_flip_up             and
                          vol_above_avg)
    cloud_bounce_short = (cur.high >= cur.ema34 and
                          cur.close <  cur.ema34 and
                          cur.close <  cur.ema50 and
                          c2_flip_down           and
                          vol_above_avg)

    # TYPE 3 guards:
    #   Require a confirmed 10-min trend in the SAME direction.
    #   A 3-min reversal bounce with no 10-min backing (trend=none) is more
    #   likely a dead-cat bounce in choppy midday conditions than a real reversal.

    if cloud_bounce_long and trend == "bullish":
        stop = compute_stop(df_3m, "long", entry_price)
        if _stop_ok(entry_price, stop, "long"):
            return "long", stop

    if cloud_bounce_short and trend == "bearish":
        stop = compute_stop(df_3m, "short", entry_price)
        if _stop_ok(entry_price, stop, "short"):
            return "short", stop

    # --- TYPE 2: taking off from cloud level in an established trend ---
    # Candle must be GREEN for longs (close > open) and RED for shorts (close < open).
    # A bar that dips to the cloud but only half-recovers while still closing below its
    # open is NOT "taking off" — it's a weak bounce that often fails immediately.
    if trend == "bullish" and _taking_off(cur, "long") and cur.ema5 > cur.ema12 and cur.close > cur.open:
        stop = compute_stop(df_3m, "long", entry_price)
        if _stop_ok(entry_price, stop, "long"):
            return "long", stop

    if trend == "bearish" and _taking_off(cur, "short") and cur.ema5 < cur.ema12 and cur.close < cur.open:
        stop = compute_stop(df_3m, "short", entry_price)
        if _stop_ok(entry_price, stop, "short"):
            return "short", stop

    # --- TYPE 4: pre-market high / low breakout ---
    # Fires when the regular session first closes ABOVE the pre-market high
    # (long) or BELOW the pre-market low (short).
    # The previous bar must have closed on the other side of the level — this
    # ensures we catch only the crossing candle, not every bar above/below.
    # Fast cloud (5/12) and volume must confirm.
    if pmh is not None and pml is not None and pmh > pml:
        # 4a — crossing PMH/PML for the first time in the regular session
        pmh_break = (prev.close < pmh and cur.close > pmh and
                     cur.ema5 > cur.ema12 and vol_above_avg)
        pml_break = (prev.close > pml and cur.close < pml and
                     cur.ema5 < cur.ema12 and vol_above_avg)

        # 4b — already above PMH / below PML by a meaningful gap.
        # ema12 must also be through the level (whole fast cloud cleared it).
        # Only fires if price is >GAP_THRESHOLD (2%) past the pre-market level
        # to avoid triggering on stocks that are just grazing the boundary.
        # 4b — only for LONGS (above PMH).
        # Gap-up stocks that hold above PMH tend to continue higher (DELL-style).
        # Gap-DOWN stocks (below PML) almost always see a morning squeeze that
        # exceeds a 2% stop before any resumption — TYPE 4b short is 0W in all
        # tested data and structurally unreliable.  Use TYPE 4a for PML breaks.
        above_pmh = (cur.close  > pmh * (1 + GAP_THRESHOLD) and
                     cur.ema12  > pmh and
                     cur.ema5   > cur.ema12 and
                     cur.close  > prev.close and   # momentum: bar still rising
                     vol_above_avg)

        if pmh_break or above_pmh:
            if above_pmh and not pmh_break:
                # 4b: use 2% momentum floor when ema12 hasn't caught up yet
                stop = max(cur.ema12, entry_price * (1 - MAX_STOP_PCT * 0.8))
            else:
                stop = cur.ema12   # 4a crossing: ema12 is near the level
            if _stop_ok(entry_price, stop, "long"):
                return "long", stop

        if pml_break:
            # TYPE 4a only for shorts — first crossing below PML
            stop = cur.ema12
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
