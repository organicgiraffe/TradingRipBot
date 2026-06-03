"""
velez_experiment.py - standalone Oliver Velez style backtest.
Test-only. NOT wired into the live bot.

Rules mechanized from the researched Velez "color game" framework:
  1. Location / picture of power:
       long  = price above rising 20 EMA, 20 EMA above 200 EMA
       short = price below falling 20 EMA, 20 EMA below 200 EMA
       optional 200-bounce mode also allows price in the 20/200 zone
       while the 20 is turning up/down from the 200.
  2. Trade with the 20 EMA:
       rising 20 = longs only; falling 20 = shorts only.
  3. Respect the 200 EMA:
       above 200 = buy side only; below 200 = short side only.
  4. Entry trigger must happen at/near the 20 EMA:
       long  = green takes out red: place buy stop 1c above red bar high
       short = red takes out green: place sell stop 1c below green bar low
       stop  = 1c beyond the other side of the reason bar.

Realism assumptions:
  - No same-bar hindsight: orders are created only after the reason bar closes.
  - Entry stop orders are live for the next bar only.
  - If entry and stop both occur in the trigger bar, assume the stop hit.
  - Optional profit management can protect trades that move away from the 20 EMA.
  - Fills include adverse slippage and IBKR-style minimum commissions.
  - The main report uses live-style global slot and per-symbol trade limits.
"""

import warnings

import pandas as pd
import yfinance as yf

from config import (
    FIXED_SHARES,
    FIXED_SHARES_HIGH,
    HIGH_PRICE_THRESHOLD,
    MAX_RISK_DOLLARS,
    MAX_RISK_DOLLARS_HIGH,
    MAX_SIMULTANEOUS_POSITIONS,
    MAX_TRADES_PER_DAY,
)

warnings.filterwarnings("ignore")


STOCKS = ["LLY", "AVGO", "ARM", "NET", "NVDA", "CRM", "ZS", "CRWV"]
LOOKBACK_DAYS = 5
DOWNLOAD_PERIOD = "7d"
DOWNLOAD_INTERVAL = "1m"

BAR_SIZE = "10min"
ENTRY_MODE = "orb10"                  # "classic", "200_bounce", "orb10", or combos.
MANAGEMENT_MODE = "rip_cloud"         # "oliver_20" or "rip_cloud".
NEAR_20_PCT = 0.0030          # 0.30% zone around the 20 EMA.
NEAR_200_PCT = 0.0050         # Wider zone for 200 EMA bounce/reclaim setups.
EMA20_SLOPE_LOOKBACK = 3      # 3 bars.
EMA200_SLOPE_LOOKBACK = 10
REQUIRE_20_SLOPE_FOR_200_BOUNCE = False
ORB_MINUTES = 10
ORB_STOP_MODE = "third_from_break"  # "full_range", "midpoint", or "third_from_break".
ORB_ONE_SHOT_PER_SYMBOL = True
ALLOW_COARSE_ORB_SOURCE = False
ENTRY_PAD = 0.01              # Velez: one penny above/below reason bar.
STOP_PAD = 0.01
SLIPPAGE_PER_SHARE = 0.02     # Adverse fill model for stop/marketable orders.
COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.00
RATCHET_START_R = 1.0         # Start protecting once open profit reaches 1R.
RATCHET_LOCK_PCT = 0.50       # Lock 50% of best open profit after activation.
EXTENSION_R = 1.0             # Require price to be at least 1R from 20 EMA.
RATCHET_ENABLED = False       # Full-position ratchet was too tight in testing.
PARTIAL_ENABLED = False       # Available for experiments; baseline tested better.
PARTIAL_START_R = 2.0         # Take a partial once price reaches +2R.
PARTIAL_EXTENSION_R = 1.0     # Require the best price to be at least 1R from 20 EMA.
PARTIAL_FRACTION = 0.50       # Close half, let the rest work.
PARTIAL_MOVE_STOP_TO_BE = True


def _with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    out["hl2"] = (out["high"] + out["low"]) / 2
    for period in (5, 12, 34, 50, 200):
        out[f"rip_ema{period}"] = out["hl2"].ewm(span=period, adjust=False).mean()
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    return out


def _resample_with_indicators(raw: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = (
        raw.resample(freq, label="right", closed="right")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Close"])
    )
    return _with_indicators(agg)


def _is_one_minute_frame(df: pd.DataFrame | None) -> bool:
    if df is None or df.empty or len(df.index) < 2:
        return False
    diffs = pd.Series(df.index).diff().dropna()
    diffs = diffs[(diffs > pd.Timedelta(0)) & (diffs <= pd.Timedelta("5min"))]
    if diffs.empty:
        return False
    return diffs.median() <= pd.Timedelta("75s")


def _bars_frame(ds) -> pd.DataFrame:
    return ds["bars"] if isinstance(ds, dict) else ds


def _orb_source_frame(ds) -> pd.DataFrame | None:
    if isinstance(ds, dict):
        return ds.get("orb_1m")
    return ds if _is_one_minute_frame(ds) else None


def prep(sym: str):
    raw = yf.download(
        sym,
        period=DOWNLOAD_PERIOD,
        interval=DOWNLOAD_INTERVAL,
        progress=False,
        auto_adjust=True,
        prepost=True,
    )
    if raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        raw = raw.tz_convert("US/Eastern")

    raw = raw.between_time("04:00", "20:00")
    agg = _resample_with_indicators(raw, BAR_SIZE)
    orb_1m = raw.between_time("09:30", "16:00").rename(columns=str.lower)
    return {"bars": agg, "orb_1m": orb_1m}


def _day_frame(df2: pd.DataFrame, day: str) -> pd.DataFrame:
    df = _bars_frame(df2)
    return df[df.index.date.astype(str) == day].between_time("09:30", "16:00")


def _session_context_frame(df2: pd.DataFrame, day: str) -> pd.DataFrame:
    df = _bars_frame(df2)
    return df[df.index.date.astype(str) == day].between_time("04:00", "16:00")


def _orb_day_frame(df2, day: str) -> pd.DataFrame | None:
    df = _orb_source_frame(df2)
    if df is None:
        return None
    return df[df.index.date.astype(str) == day].between_time("09:30", "16:00")


def latest_common_days(data: dict[str, pd.DataFrame], lookback: int = LOOKBACK_DAYS) -> list[str]:
    common = None
    for ds in data.values():
        df = _bars_frame(ds)
        dates = set(str(d) for d in df.between_time("09:30", "16:00").index.date)
        common = dates if common is None else common & dates
    return sorted(common or [])[-lookback:]


def _near_ma(bar, ma: float, pct: float = NEAR_20_PCT) -> bool:
    if ma <= 0 or pd.isna(ma):
        return False
    zone_low = ma * (1 - pct)
    zone_high = ma * (1 + pct)
    return bar.low <= zone_high and bar.high >= zone_low


def _touches_20_200_zone(bar, pct: float = NEAR_200_PCT) -> bool:
    lower = min(float(bar.ema20), float(bar.ema200)) * (1 - pct)
    upper = max(float(bar.ema20), float(bar.ema200)) * (1 + pct)
    return float(bar.low) <= upper and float(bar.high) >= lower


def _entry_mode_allows(mode: str) -> bool:
    return mode in ENTRY_MODE.split("_or_")


def _orb_window_end(t) -> pd.Timestamp:
    start = pd.Timestamp(t).replace(hour=9, minute=30, second=0, microsecond=0)
    return start + pd.Timedelta(minutes=ORB_MINUTES)


def _orb_order(frame: pd.DataFrame, t, allow_short: bool = True,
               orb_frame: pd.DataFrame | None = None) -> dict | None:
    if not _entry_mode_allows("orb10") or frame.empty:
        return None

    end = _orb_window_end(t)
    if pd.Timestamp(t) < end:
        return None

    source = orb_frame if orb_frame is not None else frame
    if not _is_one_minute_frame(source) and not ALLOW_COARSE_ORB_SOURCE:
        raise ValueError(
            "ORB-10 requires exact 1-minute source data for the opening range. "
            "The supplied experiment data is coarse, so the old 60-day Yahoo "
            "5-minute ORB numbers are approximate and blocked by default. "
            "Set velez_experiment.ALLOW_COARSE_ORB_SOURCE=True only for a "
            "clearly labelled approximation."
        )

    start = end - pd.Timedelta(minutes=ORB_MINUTES)
    if _is_one_minute_frame(source):
        opening_range = source[(source.index >= start) & (source.index < end)]
    else:
        opening_range = source[(source.index > start) & (source.index <= end)]
    if opening_range.empty:
        return None

    orb_high = round(float(opening_range["high"].max()), 2)
    orb_low = round(float(opening_range["low"].min()), 2)
    if orb_high <= orb_low:
        return None

    orb_mid = round((orb_high + orb_low) / 2, 2)
    orb_range = orb_high - orb_low
    if ORB_STOP_MODE == "full_range":
        long_stop = round(orb_low - STOP_PAD, 2)
        short_stop = round(orb_high + STOP_PAD, 2)
    elif ORB_STOP_MODE == "midpoint":
        long_stop = orb_mid
        short_stop = orb_mid
    elif ORB_STOP_MODE == "third_from_break":
        long_stop = round(orb_high - orb_range / 3, 2)
        short_stop = round(orb_low + orb_range / 3, 2)
    else:
        raise ValueError(f"Unknown ORB_STOP_MODE: {ORB_STOP_MODE}")

    return {
        "dir": "orb10",
        "long_trigger": round(orb_high + ENTRY_PAD, 2),
        "short_trigger": round(orb_low - ENTRY_PAD, 2) if allow_short else None,
        "long_stop": long_stop,
        "short_stop": short_stop if allow_short else None,
        "setup_time": end,
        "signal": "orb10",
    }


def _setup_locations(rows: list, i: int) -> dict[str, bool]:
    reason = rows[i]
    ema20_ref = rows[i - EMA20_SLOPE_LOOKBACK]
    ema200_ref = rows[max(0, i - EMA200_SLOPE_LOOKBACK)]

    ema20_rising = reason.ema20 > ema20_ref.ema20
    ema20_falling = reason.ema20 < ema20_ref.ema20
    ema200_flat_or_rising = reason.ema200 >= ema200_ref.ema200 * (1 - 0.0005)
    ema200_flat_or_falling = reason.ema200 <= ema200_ref.ema200 * (1 + 0.0005)

    classic_long = reason.close > reason.ema20 > reason.ema200 and ema20_rising
    classic_short = reason.close < reason.ema20 < reason.ema200 and ema20_falling

    above_or_reclaiming_200 = (
        reason.close > reason.ema200
        or (reason.low <= reason.ema200 and reason.close >= reason.ema200 * (1 - NEAR_200_PCT))
    )
    below_or_rejecting_200 = (
        reason.close < reason.ema200
        or (reason.high >= reason.ema200 and reason.close <= reason.ema200 * (1 + NEAR_200_PCT))
    )
    long_20_200_zone = _touches_20_200_zone(reason) or _near_ma(reason, reason.ema200, NEAR_200_PCT)
    short_20_200_zone = _touches_20_200_zone(reason) or _near_ma(reason, reason.ema200, NEAR_200_PCT)

    bounce_long = (
        (ema20_rising or not REQUIRE_20_SLOPE_FOR_200_BOUNCE)
        and ema200_flat_or_rising
        and above_or_reclaiming_200
        and long_20_200_zone
    )
    bounce_short = (
        (ema20_falling or not REQUIRE_20_SLOPE_FOR_200_BOUNCE)
        and ema200_flat_or_falling
        and below_or_rejecting_200
        and short_20_200_zone
    )

    return {
        "classic_long": classic_long,
        "classic_short": classic_short,
        "bounce_long": bounce_long,
        "bounce_short": bounce_short,
    }


def _position_size(entry: float) -> tuple[int, float]:
    shares = FIXED_SHARES_HIGH if entry >= HIGH_PRICE_THRESHOLD else FIXED_SHARES
    risk_cap = MAX_RISK_DOLLARS_HIGH if entry >= HIGH_PRICE_THRESHOLD else MAX_RISK_DOLLARS
    return shares, risk_cap


def _commission(shares: int) -> float:
    return max(MIN_COMMISSION, COMMISSION_PER_SHARE * shares)


def _slipped(price: float, direction: str, side: str) -> float:
    """Return adverse fill price for a buy/sell side."""
    if side == "entry":
        return price + SLIPPAGE_PER_SHARE if direction == "long" else price - SLIPPAGE_PER_SHARE
    return price - SLIPPAGE_PER_SHARE if direction == "long" else price + SLIPPAGE_PER_SHARE


def _entry_order(rows: list, i: int, allow_short: bool = True) -> dict | None:
    """Create the next-bar stop order from a closed reason bar."""
    if i < max(EMA20_SLOPE_LOOKBACK, EMA200_SLOPE_LOOKBACK):
        return None

    reason = rows[i]

    reason_red = reason.close < reason.open
    reason_green = reason.close > reason.open
    locations = _setup_locations(rows, i)

    reason_near_20 = _near_ma(reason, reason.ema20)
    reason_near_20_200 = reason_near_20 or _near_ma(reason, reason.ema200, NEAR_200_PCT)
    classic_long = _entry_mode_allows("classic") and locations["classic_long"] and reason_near_20
    classic_short = _entry_mode_allows("classic") and locations["classic_short"] and reason_near_20
    bounce_long = _entry_mode_allows("200_bounce") and locations["bounce_long"] and reason_near_20_200
    bounce_short = _entry_mode_allows("200_bounce") and locations["bounce_short"] and reason_near_20_200

    if reason_red and (classic_long or bounce_long):
        entry = round(reason.high + ENTRY_PAD, 2)
        stop = round(reason.low - STOP_PAD, 2)
        fill_entry = _slipped(entry, "long", "entry")
        if stop < fill_entry:
            shares, risk_cap = _position_size(fill_entry)
            if (fill_entry - stop) * shares <= risk_cap:
                return {
                    "dir": "long",
                    "trigger": entry,
                    "stop": stop,
                    "sh": shares,
                    "setup_time": reason.Index,
                    "signal": "green_takes_red_200_bounce" if bounce_long and not classic_long else "green_takes_red",
                }

    if allow_short and reason_green and (classic_short or bounce_short):
        entry = round(reason.low - ENTRY_PAD, 2)
        stop = round(reason.high + STOP_PAD, 2)
        fill_entry = _slipped(entry, "short", "entry")
        if stop > fill_entry:
            shares, risk_cap = _position_size(fill_entry)
            if (stop - fill_entry) * shares <= risk_cap:
                return {
                    "dir": "short",
                    "trigger": entry,
                    "stop": stop,
                    "sh": shares,
                    "setup_time": reason.Index,
                    "signal": "red_takes_green_200_bounce" if bounce_short and not classic_short else "red_takes_green",
                }

    return None


def _stop_exit_signal(pos: dict, bar) -> tuple[float, str] | None:
    if pos["dir"] == "long":
        if bar.low <= pos["stop"]:
            return _slipped(pos["stop"], "long", "exit"), "stop"
    else:
        if bar.high >= pos["stop"]:
            return _slipped(pos["stop"], "short", "exit"), "stop"
    return None


def _lost_20_exit_signal(pos: dict, bar) -> tuple[float, str] | None:
    if pos["dir"] == "long":
        if pos.get("needs_20_reclaim"):
            if bar.close >= bar.ema20:
                pos["needs_20_reclaim"] = False
            elif bar.close < bar.ema200:
                return _slipped(float(bar.close), "long", "exit"), "lost_200"
            return None
        if bar.close < bar.ema20:
            return _slipped(float(bar.close), "long", "exit"), "lost_20"
    else:
        if pos.get("needs_20_reclaim"):
            if bar.close <= bar.ema20:
                pos["needs_20_reclaim"] = False
            elif bar.close > bar.ema200:
                return _slipped(float(bar.close), "short", "exit"), "lost_200"
            return None
        if bar.close > bar.ema20:
            return _slipped(float(bar.close), "short", "exit"), "lost_20"
    return None


def _exit_signal(pos: dict, bar) -> tuple[float, str] | None:
    if MANAGEMENT_MODE == "rip_cloud":
        return _stop_exit_signal(pos, bar)
    if RATCHET_ENABLED:
        _apply_profit_ratchet(pos, bar)
    return _stop_exit_signal(pos, bar) or _lost_20_exit_signal(pos, bar)


def _apply_profit_ratchet(pos: dict, bar) -> None:
    risk = pos.get("risk", 0.0)
    if risk <= 0:
        return

    if pos["dir"] == "long":
        best_profit = float(bar.high) - pos["entry"]
        extended = float(bar.high) - float(bar.ema20)
        if best_profit < RATCHET_START_R * risk or extended < EXTENSION_R * risk:
            return
        lock_profit = max(0.0, best_profit * RATCHET_LOCK_PCT)
        new_stop = round(pos["entry"] + lock_profit, 2)
        pos["stop"] = max(pos["stop"], new_stop)
    else:
        best_profit = pos["entry"] - float(bar.low)
        extended = float(bar.ema20) - float(bar.low)
        if best_profit < RATCHET_START_R * risk or extended < EXTENSION_R * risk:
            return
        lock_profit = max(0.0, best_profit * RATCHET_LOCK_PCT)
        new_stop = round(pos["entry"] - lock_profit, 2)
        pos["stop"] = min(pos["stop"], new_stop)


def _close_trade(pos: dict, symbol: str, exit_price: float, exit_time, reason: str) -> dict:
    shares = pos["sh"]
    pnl = (
        (exit_price - pos["entry"]) * shares
        if pos["dir"] == "long"
        else (pos["entry"] - exit_price) * shares
    )
    base_shares = pos.get("sh_initial", shares)
    entry_commission_total = pos.get(
        "entry_commission_total",
        pos.get("entry_commission", 0.0),
    )
    entry_commission = entry_commission_total * (shares / base_shares)
    pnl -= entry_commission + _commission(shares)
    return {
        **pos,
        "symbol": symbol,
        "exit": float(exit_price),
        "exit_time": exit_time,
        "exit_reason": reason,
        "pnl": pnl,
    }


def _maybe_take_partial(pos: dict, symbol: str, bar, exit_time) -> dict | None:
    if not PARTIAL_ENABLED or pos.get("partial_taken") or pos["sh"] <= 1:
        return None

    risk = pos.get("risk", 0.0)
    if risk <= 0:
        return None

    if pos["dir"] == "long":
        target = pos["entry"] + PARTIAL_START_R * risk
        best_profit = float(bar.high) - pos["entry"]
        extension = float(bar.high) - float(bar.ema20)
        if best_profit < PARTIAL_START_R * risk or extension < PARTIAL_EXTENSION_R * risk:
            return None
        exit_price = _slipped(round(target, 2), "long", "exit")
        if PARTIAL_MOVE_STOP_TO_BE:
            pos["stop"] = max(pos["stop"], round(pos["entry"], 2))
    else:
        target = pos["entry"] - PARTIAL_START_R * risk
        best_profit = pos["entry"] - float(bar.low)
        extension = float(bar.ema20) - float(bar.low)
        if best_profit < PARTIAL_START_R * risk or extension < PARTIAL_EXTENSION_R * risk:
            return None
        exit_price = _slipped(round(target, 2), "short", "exit")
        if PARTIAL_MOVE_STOP_TO_BE:
            pos["stop"] = min(pos["stop"], round(pos["entry"], 2))

    partial_shares = int(pos.get("sh_initial", pos["sh"]) * PARTIAL_FRACTION)
    partial_shares = max(1, min(pos["sh"] - 1, partial_shares))
    partial_pos = {**pos, "sh": partial_shares}
    trade = _close_trade(
        partial_pos,
        symbol,
        exit_price,
        exit_time,
        f"partial_{PARTIAL_START_R:.1f}R",
    )
    pos["sh"] -= partial_shares
    pos["partial_taken"] = True
    return trade


def _rip_fast_state(bar) -> str:
    if pd.isna(getattr(bar, "rip_ema5", None)) or pd.isna(getattr(bar, "rip_ema12", None)):
        return "none"
    if bar.rip_ema5 > bar.rip_ema12:
        return "bull"
    if bar.rip_ema5 < bar.rip_ema12:
        return "bear"
    return "none"


def _apply_rip_cloud_trail(pos: dict, bar) -> None:
    trail_to = getattr(bar, "rip_ema50", None)
    if trail_to is None or pd.isna(trail_to):
        return

    if pos["dir"] == "long":
        if trail_to < bar.close:
            pos["stop"] = max(pos["stop"], round(float(trail_to), 2))
    else:
        if trail_to > bar.close:
            pos["stop"] = min(pos["stop"], round(float(trail_to), 2))


def _rip_cloud_exit_signal(pos: dict, bar) -> tuple[float, str] | None:
    state = _rip_fast_state(bar)
    if state == "none":
        return None

    if pos["dir"] == "long":
        if state == "bull":
            pos["rip_fast_confirmed"] = True
        elif pos.get("rip_fast_confirmed") and pos.get("last_rip_fast_state") == "bull":
            return _slipped(float(bar.close), "long", "exit"), "rip_5_12_exit"
    else:
        if state == "bear":
            pos["rip_fast_confirmed"] = True
        elif pos.get("rip_fast_confirmed") and pos.get("last_rip_fast_state") == "bear":
            return _slipped(float(bar.close), "short", "exit"), "rip_5_12_exit"

    pos["last_rip_fast_state"] = state
    return None


def _manage_position_rip(pos: dict, symbol: str, bar, exit_time) -> tuple[list[dict], dict | None]:
    stop_exit = _stop_exit_signal(pos, bar)
    if stop_exit is not None:
        px, reason = stop_exit
        return [_close_trade(pos, symbol, px, exit_time, reason)], None

    _apply_rip_cloud_trail(pos, bar)

    partial = _maybe_take_partial(pos, symbol, bar, exit_time)
    trades = [partial] if partial is not None else []

    cloud_exit = _rip_cloud_exit_signal(pos, bar)
    if cloud_exit is not None:
        px, reason = cloud_exit
        trades.append(_close_trade(pos, symbol, px, exit_time, reason))
        return trades, None

    return trades, pos


def _manage_position(pos: dict, symbol: str, bar, exit_time) -> tuple[list[dict], dict | None]:
    if MANAGEMENT_MODE == "rip_cloud":
        return _manage_position_rip(pos, symbol, bar, exit_time)

    if RATCHET_ENABLED:
        _apply_profit_ratchet(pos, bar)

    stop_exit = _stop_exit_signal(pos, bar)
    if stop_exit is not None:
        px, reason = stop_exit
        return [_close_trade(pos, symbol, px, exit_time, reason)], None

    partial = _maybe_take_partial(pos, symbol, bar, exit_time)
    trades = [partial] if partial is not None else []

    lost_20 = _lost_20_exit_signal(pos, bar)
    if lost_20 is not None:
        px, reason = lost_20
        trades.append(_close_trade(pos, symbol, px, exit_time, reason))
        return trades, None

    return trades, pos


def _trigger_order(order: dict, bar, bar_time=None) -> dict | None:
    if order["dir"] == "orb10":
        hit_long = bar.high >= order["long_trigger"]
        hit_short = order.get("short_trigger") is not None and bar.low <= order["short_trigger"]
        if hit_long and hit_short:
            return None
        if not hit_long and not hit_short:
            return None

        direction = "long" if hit_long else "short"
        trigger = order[f"{direction}_trigger"]
        stop = order[f"{direction}_stop"]
        entry = _slipped(trigger, direction, "entry")
        if (direction == "long" and stop >= entry) or (direction == "short" and stop <= entry):
            return None
        shares, risk_cap = _position_size(entry)
        if abs(entry - stop) * shares > risk_cap:
            return None

        fast_state = _rip_fast_state(bar)
        return {
            "dir": direction,
            "entry": entry,
            "stop": stop,
            "initial_stop": stop,
            "risk": abs(entry - stop),
            "sh": shares,
            "sh_initial": shares,
            "entry_time": bar_time if bar_time is not None else bar.Index,
            "setup_time": order["setup_time"],
            "signal": "orb10_breakout" if direction == "long" else "orb10_breakdown",
            "needs_20_reclaim": False,
            "last_rip_fast_state": fast_state,
            "rip_fast_confirmed": (
                (direction == "long" and fast_state == "bull")
                or (direction == "short" and fast_state == "bear")
            ),
            "entry_commission": _commission(shares),
            "entry_commission_total": _commission(shares),
            "partial_taken": False,
        }

    if order["dir"] == "long":
        if bar.high < order["trigger"]:
            return None
        entry = _slipped(order["trigger"], "long", "entry")
    else:
        if bar.low > order["trigger"]:
            return None
        entry = _slipped(order["trigger"], "short", "entry")

    entry_time = bar_time if bar_time is not None else bar.Index
    is_200_bounce = "200_bounce" in order["signal"]
    fast_state = _rip_fast_state(bar)
    needs_20_reclaim = (
        is_200_bounce
        and (
            (order["dir"] == "long" and bar.close < bar.ema20)
            or (order["dir"] == "short" and bar.close > bar.ema20)
        )
    )
    return {
        "dir": order["dir"],
        "entry": entry,
        "stop": order["stop"],
        "initial_stop": order["stop"],
        "risk": abs(entry - order["stop"]),
        "sh": order["sh"],
        "sh_initial": order["sh"],
        "entry_time": entry_time,
        "setup_time": order["setup_time"],
        "signal": order["signal"],
        "needs_20_reclaim": needs_20_reclaim,
        "last_rip_fast_state": fast_state,
        "rip_fast_confirmed": (
            (order["dir"] == "long" and fast_state == "bull")
            or (order["dir"] == "short" and fast_state == "bear")
        ),
        "entry_commission": _commission(order["sh"]),
        "entry_commission_total": _commission(order["sh"]),
        "partial_taken": False,
    }


def velez_sim(
    df2: pd.DataFrame,
    day: str,
    allow_short: bool = True,
    max_trades: int = MAX_TRADES_PER_DAY,
) -> tuple[int, int, float]:
    """Single-symbol Velez replay. Kept for quick symbol-level experiments."""
    d = _day_frame(df2, day)
    ctx = _session_context_frame(df2, day)
    orb = _orb_day_frame(df2, day)
    rows = list(d.itertuples())
    ctx_rows = list(ctx.itertuples())
    if len(rows) < max(3, EMA20_SLOPE_LOOKBACK + 1):
        return 0, 0, 0.0

    trades = []
    pos = None
    pending = None
    entries = 0

    for i in range(max(2, EMA20_SLOPE_LOOKBACK), len(rows)):
        bar = rows[i]

        if pos is not None:
            closed, pos = _manage_position(pos, "", bar, bar.Index)
            trades.extend(closed)

        if pos is None and pending is not None and entries < max_trades:
            filled = _trigger_order(pending, bar)
            if filled is not None:
                pos = filled
                entries += 1
                same_bar_stop = _exit_signal(pos, bar)
                if same_bar_stop is not None:
                    px, reason = same_bar_stop
                    trades.append(_close_trade(pos, "", px, bar.Index, reason + "_same_bar"))
                    pos = None
            pending = None

        if pos is not None or entries >= max_trades:
            continue

        pending = None
        if not ORB_ONE_SHOT_PER_SYMBOL or entries == 0:
            pending = _orb_order(d, bar.Index, allow_short=allow_short,
                                 orb_frame=orb)
        if pending is None:
            ctx_i = ctx.index.get_loc(bar.Index) if bar.Index in ctx.index else None
            if isinstance(ctx_i, int):
                pending = _entry_order(ctx_rows, ctx_i, allow_short=allow_short)

    if pos is not None:
        eod_px = _slipped(float(rows[-1].close), pos["dir"], "exit")
        trades.append(_close_trade(pos, "", eod_px, rows[-1].Index, "eod"))

    wins = sum(1 for t in trades if t["pnl"] > 0)
    net = sum(t["pnl"] for t in trades)
    return len(trades), wins, net


def velez_multi(
    data: dict[str, pd.DataFrame],
    day: str,
    allow_short: bool = True,
    max_positions: int = MAX_SIMULTANEOUS_POSITIONS,
    max_trades_per_symbol: int = MAX_TRADES_PER_DAY,
) -> list[dict]:
    """Live-style replay: shared position slots across all symbols."""
    frames = {s: _day_frame(df, day) for s, df in data.items()}
    contexts = {s: _session_context_frame(df, day) for s, df in data.items()}
    orb_frames = {s: _orb_day_frame(df, day) for s, df in data.items()}
    times = sorted(set().union(*(set(df.index) for df in frames.values() if not df.empty)))
    positions: dict[str, dict] = {}
    pending_orders: dict[str, dict] = {}
    trades: list[dict] = []
    counts = {s: 0 for s in data}

    for t in times:
        # Existing positions exit first, matching the live bot's risk-first loop.
        for sym in list(data):
            if sym not in positions or t not in frames[sym].index:
                continue
            bar = frames[sym].loc[t]
            closed, open_pos = _manage_position(positions[sym], sym, bar, t)
            trades.extend(closed)
            if open_pos is None:
                del positions[sym]

        # Next-bar stop orders trigger after the setup bar has closed.
        for sym in list(data):
            if len(positions) >= max_positions:
                break
            if sym in positions or sym not in pending_orders:
                continue
            if counts[sym] >= max_trades_per_symbol or t not in frames[sym].index:
                pending_orders.pop(sym, None)
                continue

            bar = frames[sym].loc[t]
            order = pending_orders.pop(sym)
            filled = _trigger_order(order, bar, t)
            if filled is None:
                continue

            positions[sym] = filled
            counts[sym] += 1
            same_bar_stop = _exit_signal(filled, bar)
            if same_bar_stop is not None:
                px, reason = same_bar_stop
                trades.append(_close_trade(filled, sym, px, t, reason + "_same_bar"))
                del positions[sym]

        # At bar close, create only next-bar orders. They expire if not filled
        # on the following bar.
        pending_orders = {}
        for sym in data:
            if len(positions) + len(pending_orders) >= max_positions:
                break
            if sym in positions or counts[sym] >= max_trades_per_symbol:
                continue
            order = None
            if not ORB_ONE_SHOT_PER_SYMBOL or counts[sym] == 0:
                order = _orb_order(frames[sym], t, allow_short=allow_short,
                                   orb_frame=orb_frames.get(sym))
            if order is None:
                frame = contexts[sym]
                if t not in frame.index:
                    continue

                i = frame.index.get_loc(t)
                if not isinstance(i, int):
                    continue
                rows = list(frame.itertuples())
                order = _entry_order(rows, i, allow_short=allow_short)
            if order is not None:
                pending_orders[sym] = order

    for sym, pos in list(positions.items()):
        frame = frames[sym]
        if not frame.empty:
            eod_px = _slipped(float(frame.iloc[-1].close), pos["dir"], "exit")
            trades.append(_close_trade(pos, sym, eod_px, frame.index[-1], "eod"))

    return trades


def _print_summary(label: str, trades: list[dict]) -> None:
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    net = sum(t["pnl"] for t in trades)
    win_rate = 100 * len(wins) / len(trades) if trades else 0.0
    profit_factor = (
        sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses))
        if losses
        else float("inf")
    )
    print(
        f"  {label:<12} {len(trades):3d}t  {len(wins):2d}W"
        f"  ({win_rate:4.1f}%)  NET ${net:+.0f}  PF={profit_factor:.2f}"
    )


if __name__ == "__main__":
    print(f"Downloading + resampling to {BAR_SIZE} (once each)...")
    data = {}
    for s in STOCKS:
        d = prep(s)
        if d is not None:
            data[s] = d
    print(f"  {len(data)}/{len(STOCKS)} loaded\n")

    days = latest_common_days(data)

    print("=" * 72)
    print("  VELEZ COLOR GAME SIM - 20/200 EMA, near-20 entries, live slot cap")
    print("=" * 72)
    print(
        f"  max_pos={MAX_SIMULTANEOUS_POSITIONS}  "
        f"max_trades/symbol={MAX_TRADES_PER_DAY}  "
        f"near20={NEAR_20_PCT:.2%}  "
        f"slip=${SLIPPAGE_PER_SHARE:.2f}/sh  "
        f"commission=max(${MIN_COMMISSION:.0f}, {COMMISSION_PER_SHARE:.3f}/sh)"
    )
    print(f"  dates={', '.join(days)}")
    print()

    all_trades = []
    for day in days:
        trades = velez_multi(data, day)
        all_trades.extend(trades)
        _print_summary(day, trades)

    print("  " + "-" * 64)
    _print_summary("TOTAL", all_trades)
