"""
week_backtest.py — run the multi-symbol engine across the last 5 trading days.

Downloads data once per symbol, then simulates each day with the full
MAX_SIMULTANEOUS_POSITIONS logic.

Usage:  python week_backtest.py
"""
import sys, warnings, datetime
warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf
sys.path.insert(0, ".")

from config import (EMA_PERIODS, MIN_BARS_3M, MIN_BARS_10M,
                    MAX_TRADES_PER_DAY, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
                    MAX_RISK_PER_TRADE, MIN_SHARES, MIN_STOP_DIST,
                    MAX_SIMULTANEOUS_POSITIONS, DTR_MAX_PCT,
                    FIXED_SHARES, MIN_DAILY_RANGE, BREAKEVEN_TRIGGER,
                    LEVEL_PROX_LONG, LEVEL_PROX_SHORT)
from ema_engine import (get_trend_10m, get_entry_signal_3m,
                        should_exit_10m, should_exit_rvol,
                        compute_trailing_stop, compute_dtr_atr_ratio)


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


def _print_trade(t):
    sign   = "+" if t["pnl"] >= 0 else ""
    reason = t.get("reason", "")
    if reason == "half@level":
        w = "H"                              # half exit at Rip's level
    else:
        w = "W" if t["pnl"] > 0 else ("B" if t["pnl"] == 0 else "L")
    print(f"     {w}  {t['dir'].upper():<5} {t['symbol']:6s} "
          f"{t['entry_time'].strftime('%H:%M')}->{t['exit_time'].strftime('%H:%M')}  "
          f"${t['entry']:.2f}->${t['exit']:.2f}  "
          f"x{t['shares']}sh  ${sign}{t['pnl']:.0f}  [{reason}]")


# ── Download all data once ─────────────────────────────────────────────────

def load_symbols(symbols: list) -> dict:
    """Download data for each symbol, return prepared dict.

    - 60d / 5m  -> resampled to 10m for trend direction
    - 7d  / 1m  -> resampled to 3m for entry signals + trade management
      (yfinance limit: 1m data only available for the last 7 days)
    - Pre-market high/low taken from the 1m feed (04:00-09:29 ET).
    """
    print("Downloading data (once)...")
    sym_data = {}
    for sym in symbols:
        # ---- 5m feed (60d) — trend only --------------------------------
        raw5 = yf.download(sym, period="60d", interval="5m",
                           progress=False, auto_adjust=True, prepost=True)
        if raw5.empty:
            print(f"  {sym}: no 5m data")
            continue
        if isinstance(raw5.columns, pd.MultiIndex):
            raw5.columns = raw5.columns.get_level_values(0)
        raw5 = raw5.tz_convert("US/Eastern")
        raw5_rth = raw5.between_time("09:30", "16:00")
        df_10m = _resample(raw5_rth, "10min")

        # ---- 1m feed (7d) — signal bars + pre-market levels ------------
        raw1 = yf.download(sym, period="7d", interval="1m",
                           progress=False, auto_adjust=True, prepost=True)
        if raw1.empty:
            print(f"  {sym}: no 1m data — skipping")
            continue
        if isinstance(raw1.columns, pd.MultiIndex):
            raw1.columns = raw1.columns.get_level_values(0)
        raw1 = raw1.tz_convert("US/Eastern")

        _pre = raw1.between_time("04:00", "09:29")
        pmh_by = {}; pml_by = {}
        for _dt, _grp in _pre.groupby(_pre.index.date):
            if not _grp.empty:
                pmh_by[_dt] = float(_grp["High"].max())
                pml_by[_dt] = float(_grp["Low"].min())

        raw1_rth = raw1.between_time("09:30", "16:00")
        df_3m = _resample(raw1_rth, "3min")

        sym_data[sym] = {"df_3m": df_3m, "df_10m": df_10m,
                         "pmh_by": pmh_by, "pml_by": pml_by,
                         "bar_staleness": 360}   # 3-min bars — stale after 6 min
        print(f"  {sym:6s} {len(df_3m)} x 3m bars  |  {len(df_10m)} x 10m bars")
    return sym_data


def load_symbols_5m(symbols: list) -> dict:
    """Download 60d of 5-min data, use 5m bars as signal bars.
    Use this for dates older than 7 days (beyond the 1m yfinance limit).
    """
    print("Downloading 5m data (60d range)...")
    sym_data = {}
    for sym in symbols:
        raw5 = yf.download(sym, period="60d", interval="5m",
                           progress=False, auto_adjust=True, prepost=True)
        if raw5.empty:
            print(f"  {sym}: no data")
            continue
        if isinstance(raw5.columns, pd.MultiIndex):
            raw5.columns = raw5.columns.get_level_values(0)
        raw5 = raw5.tz_convert("US/Eastern")

        _pre = raw5.between_time("04:00", "09:29")
        pmh_by = {}; pml_by = {}
        for _dt, _grp in _pre.groupby(_pre.index.date):
            if not _grp.empty:
                pmh_by[_dt] = float(_grp["High"].max())
                pml_by[_dt] = float(_grp["Low"].min())

        raw5_rth = raw5.between_time("09:30", "16:00")
        df_3m  = _resample(raw5_rth, "5min")   # 5m bars used as signal proxy
        df_10m = _resample(raw5_rth, "10min")

        sym_data[sym] = {"df_3m": df_3m, "df_10m": df_10m,
                         "pmh_by": pmh_by, "pml_by": pml_by,
                         "bar_staleness": 600}   # 5-min bars — stale after 10 min
        print(f"  {sym:6s} {len(df_3m)} x 5m bars (proxy)  |  {len(df_10m)} x 10m bars")
    return sym_data


# ── Single-day simulation (multi-symbol, shared position slots) ────────────

def sim_day(target: datetime.date, sym_data: dict,
            daily_plan: dict = None) -> list:
    """Run one trading day across all loaded symbols with 2-slot limit.

    daily_plan (optional): dict keyed by symbol, each value a dict with:
        bias       - 'long' | 'short' | 'both'
        support    - Rip's support pivot (used as tighter long stop)
        resistance - Rip's resistance pivot (used as tighter short stop)
    When supplied, only plan symbols are traded and directional bias is enforced.
    """
    # No plan = no trade.  Blind trading (no Rip levels, no bias) produces
    # noise entries with no R:R reference — skip the day entirely.
    if not daily_plan:
        return []

    # Active symbol set — always plan-restricted when we reach here
    active_syms = set(daily_plan.keys())
    active_syms = active_syms & set(sym_data.keys())   # must have data

    all_times = sorted({
        t for sym in active_syms
        for t in sym_data[sym]["df_3m"].index
        if t.date() == target
    })
    if not all_times:
        return []

    positions      = {}
    trades         = []
    trades_today   = {s: 0    for s in active_syms}
    lost_dir_today = {s: None for s in active_syms}

    for bar_time in all_times:
        t      = bar_time.time()
        is_eod = (t.hour == MARKET_CLOSE_HOUR and t.minute >= MARKET_CLOSE_MINUTE)
        no_new = (t.hour > MARKET_CLOSE_HOUR or
                  (t.hour == MARKET_CLOSE_HOUR and t.minute >= MARKET_CLOSE_MINUTE))

        # ---- Manage open positions ------------------------------------
        for sym in list(positions.keys()):
            pos      = positions[sym]
            df3      = sym_data[sym]["df_3m"]
            df3_now  = df3[df3.index <= bar_time]
            if df3_now.empty:
                continue
            cur = df3_now.iloc[-1]
            sh  = pos["shares"]

            # 10-min slice for exit signal
            df10_now = sym_data[sym]["df_10m"]
            df10_now = df10_now[df10_now.index <= bar_time]

            if is_eod:
                ep  = cur["close"]
                pnl = ((ep - pos["entry"]) * sh if pos["dir"] == "long"
                       else (pos["entry"] - ep) * sh)
                trades.append({**pos, "exit": ep, "exit_time": bar_time,
                               "pnl": pnl, "reason": "EOD"})
                _print_trade(trades[-1])
                del positions[sym]
                continue

            # ── Half exit at Rip's level ─────────────────────────────────
            # Exit 50 shares when price reaches the target level:
            #   Long  → Rip's resistance   Short → Rip's support
            # Locks guaranteed partial profit; remaining 50 run with the
            # tighter ratchet trailing stop for the bigger move.
            # Only fires once per position (half_exited flag).
            if not pos.get("half_exited", False):
                p_res    = pos.get("level_res")
                p_sup    = pos.get("level_sup")
                half_sh  = pos["shares"] // 2
                half_px  = None
                if pos["dir"] == "long"  and p_res is not None and cur["high"] >= p_res:
                    half_px = p_res
                elif pos["dir"] == "short" and p_sup is not None and cur["low"]  <= p_sup:
                    half_px = p_sup
                if half_px is not None and half_sh > 0:
                    half_pnl = ((half_px - pos["entry"]) * half_sh if pos["dir"] == "long"
                                else (pos["entry"] - half_px) * half_sh)
                    if half_pnl > 0:           # only exit if genuinely profitable
                        trades.append({**pos, "shares": half_sh,
                                       "exit": half_px, "exit_time": bar_time,
                                       "pnl": half_pnl, "reason": "half@level"})
                        _print_trade(trades[-1])
                        pos["shares"] -= half_sh
                    pos["half_exited"] = True  # don't check again even if pnl <= 0

            sh = pos["shares"]               # refresh — may have dropped to 50 after half exit

            # Compute unrealised BEFORE updating trailing stop.
            # best_unrealised is the high-water-mark of per-share profit seen
            # across all PREVIOUS bars — it is NOT updated until AFTER the
            # stop check below.  This prevents the intrabar phantom stop:
            #   old bug: bar close raises ratchet floor → same bar's low hits it
            #   fix:     ratchet floor is based on confirmed prior-bar HWM only
            unrealised = (cur["close"] - pos["entry"] if pos["dir"] == "long"
                          else pos["entry"] - cur["close"])

            new_stop = compute_trailing_stop(
                df3_now, pos["dir"], pos["stop"], pos["entry"],
                best_unrealised=pos.get("best_unrealised", 0.0))
            pos["stop"] = new_stop

            if pos["dir"] == "long" and cur["low"] <= pos["stop"]:
                pnl = (pos["stop"] - pos["entry"]) * sh
                trades.append({**pos, "exit": pos["stop"],
                               "exit_time": bar_time, "pnl": pnl,
                               "reason": "stop"})
                _print_trade(trades[-1])
                if pnl < 0: lost_dir_today[sym] = pos["dir"]
                trades_today[sym] += 1; del positions[sym]; continue

            if pos["dir"] == "short" and cur["high"] >= pos["stop"]:
                pnl = (pos["entry"] - pos["stop"]) * sh
                trades.append({**pos, "exit": pos["stop"],
                               "exit_time": bar_time, "pnl": pnl,
                               "reason": "stop"})
                _print_trade(trades[-1])
                if pnl < 0: lost_dir_today[sym] = pos["dir"]
                trades_today[sym] += 1; del positions[sym]; continue

            # Update the high-water-mark AFTER stop checks.
            # This ensures the ratchet floor can only tighten on the NEXT bar,
            # preventing the same bar's close from raising the floor and its
            # low from immediately hitting that same floor.
            pos["best_unrealised"] = max(pos.get("best_unrealised", 0.0), unrealised)

            # Exit: 10-min fast cloud flip OR relative volume dried up.
            # Once the ratchet stop is above entry (profit locked), suppress
            # the rvol exit — let the trailing stop manage the position.
            # Still allow rvol exit before the ratchet activates.
            stop_above_entry = (pos["stop"] > pos["entry"] if pos["dir"] == "long"
                                else pos["stop"] < pos["entry"])
            _exit_10m  = should_exit_10m(df10_now, pos["dir"])
            _exit_rvol = should_exit_rvol(df3_now) and not stop_above_entry
            _exit_reason = "10m exit" if _exit_10m else ("low rvol" if _exit_rvol else None)
            if _exit_10m or _exit_rvol:
                ep  = cur["close"]
                pnl = ((ep - pos["entry"]) * sh if pos["dir"] == "long"
                       else (pos["entry"] - ep) * sh)
                trades.append({**pos, "exit": ep, "exit_time": bar_time,
                               "pnl": pnl, "reason": _exit_reason})
                _print_trade(trades[-1])
                if pnl < 0: lost_dir_today[sym] = pos["dir"]
                trades_today[sym] += 1; del positions[sym]

        # ---- Entry ---------------------------------------------------
        if no_new or len(positions) >= MAX_SIMULTANEOUS_POSITIONS:
            continue

        for sym in sorted(active_syms):   # sorted = deterministic slot assignment
            if len(positions) >= MAX_SIMULTANEOUS_POSITIONS:
                break
            if sym in positions:
                continue
            if trades_today.get(sym, 0) >= MAX_TRADES_PER_DAY:
                continue

            df3     = sym_data[sym]["df_3m"]
            df3_now = df3[df3.index <= bar_time]
            if len(df3_now) < MIN_BARS_3M:
                continue
            cur       = df3_now.iloc[-1]
            staleness = sym_data[sym].get("bar_staleness", 360)
            if (bar_time - df3_now.index[-1]).total_seconds() > staleness:
                continue

            df10_now = sym_data[sym]["df_10m"]
            df10_now = df10_now[df10_now.index <= bar_time]
            if df10_now.empty:
                continue

            trend = get_trend_10m(df10_now)
            pmh   = sym_data[sym]["pmh_by"].get(target)
            pml   = sym_data[sym]["pml_by"].get(target)

            # DTR/ATR gate — skip if today's range is already ≥ 75% spent
            # e.g. Rip's sheet: "DTR: 6.21 vs ATR: 7.59  82%" → skip entry
            dtr_ratio = compute_dtr_atr_ratio(df10_now, target,
                                              bar_time=bar_time)
            if dtr_ratio > DTR_MAX_PCT:
                # Uncomment to debug: print(f"  DTR filter: {sym} {dtr_ratio:.0%} of ATR")
                continue

            # Daily range filter — 5-day avg range must be ≥ $7 so a $5 move is realistic
            recent_ranges = [
                float(g["high"].max() - g["low"].min())
                for d, g in df10_now.groupby(df10_now.index.date)
                if d < target
            ]
            if recent_ranges:
                avg_range = sum(recent_ranges[-5:]) / min(len(recent_ranges[-5:]), 5)
                if avg_range < MIN_DAILY_RANGE:
                    continue   # stock doesn't move enough to hit $5 target reliably

            # Pull Rip's levels + bias for this symbol (if plan provided)
            plan  = (daily_plan or {}).get(sym, {})
            bias  = plan.get("bias", "both")
            sup   = plan.get("support")
            res   = plan.get("resistance")

            signal, stop_price = get_entry_signal_3m(
                df3_now, trend, bar_time=bar_time, pmh=pmh, pml=pml,
                support=sup, resistance=res)
            if signal == "none" or signal == lost_dir_today.get(sym):
                continue

            entry_price = cur["close"]
            stop_dist   = abs(entry_price - stop_price)
            if stop_dist < MIN_STOP_DIST:
                continue

            # ── Level proximity gate ──────────────────────────────────────
            # Only enter when price is AT Rip's level — not chasing mid-range.
            # Long:  must be ≤ resistance + 1.5%  (at support, or fresh breakout)
            # Short: must be ≥ support   - 2.0%   (at resistance, or fresh breakdown)
            # When no level provided (symbol on plan but level unknown), skip check.
            if signal == "long" and res is not None:
                if entry_price > res * (1 + LEVEL_PROX_LONG):
                    continue   # chasing — price already too far above resistance

            if signal == "short" and sup is not None:
                if entry_price < sup * (1 - LEVEL_PROX_SHORT):
                    continue   # chasing — price already too far below support

            # Fixed 100 shares — trail stop to entry once up $5, let winner run
            n    = FIXED_SHARES
            risk = stop_dist * n
            slot = len(positions) + 1
            print(f"  >> {signal.upper():<5} {sym:6s} "
                  f"{bar_time.strftime('%H:%M')}  "
                  f"${entry_price:.2f}  stop=${stop_price:.2f}  "
                  f"x{n}sh  risk=${risk:.0f}  [{slot}/{MAX_SIMULTANEOUS_POSITIONS}]")
            positions[sym] = {
                "symbol": sym, "dir": signal,
                "entry": entry_price, "stop": stop_price,
                "shares": n, "entry_time": bar_time, "risk": risk,
                "best_unrealised": 0.0,   # HWM for intrabar-safe ratchet
                "level_res": res,          # Rip's resistance — half-exit target for longs
                "level_sup": sup,          # Rip's support    — half-exit target for shorts
                "half_exited": False,      # True once 50 shares sold at the level
            }

    return trades


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Full universe — download all names that might appear on any plan
    WATCHLIST = [
        "TSLA", "NVDA", "AAPL", "META", "AMD",
        "MSFT", "GOOGL", "AMZN", "NFLX", "CRWD",
    ]

    # ── Rip's daily plans (screenshot → dict) ─────────────────────────────
    # bias: 'long' | 'short' | 'both'
    # support / resistance: Rip's pivot levels from the sheet

    PLANS = {
        # May 12 — from Rip's 5/12 News Play + Day2/Day3 sheets (Tuesday)
        datetime.date(2026, 5, 12): {
            "TSLA":  {"support": 437.00, "resistance": 440.00},
            "AAPL":  {"support": 291.00, "resistance": 294.00},
            "META":  {"support": 598.00, "resistance": 600.00},
            "AMD":   {"support": 444.00, "resistance": 451.00},
            "GOOGL": {"support": 394.20, "resistance": 399.00},
            "AMZN":  {"support": 267.00, "resistance": 270.00},
            "MSFT":  {"support": 412.69, "resistance": 418.00},
        },
        # May 13 — extracted from Rip's 5/13 sheet (5m data used, 1m unavailable)
        # No bias — cloud signals determine direction
        datetime.date(2026, 5, 13): {
            "TSLA":  {"support": 432.00, "resistance": 437.00},
            "NVDA":  {"support": 223.00, "resistance": 225.00},
            "AAPL":  {"support": 293.60, "resistance": 295.00},
            "META":  {"support": 598.00, "resistance": 600.00},
            "GOOGL": {"support": None,   "resistance": None  },
            "AMZN":  {"support": 264.50, "resistance": 266.00},
        },
        # May 15 — from Rip's 5/15 News Play + Day2/Day3 sheets (Lotto Friday)
        datetime.date(2026, 5, 15): {
            "TSLA":  {"support": 430.00, "resistance": 437.00},
            "NVDA":  {"support": 228.00, "resistance": 232.00},
            "AAPL":  {"support": 291.00, "resistance": 294.00},
            "META":  {"support": 598.00, "resistance": 600.00},
            "AMD":   {"support": 435.00, "resistance": 449.00},
            "GOOGL": {"support": 394.20, "resistance": 399.00},
            "AMZN":  {"support": 262.00, "resistance": 267.00},
            "MSFT":  {"support": 412.69, "resistance": 418.00},
            "CRWD":  {"support": 570.00, "resistance": 574.00},
        },
        # May 19 — extracted from Rip's 5/19 sheet
        datetime.date(2026, 5, 19): {
            "MSFT":  {"support": 427.50, "resistance": 430.08},
            "TSLA":  {"support": 404.00, "resistance": 406.50},
            "AAPL":  {"support": 295.21, "resistance": 296.25},
            "NFLX":  {"support":  89.65, "resistance":  90.40},
            "GOOGL": {"support": 398.00, "resistance": 403.75},
        },
        # May 21 — extracted from Rip's 5/21 sheet
        datetime.date(2026, 5, 21): {
            "AMD":   {"support": 433.00, "resistance": 441.00},
            "MSFT":  {"support": 422.00, "resistance": 433.00},
            "AMZN":  {"support": 261.50, "resistance": 264.50},
            "GOOGL": {"support": 386.00, "resistance": 390.00},
            "TSLA":  {"support": 420.00, "resistance": 425.00},
            "NVDA":  {"support": 219.00, "resistance": 223.00},
        },
        # May 22 — from Rip's 5/22 daily levels sheet (Lotto Friday)
        # Corrected from daily_sheet.png:
        #   GOOGL was wrong (167.20 → 387.00) and MSFT support was in resistance range
        #   Expanded: added AAPL, AMD, AMZN, META, NFLX (all confirmed on sheet)
        datetime.date(2026, 5, 22): {
            "NVDA":  {"support": 219.20, "resistance": 220.46},
            "TSLA":  {"support": 417.80, "resistance": 420.00},
            "MSFT":  {"support": 417.53, "resistance": 419.00},
            "GOOGL": {"support": 387.00, "resistance": 389.00},  # fixed: was 167.20/168.50
            "AAPL":  {"support": 305.50, "resistance": 306.00},
            "AMD":   {"support": 458.50, "resistance": 464.50},
            "AMZN":  {"support": 268.44, "resistance": 269.50},
            "META":  {"support": 607.00, "resistance": 608.70},
            "NFLX":  {"support":  89.59, "resistance":  90.00},
        },
    }
    # ──────────────────────────────────────────────────────────────────────

    sym_data = load_symbols(WATCHLIST)

    # Last 5 trading days available in the 3m feed (7-day window)
    all_dates = sorted({
        d for sd in sym_data.values()
        for d in sd["df_3m"].index.date
    })
    last5 = all_dates[-5:]

    print(f"\nWATCHLIST: {' | '.join(WATCHLIST)}")
    print(f"MAX SIMULTANEOUS: {MAX_SIMULTANEOUS_POSITIONS}  |  "
          f"RISK/TRADE: ~${MAX_RISK_PER_TRADE:.0f}  |  "
          f"ENTRY CUTOFF: 15:00 ET")

    # all_plan_results collects (date, n, wins, losses, pnl, note) for every plan day
    all_plan_results = []
    all_plan_trades  = []

    for target in last5:
        plan = PLANS.get(target)
        dow  = target.strftime("%A")
        if not plan:
            print(f"\n{'='*60}")
            print(f"  {target}  ({dow})  [no Rip plan — skipped]")
            print(f"{'='*60}")
            continue                          # no plan = no trades (already enforced in sim_day)

        print(f"\n{'='*60}")
        print(f"  {target}  ({dow})  [Rip plan: {', '.join(plan.keys())}]")
        print(f"{'='*60}")
        day_trades = sim_day(target, sym_data, daily_plan=plan)
        all_plan_trades.extend(day_trades)

        day_pnl  = sum(t["pnl"] for t in day_trades)
        day_wins = sum(1 for t in day_trades if t["pnl"] > 0)
        day_loss = sum(1 for t in day_trades if t["pnl"] <= 0)
        if day_trades:
            print(f"  Day total: {len(day_trades)} trades  "
                  f"{day_wins}W/{day_loss}L  ${day_pnl:+.0f}")
        else:
            print("  No trades.")
        all_plan_results.append((target, len(day_trades), day_wins, day_loss, day_pnl, "1m"))

    # ── Older-date tests (5m proxy — 1m data unavailable beyond 7 days) ──────
    older_dates = sorted([d for d in PLANS if d < min(last5)])
    if older_dates:
        older_syms = set()
        for d in older_dates:
            older_syms.update(PLANS[d].keys())

        sym_data_5m = load_symbols_5m(list(older_syms))

        for target in older_dates:
            plan = PLANS[target]
            dow  = target.strftime("%A")
            print(f"\n{'='*60}")
            print(f"  {target}  ({dow})  [Rip plan: {', '.join(plan.keys())}]")
            print(f"  NOTE: 5m bars used as signal proxy (1m data > 7 days old)")
            print(f"{'='*60}")
            day_trades = sim_day(target, sym_data_5m, daily_plan=plan)
            all_plan_trades.extend(day_trades)

            day_pnl  = sum(t["pnl"] for t in day_trades)
            day_wins = sum(1 for t in day_trades if t["pnl"] > 0)
            day_loss = sum(1 for t in day_trades if t["pnl"] <= 0)
            if day_trades:
                print(f"  Day total: {len(day_trades)} trades  "
                      f"{day_wins}W/{day_loss}L  ${day_pnl:+.0f}")
            else:
                print("  No trades.")
            all_plan_results.append((target, len(day_trades), day_wins, day_loss, day_pnl, "5m"))

    # ── Combined summary — ALL Rip plan days ──────────────────────────────
    all_plan_results.sort(key=lambda x: x[0])   # chronological order
    ap_wins   = [t for t in all_plan_trades if t["pnl"] > 0]
    ap_losses = [t for t in all_plan_trades if t["pnl"] <= 0]
    ap_total  = sum(t["pnl"] for t in all_plan_trades)

    print(f"\n{'='*60}")
    print(f"  ALL RIP PLAN DAYS  ({all_plan_results[0][0]} to {all_plan_results[-1][0]})")
    print(f"{'='*60}")
    for (d, n, w, l, pnl, src) in all_plan_results:
        bar  = ("+" * w + "-" * l) if n else "."
        note = f"  [{src} bars]" if src == "5m" else ""
        print(f"  {d}  {n:2d} trades  {w}W/{l}L  ${pnl:+7.0f}  {bar}{note}")
    print(f"  {'-'*50}")
    print(f"  TOTAL         {len(all_plan_trades):2d} trades  "
          f"{len(ap_wins)}W/{len(ap_losses)}L  ${ap_total:+.0f}")
    if ap_wins:
        print(f"  Avg win : ${sum(t['pnl'] for t in ap_wins)/len(ap_wins):+.0f}"
              f"   Best: ${max(t['pnl'] for t in ap_wins):+.0f}")
    if ap_losses:
        print(f"  Avg loss: ${sum(t['pnl'] for t in ap_losses)/len(ap_losses):+.0f}"
              f"   Worst: ${min(t['pnl'] for t in ap_losses):+.0f}")
    plan_days_with_trades = sum(1 for r in all_plan_results if r[1] > 0)
    if plan_days_with_trades:
        print(f"  Avg P&L per active day: ${ap_total/plan_days_with_trades:+.0f}")
    print(f"{'='*60}")
