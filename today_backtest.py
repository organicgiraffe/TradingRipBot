"""
today_backtest.py — run today's intraday simulation on the stocks
from Rip's Daily Levels sheet.

Usage:  python today_backtest.py
"""
import sys, warnings, datetime
warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf
sys.path.insert(0, ".")

from config import (EMA_PERIODS, MIN_BARS_3M, MIN_BARS_10M,
                    MAX_TRADES_PER_DAY, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
                    MAX_RISK_PER_TRADE, MIN_SHARES, MIN_STOP_DIST,
                    MAX_SIMULTANEOUS_POSITIONS,
                    # Live-bot fixed-starter + pyramid sizing constants
                    FIXED_SHARES, FIXED_SHARES_HIGH, HIGH_PRICE_THRESHOLD,
                    STARTER_RATIO, ADD_TRIGGER_PROFIT,
                    MAX_RISK_DOLLARS, MAX_RISK_DOLLARS_HIGH)
from ema_engine import (get_trend_10m, get_entry_signal_3m,
                        should_exit_3m, compute_trailing_stop)


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


# ------------------------------------------------------------------ #
# Single-symbol runner (kept for standalone testing)
# ------------------------------------------------------------------ #

def run_today(symbol: str, shares: int = None,
              rip_levels: dict = None, date_str: str = None,
              df_spy_10m=None):
    """
    Download 60 days of 5-min data for EMA warm-up, simulate entries on today only.
    For multi-symbol runs with position limits, use run_multi_today() instead.
    """
    print(f"\n{'='*55}")
    label = f"  {symbol}  |  {'auto-sized' if shares is None else str(shares) + ' shares'}"
    print(label, end="")
    if rip_levels:
        print(f"  |  sup={rip_levels.get('support')}  "
              f"res={rip_levels.get('resistance')}  "
              f"bias={rip_levels.get('bias','any')}", end="")
    print(f"\n{'='*55}")

    raw = yf.download(symbol, period="60d", interval="5m",
                      progress=False, auto_adjust=True, prepost=True)
    if raw.empty:
        print("  No data returned.")
        return []
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.tz_convert("US/Eastern")

    all_dates = sorted(set(raw.between_time("09:30", "16:00").index.date))
    target = (datetime.date.fromisoformat(date_str) if date_str
              else all_dates[-1])

    _pre = raw.between_time("04:00", "09:29")
    pmh_by_date = {}; pml_by_date = {}
    for _dt, _grp in _pre.groupby(_pre.index.date):
        if not _grp.empty:
            pmh_by_date[_dt] = float(_grp["High"].max())
            pml_by_date[_dt] = float(_grp["Low"].min())

    pmh_today = pmh_by_date.get(target)
    pml_today = pml_by_date.get(target)
    print(f"  Pre-market: H=${pmh_today:.2f}  L=${pml_today:.2f}" if pmh_today else
          "  Pre-market: no data")
    print(f"  Simulating: {target}")

    raw = raw.between_time("09:30", "16:00")
    df_5m  = _resample(raw, "5min")
    df_10m = _resample(raw, "10min")
    print(f"  5-min bars: {len(df_5m)}  |  10-min bars: {len(df_10m)}")

    position = None; trades = []; trades_today = 0; lost_dir_today = None

    for i in range(MIN_BARS_3M, len(df_5m)):
        bar_time = df_5m.index[i]
        if bar_time.date() != target:
            continue

        df_3m_now  = df_5m.iloc[:i + 1]
        cur        = df_3m_now.iloc[-1]
        df_10m_now = df_10m[df_10m.index <= bar_time]
        if df_10m_now.empty:
            continue

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
                           "pnl": pnl, "reason": "EOD close"})
            _print_trade(trades[-1])
            position = None
            continue

        if position:
            new_stop = compute_trailing_stop(
                df_3m_now, position["dir"], position["stop"], position["entry"])
            position["stop"] = new_stop
            sh = position["shares"]

            if position["dir"] == "long" and cur["low"] <= position["stop"]:
                pnl = (position["stop"] - position["entry"]) * sh
                trades.append({**position, "exit": position["stop"],
                               "exit_time": bar_time, "pnl": pnl,
                               "reason": "trailing stop"})
                _print_trade(trades[-1])
                if pnl < 0: lost_dir_today = position["dir"]
                trades_today += 1; position = None; continue

            if position["dir"] == "short" and cur["high"] >= position["stop"]:
                pnl = (position["entry"] - position["stop"]) * sh
                trades.append({**position, "exit": position["stop"],
                               "exit_time": bar_time, "pnl": pnl,
                               "reason": "trailing stop"})
                _print_trade(trades[-1])
                if pnl < 0: lost_dir_today = position["dir"]
                trades_today += 1; position = None; continue

            if should_exit_3m(df_3m_now, position["dir"]):
                ep  = cur["close"]
                pnl = ((ep - position["entry"]) * sh if position["dir"] == "long"
                       else (position["entry"] - ep) * sh)
                trades.append({**position, "exit": ep, "exit_time": bar_time,
                               "pnl": pnl, "reason": "cloud exit"})
                _print_trade(trades[-1])
                if pnl < 0: lost_dir_today = position["dir"]
                trades_today += 1; position = None
            continue

        if no_new or trades_today >= MAX_TRADES_PER_DAY:
            continue

        signal, stop_price, _reason = get_entry_signal_3m(
            df_3m_now, trend, bar_time=bar_time,
            pmh=pmh_today, pml=pml_today,
            support=rip_levels.get("support") if rip_levels else None,
            resistance=rip_levels.get("resistance") if rip_levels else None)
        if signal == "none" or signal == lost_dir_today:
            continue

        entry_price = cur["close"]
        stop_dist   = abs(entry_price - stop_price)
        if stop_dist < MIN_STOP_DIST:
            continue
        n_shares = (max(MIN_SHARES, int(MAX_RISK_PER_TRADE / stop_dist))
                    if shares is None else shares)
        risk = stop_dist * n_shares
        print(f"  >> ENTRY  {signal.upper():<5} {bar_time.strftime('%H:%M')}  "
              f"@ ${entry_price:.2f}  stop=${stop_price:.2f}  "
              f"shares={n_shares}  risk=${risk:.2f}  trend={trend}")
        position = {"symbol": symbol, "dir": signal,
                    "entry": entry_price, "stop": stop_price,
                    "shares": n_shares, "entry_time": bar_time, "risk": risk}

    print(f"  Completed trades: {len(trades)}")
    if trades:
        total = sum(t["pnl"] for t in trades)
        wins  = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        print(f"  Win rate : {len(wins)}/{len(trades)}  |  Total P&L: ${total:+.2f}")
        if wins:   print(f"  Avg win  : ${sum(t['pnl'] for t in wins)/len(wins):+.2f}")
        if losses: print(f"  Avg loss : ${sum(t['pnl'] for t in losses)/len(losses):+.2f}")
    return trades


# ------------------------------------------------------------------ #
# Multi-symbol runner — all symbols share position slots
# ------------------------------------------------------------------ #

def run_multi_today(setups: dict, date_str: str = None,
                    interval: str = "5m", period: str = "60d",
                    entry_freq: str = "5min", stale_secs: int = 600,
                    sizing: str = "risk", atr_ratchet: bool = False) -> list:
    """
    Run all symbols in ONE interleaved time loop, enforcing
    MAX_SIMULTANEOUS_POSITIONS across all symbols.

    setups: {symbol: rip_levels_dict}
      rip_levels_dict keys: 'support', 'resistance', 'bias', 'note'

    interval/period: yfinance download params.  Default 5m/60d.  Pass
      interval="1m", period="7d", entry_freq="3min" to mirror the LIVE bot's
      3-minute entry timeframe (yfinance caps 1m history at ~8 days).
    entry_freq:  pandas resample rule for the entry/management timeframe
      ("5min" default, "3min" to match live).  The 10-min trend frame is
      always resampled to "10min" regardless.
    stale_secs:  max age of a symbol's latest bar before its entry is skipped.

    Returns list of all completed trades across all symbols.
    """
    print(f"\n{'='*65}")
    print(f"  MULTI-SYMBOL  |  {len(setups)} setups  "
          f"|  max {MAX_SIMULTANEOUS_POSITIONS} simultaneous positions"
          f"  |  {interval} -> {entry_freq} entry frame")
    print(f"{'='*65}")

    # ── Download & prepare data for each symbol ───────────────────────────
    sym_data = {}
    for sym in setups:
        raw = yf.download(sym, period=period, interval=interval,
                          progress=False, auto_adjust=True, prepost=True)
        if raw.empty:
            print(f"  {sym}: no data — skipped")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.tz_convert("US/Eastern")

        _pre = raw.between_time("04:00", "09:29")
        pmh_by = {}; pml_by = {}
        for _dt, _grp in _pre.groupby(_pre.index.date):
            if not _grp.empty:
                pmh_by[_dt] = float(_grp["High"].max())
                pml_by[_dt] = float(_grp["Low"].min())

        raw_rth = raw.between_time("09:30", "16:00")
        df_5m   = _resample(raw_rth, entry_freq)   # entry/management frame
        df_10m  = _resample(raw_rth, "10min")

        # 5-day avg daily range (ATR proxy) — same definition the live bot uses
        # in subscribe_bars.  Feeds the ATR-scaled ratchet.
        _daily = raw_rth.groupby(raw_rth.index.date).agg(
            {"High": "max", "Low": "min"})
        _ranges = (_daily["High"] - _daily["Low"]).dropna()
        sym_atr = float(_ranges.tail(5).mean()) if len(_ranges) else 0.0

        sym_data[sym] = {
            "df_5m": df_5m, "df_10m": df_10m,
            "pmh_by": pmh_by, "pml_by": pml_by, "atr": sym_atr,
        }
        print(f"  {sym:6s} loaded  {len(df_5m)} bars  ATR=${sym_atr:.2f}")

    if not sym_data:
        print("  No data for any symbol.")
        return []

    # ── Determine target date ─────────────────────────────────────────────
    if date_str:
        target = datetime.date.fromisoformat(date_str)
    else:
        target = sorted(
            d for sd in sym_data.values() for d in sd["df_5m"].index.date
        )[-1]

    print(f"\n  Simulating: {target}")
    for sym, sd in sym_data.items():
        pmh = sd["pmh_by"].get(target)
        pml = sd["pml_by"].get(target)
        lvl = setups.get(sym) or {}
        note = f"  sup={lvl.get('support')}  res={lvl.get('resistance')}" if lvl.get('support') else ""
        if pmh:
            print(f"  {sym:6s}  PM H=${pmh:.2f}  L=${pml:.2f}{note}")

    # ── Merged timeline for target date ───────────────────────────────────
    all_times = sorted({
        t for sd in sym_data.values()
        for t in sd["df_5m"].index
        if t.date() == target
    })

    # ── Simulation state ──────────────────────────────────────────────────
    positions      = {}   # sym -> position dict  (max MAX_SIMULTANEOUS_POSITIONS)
    all_trades     = []
    trades_today   = {sym: 0    for sym in sym_data}
    lost_dir_today = {sym: None for sym in sym_data}

    print()

    for bar_time in all_times:
        t      = bar_time.time()
        is_eod = (t.hour == MARKET_CLOSE_HOUR and t.minute >= MARKET_CLOSE_MINUTE)
        no_new = (t.hour > MARKET_CLOSE_HOUR or
                  (t.hour == MARKET_CLOSE_HOUR and t.minute >= MARKET_CLOSE_MINUTE))

        # ---- Manage every open position ----------------------------------
        for sym in list(positions.keys()):
            pos     = positions[sym]
            df5     = sym_data[sym]["df_5m"]
            df3_now = df5[df5.index <= bar_time]
            if df3_now.empty:
                continue
            cur = df3_now.iloc[-1]

            # ---- Pyramid add-in (live sizing) ----------------------------
            # When the starter shows ADD_TRIGGER_PROFIT ($3/sh) of favorable
            # excursion, the live bot adds the remaining shares at market.
            # Model the add filling at the trigger level (entry ± $3) and
            # blend the entry, exactly like ibkr_client._on_add_fill.
            if (not pos.get("add_triggered") and pos.get("shares_add", 0) > 0):
                if pos["dir"] == "long":
                    _reached = cur["high"] >= pos["entry"] + ADD_TRIGGER_PROFIT
                    _add_px  = pos["entry"] + ADD_TRIGGER_PROFIT
                else:
                    _reached = cur["low"] <= pos["entry"] - ADD_TRIGGER_PROFIT
                    _add_px  = pos["entry"] - ADD_TRIGGER_PROFIT
                if _reached:
                    _ash     = pos["shares_add"]
                    _total   = pos["shares"] + _ash
                    pos["entry"]         = (pos["entry"] * pos["shares"]
                                            + _add_px * _ash) / _total
                    pos["shares"]        = _total
                    pos["add_triggered"] = True
                    print(f"     ADD   {sym:6s} {pos['dir'].upper():<5} "
                          f"{bar_time.strftime('%H:%M')}  +{_ash}sh @${_add_px:.2f}  "
                          f"avg=${pos['entry']:.2f}  total={_total}sh")

            sh  = pos["shares"]

            if is_eod:
                ep  = cur["close"]
                pnl = ((ep - pos["entry"]) * sh if pos["dir"] == "long"
                       else (pos["entry"] - ep) * sh)
                all_trades.append({**pos, "exit": ep, "exit_time": bar_time,
                                   "pnl": pnl, "reason": "EOD close"})
                _print_trade(all_trades[-1])
                del positions[sym]
                continue

            # Update running high-water-mark of per-share profit (HWM never
            # decreases), exactly like the live bot's best_unrealised.
            _bar_bu = (cur["high"] - pos["entry"] if pos["dir"] == "long"
                       else pos["entry"] - cur["low"])
            pos["best_unrealised"] = max(pos["best_unrealised"], _bar_bu)
            new_stop = compute_trailing_stop(
                df3_now, pos["dir"], pos["stop"], pos["entry"],
                best_unrealised=pos["best_unrealised"],
                atr=(sym_data[sym]["atr"] if atr_ratchet else 0.0))
            pos["stop"] = new_stop

            if pos["dir"] == "long" and cur["low"] <= pos["stop"]:
                pnl = (pos["stop"] - pos["entry"]) * sh
                all_trades.append({**pos, "exit": pos["stop"],
                                   "exit_time": bar_time, "pnl": pnl,
                                   "reason": "trailing stop"})
                _print_trade(all_trades[-1])
                if pnl < 0: lost_dir_today[sym] = pos["dir"]
                trades_today[sym] += 1; del positions[sym]; continue

            if pos["dir"] == "short" and cur["high"] >= pos["stop"]:
                pnl = (pos["entry"] - pos["stop"]) * sh
                all_trades.append({**pos, "exit": pos["stop"],
                                   "exit_time": bar_time, "pnl": pnl,
                                   "reason": "trailing stop"})
                _print_trade(all_trades[-1])
                if pnl < 0: lost_dir_today[sym] = pos["dir"]
                trades_today[sym] += 1; del positions[sym]; continue

            if should_exit_3m(df3_now, pos["dir"]):
                ep  = cur["close"]
                pnl = ((ep - pos["entry"]) * sh if pos["dir"] == "long"
                       else (pos["entry"] - ep) * sh)
                all_trades.append({**pos, "exit": ep, "exit_time": bar_time,
                                   "pnl": pnl, "reason": "cloud exit"})
                _print_trade(all_trades[-1])
                if pnl < 0: lost_dir_today[sym] = pos["dir"]
                trades_today[sym] += 1; del positions[sym]

        # ---- Entry: fill open position slots ----------------------------
        if no_new or len(positions) >= MAX_SIMULTANEOUS_POSITIONS:
            continue

        for sym, rip_levels in setups.items():
            if len(positions) >= MAX_SIMULTANEOUS_POSITIONS:
                break
            if sym not in sym_data:
                continue
            if sym in positions:
                continue
            if trades_today[sym] >= MAX_TRADES_PER_DAY:
                continue

            df5     = sym_data[sym]["df_5m"]
            df3_now = df5[df5.index <= bar_time]
            if len(df3_now) < MIN_BARS_3M:
                continue
            cur = df3_now.iloc[-1]
            # Skip if this symbol's latest bar is stale (> stale_secs old)
            if (bar_time - df3_now.index[-1]).total_seconds() > stale_secs:
                continue

            df10     = sym_data[sym]["df_10m"]
            df10_now = df10[df10.index <= bar_time]
            if df10_now.empty:
                continue

            trend = get_trend_10m(df10_now)
            pmh   = sym_data[sym]["pmh_by"].get(target)
            pml   = sym_data[sym]["pml_by"].get(target)
            lvl   = rip_levels or {}

            signal, stop_price, _reason = get_entry_signal_3m(
                df3_now, trend, bar_time=bar_time,
                pmh=pmh, pml=pml,
                support=lvl.get("support"),
                resistance=lvl.get("resistance"),
            )
            if signal == "none" or signal == lost_dir_today[sym]:
                continue

            entry_price = cur["close"]
            stop_dist   = abs(entry_price - stop_price)
            if stop_dist < MIN_STOP_DIST:
                continue

            if sizing == "live":
                # LIVE BOT sizing: fixed full size (50 if >=$500 else 100),
                # enter STARTER_RATIO now, add the rest on +$3/sh confirmation.
                # Risk check uses STARTER shares only (matches ibkr_client).
                n_full   = (FIXED_SHARES_HIGH if entry_price >= HIGH_PRICE_THRESHOLD
                            else FIXED_SHARES)
                n_shares = max(1, int(n_full * STARTER_RATIO))   # starter
                n_add    = n_full - n_shares
                risk     = stop_dist * n_shares
                risk_cap = (MAX_RISK_DOLLARS_HIGH if entry_price >= HIGH_PRICE_THRESHOLD
                            else MAX_RISK_DOLLARS)
                if risk > risk_cap:
                    continue   # live bot SKIPs when starter risk exceeds cap
            else:
                # Risk-based sizing (original behaviour)
                n_shares = max(MIN_SHARES, int(MAX_RISK_PER_TRADE / stop_dist))
                n_add    = 0
                risk     = stop_dist * n_shares
            slot     = len(positions) + 1

            print(f"  >> ENTRY  {signal.upper():<5} {sym:6s} "
                  f"{bar_time.strftime('%H:%M')}  "
                  f"@ ${entry_price:.2f}  stop=${stop_price:.2f}  "
                  f"shares={n_shares}+{n_add}add  risk=${risk:.2f}  "
                  f"[{slot}/{MAX_SIMULTANEOUS_POSITIONS}]  trend={trend}")

            positions[sym] = {
                "symbol": sym, "dir": signal,
                "entry": entry_price, "stop": stop_price,
                "shares": n_shares, "entry_time": bar_time, "risk": risk,
                # pyramid state (live sizing only; n_add=0 disables it)
                "shares_add": n_add, "add_triggered": False,
                # running high-water-mark per-share profit (drives the ratchet,
                # matching how the live bot tracks best_unrealised)
                "best_unrealised": 0.0,
            }

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n  {'-'*55}")
    wins   = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    total  = sum(t["pnl"] for t in all_trades)
    print(f"  Completed: {len(all_trades)} trades  "
          f"|  {len(wins)}W / {len(losses)}L  |  ${total:+.2f}")
    if wins:   print(f"  Avg win  : ${sum(t['pnl'] for t in wins)/len(wins):+.2f}")
    if losses: print(f"  Avg loss : ${sum(t['pnl'] for t in losses)/len(losses):+.2f}")

    # Per-symbol breakdown
    syms_traded = sorted({t["symbol"] for t in all_trades})
    if len(syms_traded) > 1:
        print()
        for sym in syms_traded:
            sym_trades = [t for t in all_trades if t["symbol"] == sym]
            sym_total  = sum(t["pnl"] for t in sym_trades)
            sym_wins   = sum(1 for t in sym_trades if t["pnl"] > 0)
            print(f"  {sym:6s}: {len(sym_trades)} trade(s)  "
                  f"{sym_wins}W/{len(sym_trades)-sym_wins}L  ${sym_total:+.2f}")

    return all_trades


def _print_trade(t):
    sign = "+" if t["pnl"] >= 0 else ""
    sh   = t.get("shares", "?")
    print(f"     CLOSE {t['symbol']:6s} {t['dir'].upper():<5} "
          f"{t['entry_time'].strftime('%H:%M')} -> {t['exit_time'].strftime('%H:%M')}  "
          f"${t['entry']:.2f} -> ${t['exit']:.2f}  "
          f"x{sh}sh  pnl=${sign}{t['pnl']:.2f}  [{t['reason']}]")


# ------------------------------------------------------------------ #
# Today's picks — read directly from Rip's Daily Levels sheet
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("\nRIP'S DAILY PLAYS  —  May 22, 2026  (Lotto Friday)")
    print("=" * 65)
    print("  SOURCE: Rip's Day2/Day3 Play + News Play + Wakeup Summary")
    print()
    print("  DAY2/DAY3 CONTINUATION PLAYS:")
    print("  TSLA  — Daily looking good, long over PMH, watch flow       bias=LONG")
    print("  AAOI  — Bullish bias long over YH or 34/50 EMA curl         bias=LONG")
    print("  INTU  — Earnings Day2, bearish under 305/EMA break          bias=SHORT")
    print("  DELL  — Evercore ISI Tactical Outperform add, PT $270       bias=LONG")
    print()
    print("  TIER 1 BANK CATALYST PLAYS:")
    print("  SPOT  — JPMorgan raises PT  (Tier 1)                        bias=LONG")
    print("  INSP  — BofA downgrade  (Tier 1)                            bias=SHORT")
    print()
    print("  SKIP:  AMD (wide $16 range, no clean bias)")
    print("  SKIP:  NVDA (Inside Day, 'no go under 220')")
    print("=" * 65)

    setups = {
        "TSLA": {"support": 449.45, "resistance": 452.00, "bias": "long",
                 "note": "Daily looking good, long over PMH"},
        "AAOI": {"support": 59.50,  "resistance": 63.90,  "bias": "long",
                 "note": "Bullish long over YH or 34/50 EMA curl"},
        "INTU": {"support": 302.40, "resistance": 309.00, "bias": "short",
                 "note": "Earnings Day2 — bearish under 305"},
        "DELL": {"support": None,   "resistance": None,   "bias": "long",
                 "note": "Evercore ISI Tactical Outperform, PT $270"},
        "SPOT": {"support": None,   "resistance": None,   "bias": "long",
                 "note": "JPMorgan raises PT (Tier 1 bank)"},
        "INSP": {"support": None,   "resistance": None,   "bias": "short",
                 "note": "BofA downgrade (Tier 1 bank)"},
    }

    all_trades = run_multi_today(setups)

    wins   = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    total  = sum(t["pnl"] for t in all_trades)
    print(f"\n{'='*65}")
    print(f"  TOTAL TODAY  |  {len(all_trades)} trades  "
          f"|  {len(wins)}W / {len(losses)}L  |  ${total:+.2f}")
    if wins:   print(f"  Avg win  : ${sum(t['pnl'] for t in wins)/len(wins):+.2f}")
    if losses: print(f"  Avg loss : ${sum(t['pnl'] for t in losses)/len(losses):+.2f}")
    print(f"{'='*65}")
