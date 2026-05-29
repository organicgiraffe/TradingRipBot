"""
manual_backtest.py — replay today with yfinance 1-min bars resampled to 3m/10m.
Mirrors the real bot EXACTLY:
  - MAX_SIMULTANEOUS_POSITIONS = 1 globally (one trade across ALL symbols)
  - Symbols evaluated in order each 3-min bar close — first valid signal wins
  - Re-entry blocked per symbol after >$50 loss in same direction
  - level_still_ahead fix, half-exit at level or $5/share target
  - RVOL exit only when volume dead AND C2 flipped against
  - Ratchet trailing stop ($3 start, $2 giveback)
  - EOD forced close at MARKET_CLOSE_MINUTE (15:50)
"""
import warnings; warnings.filterwarnings("ignore")
import sys, yfinance as yf, pandas as pd

sys.path.insert(0, r"C:\Users\nicol\OneDrive\Documents\Tradingbot")
from ema_engine import (compute_emas, get_trend_10m,
                        get_entry_signal_3m, get_gap_signal_3m,
                        get_open_cloud_break_signal_3m)
from config import (
    MIN_BARS_3M, RATCHET_START, RATCHET_GIVEBACK, PROFIT_TARGET_SHARE,
    FIXED_SHARES, FIXED_SHARES_HIGH, HIGH_PRICE_THRESHOLD,
    STARTER_RATIO, ADD_TRIGGER_PROFIT,
    MAX_TRADES_PER_DAY,
    MAX_RISK_DOLLARS, MAX_RISK_DOLLARS_HIGH, MIN_DAILY_RANGE,
    DTR_MAX_PCT, DTR_EXEMPT_ATR, FIRST_ENTRY_MINUTE,
    LAST_ENTRY_HOUR, LAST_ENTRY_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
    RVOL_EXIT_MULT, LEVEL_PROX_LONG, LEVEL_PROX_SHORT,
)

PLAN = {
    "MU":   {"support":  950.00, "resistance":  970.00},
    "SNDK": {"support": 1625.00, "resistance": 1635.00},
}
TODAY = pd.Timestamp("2026-05-27").date()
ENABLE_GAP_ENTRIES = "--cloud-only" not in sys.argv and "--no-gap" not in sys.argv


class Bar:
    def __init__(self, ts, o, h, l, c, v):
        self.date=ts; self.open=o; self.high=h
        self.low=l; self.close=c; self.volume=int(v)


def download_and_resample(sym):
    df = yf.download(sym, period="5d", interval="1m",
                     auto_adjust=True, progress=False, prepost=True)
    if df.empty: return None, None, None, None, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.index = (df.index.tz_convert("America/New_York")
                if df.index.tz else df.index.tz_localize("America/New_York"))
    today_pre = df[df.index.date == TODAY].between_time("04:00", "09:29")
    pmh = float(today_pre["high"].max()) if not today_pre.empty else None
    pml = float(today_pre["low"].min()) if not today_pre.empty else None
    df_all = df.between_time("04:00", "15:59")
    df = df.between_time("09:30", "15:59")
    # 1-min bars for today — index labeled with close time (open + 1 min).
    # Used for intrabar ratchet/stop management so we see peaks within each
    # 3-min bar, not just at bar close.
    df1_today = df[df.index.date == TODAY].copy()
    df1_today.index = df1_today.index + pd.Timedelta(minutes=1)
    def rsmp(m):
        src = df_all if m == 3 else df
        r = src.resample(f"{m}min", closed="left", label="left").agg(
            open=("open","first"), high=("high","max"),
            low=("low","min"), close=("close","last"), volume=("volume","sum")
        ).dropna(subset=["close"])
        r.index += pd.Timedelta(minutes=m)
        return r
    return rsmp(3), rsmp(10), df1_today, pmh, pml


def to_bars(df):
    return [Bar(r.Index,r.open,r.high,r.low,r.close,r.volume) for r in df.itertuples()]


def trail_stop(sig, stop, entry, best, ema50):
    t = ema50
    if sig == "long":
        if best >= RATCHET_START:
            t = max(t, entry + max(0.0, best - RATCHET_GIVEBACK))
        return max(stop, t)
    else:
        if best >= RATCHET_START:
            t = min(t, entry - max(0.0, best - RATCHET_GIVEBACK))
        new = min(stop, t)
        # Guard: crash entries have ema50 << entry, which would immediately trail
        # the stop below entry and trigger on the next bar.  Don't move the stop
        # below entry_price until the ratchet has activated ($3 profit seen).
        if best < RATCHET_START and new < entry:
            return stop   # keep initial risk stop unchanged
        return new


# ── Download all data upfront ─────────────────────────────────────────────────
print(f"\n{'='*78}")
mode = "gap-enabled" if ENABLE_GAP_ENTRIES else "cloud-only"
print(f"  BACKTEST  --  {TODAY}  --  1 global trade at a time  ({mode})")
print(f"{'='*78}\n")
print("  Downloading data...", end="", flush=True)

sym_data     = {}   # sym -> (bars3_all, bars10_all, today3, shares)
trades_today = {}
sym_atr    = {}   # sym -> 5-day avg daily range ($) — used for priority scoring

skipped_atr = {}
for sym, lev in PLAN.items():
    df3, df10, df1_today, pmh, pml = download_and_resample(sym)
    if df3 is None: continue
    b10 = to_bars(df10); b3 = to_bars(df3)
    t3  = [b for b in b3 if pd.Timestamp(b.date).date() == TODAY]
    if not t3: continue
    px  = t3[len(t3)//2].close
    sh  = FIXED_SHARES_HIGH if px >= HIGH_PRICE_THRESHOLD else FIXED_SHARES
    # 5-day ATR: avg(daily high-low) across recent days
    daily_ranges = (
        pd.DataFrame({"h": [b.high for b in b3], "l": [b.low for b in b3],
                      "d": [pd.Timestamp(b.date).date() for b in b3]})
        .groupby("d").apply(lambda g: g["h"].max() - g["l"].min(), include_groups=False)
        .tail(5).mean()
    )
    atr = round(daily_ranges, 2)
    sym_atr[sym] = atr
    if atr < MIN_DAILY_RANGE:
        skipped_atr[sym] = atr
        continue   # too low ATR — not worth trading
    sym_data[sym]     = (b3, b10, t3, sh, pmh, pml, df1_today)
    trades_today[sym] = 0

print(f" done ({len(sym_data)} active, {len(skipped_atr)} skipped)\n")
print(f"  ATR (5-day avg daily range)  — min=${MIN_DAILY_RANGE:.0f}:")
for sym in sorted(sym_atr, key=sym_atr.get, reverse=True):
    flag = "  SKIP (low ATR)" if sym in skipped_atr else ""
    print(f"    {sym:<5}  ${sym_atr[sym]:.2f}{flag}")
print()

# ── Build a unified timeline of all 3-min bar-close times today ───────────────
all_times = sorted(set(
    pd.Timestamp(b.date) for sym in sym_data for b in sym_data[sym][2]
))

# ── Walk the timeline bar by bar — global 1-position-at-a-time ───────────────
in_trade   = False
open_trade = None    # dict with trade state while position is live
all_trades = []

for bt in all_times:
    # ── If we're in a trade, manage it first ─────────────────────────────────
    if in_trade:
        t = open_trade
        sym   = t["sym"]
        sig   = t["sig"]
        entry = t["entry"]
        b3, b10, today3, shares, pmh, pml, df1_today = sym_data[sym]

        # Find the bar for this symbol at this time
        fb = next((b for b in today3 if pd.Timestamp(b.date) == bt), None)
        if fb is None:
            continue   # this symbol didn't have a bar at this exact time

        fbt = pd.Timestamp(fb.date)
        b3f = b3[:b3.index(fb)+1]
        df3f = compute_emas(b3f)
        cur  = df3f.iloc[-1]

        exit_px  = None
        exit_why = None

        # EOD close — check before 1-min loop
        if (fbt.hour > MARKET_CLOSE_HOUR or
                (fbt.hour == MARKET_CLOSE_HOUR and fbt.minute >= MARKET_CLOSE_MINUTE)):
            exit_px = fb.close; exit_why = "eod_close"

        # ── 1-min bar management: ratchet + half-exit + stop ─────────────
        # Walk the three 1-min bars that make up this 3-min period.
        # Using 1-min highs/lows means the ratchet activates when price peaks
        # (not just at bar close) — prevents giving back intrabar gains.
        # Ordering: update stop → check half-exit → check stop → update HWM.
        # "Update HWM last" is the anti-phantom rule: the same bar that raises
        # the ratchet floor cannot immediately hit it.
        if exit_px is None:
            sup = PLAN[sym]["support"]; res = PLAN[sym]["resistance"]
            if sig=="long": level_ahead = res is not None and res > entry
            else:           level_ahead = sup is not None and sup < entry
            m1_window = df1_today[
                (df1_today.index > fbt - pd.Timedelta(minutes=3)) &
                (df1_today.index <= fbt)
            ]
            for _, m1 in m1_window.iterrows():
                if exit_px is not None:
                    break

                # Step 0 — Add-in trigger (before stop update so the ratchet
                # starts from the new average entry, not the starter entry)
                if not t["add_triggered"] and t["add_shares"] > 0:
                    starter_profit = (
                        (t["entry"] - m1["close"]) if sig == "short"
                        else (m1["close"] - t["entry"])
                    )
                    if starter_profit >= ADD_TRIGGER_PROFIT:
                        add_px  = m1["close"]
                        s_sh    = t["remaining"]   # shares already open
                        a_sh    = t["add_shares"]
                        tot_sh  = s_sh + a_sh
                        avg_px  = (t["entry"] * s_sh + add_px * a_sh) / tot_sh
                        # Update trade state to reflect full position.
                        # t["shares"] stays as starter count (display/risk).
                        # t["remaining"] tracks the live share count.
                        t["entry"]        = avg_px
                        t["remaining"]    = tot_sh
                        t["add_triggered"] = True
                        t["add_entry"]    = add_px
                        # best_unr restarts from new average entry
                        t["best_unr"]     = max(0.0, (avg_px - m1["close"])
                                                      if sig == "short"
                                                      else (m1["close"] - avg_px))
                        # level_ahead still relative to original entry price
                        # (don't recalculate — it was set at trade open)
                        entry = avg_px   # local var used by ratchet/stop below

                # Step 1 — Update trailing stop using PREVIOUS best_unr
                t["stop_cur"] = trail_stop(sig, t["stop_cur"], entry,
                                           t["best_unr"], cur.ema50)

                # Step 2 — Half-exit at Rip's level (1-min resolution)
                if not t["half_done"]:
                    hpx = None; hrsn = "half@level"
                    if sig=="long"  and res and m1["high"] >= res: hpx = res
                    elif sig=="short" and sup and m1["low"]  <= sup: hpx = sup
                    m1_unr = ((m1["close"] - entry) if sig=="long"
                              else (entry - m1["close"]))
                    if hpx is None and not level_ahead and m1_unr >= PROFIT_TARGET_SHARE:
                        hpx = m1["close"]; hrsn = "half@target"
                    if hpx is not None:
                        # Use actual remaining shares for half-exit — not sym_data
                        # full-position count.  If add hasn't triggered yet, remaining
                        # is starter-only (e.g. 25sh); if add triggered, remaining is
                        # the full position (e.g. 50sh).  Both cases handled correctly.
                        half_sh = t["remaining"] // 2
                        hp = ((hpx-entry) if sig=="long" else (entry-hpx)) * half_sh
                        if hp > 0:
                            t["half_pnl"]  = hp
                            t["remaining"] -= half_sh
                        t["half_done"] = True

                # Step 3 — Hard stop using 1-min low/high
                if sig=="long"  and m1["low"]  <= t["stop_cur"]:
                    exit_px = t["stop_cur"]; exit_why = "stop"; break
                elif sig=="short" and m1["high"] >= t["stop_cur"]:
                    exit_px = t["stop_cur"]; exit_why = "stop"; break

                # Step 4 — Update HWM AFTER stop check (anti-phantom guarantee)
                if sig=="long":
                    t["best_unr"] = max(t["best_unr"], m1["high"] - entry)
                    t["peak_unr"] = max(t["peak_unr"], m1["high"] - entry)
                else:
                    t["best_unr"] = max(t["best_unr"], entry - m1["low"])
                    t["peak_unr"] = max(t["peak_unr"], entry - m1["low"])

        # ── 3-min checks: RVOL + C2 (only if still in trade) ────────────
        # Volume/momentum exits only make sense at bar closes — don't move to 1m.
        if exit_px is None:
            locked     = (t["stop_cur"] > entry) if sig=="long" else (t["stop_cur"] < entry)
            c2_against = (cur.ema5 < cur.ema12) if sig=="long" else (cur.ema5 > cur.ema12)
            rvol_dead  = cur.vol_ma20>0 and (cur.volume/cur.vol_ma20) < RVOL_EXIT_MULT
            if rvol_dead and not locked and c2_against:
                exit_px = cur.close; exit_why = "rvol+C2"

        if exit_px is not None:
            runner  = ((exit_px-entry) if sig=="long" else (entry-exit_px)) * t["remaining"]
            total   = t["half_pnl"] + runner
            held_m  = int((fbt - t["entry_t"]).total_seconds()//60)
            all_trades.append({
                "sym": sym, "time": t["entry_t"], "sig": sig.upper(),
                "entry": t.get("original_entry", entry), "avg_entry": entry,
                "stop": t["init_stop"],
                "dist": t["dist"], "shares": t["shares"], "risk": t["dist"]*t["shares"],
                "reason": t["reason"], "trend": t["trend"],
                "half_pnl": t["half_pnl"], "runner_pnl": runner, "total_pnl": total,
                "exit_px": exit_px, "exit_t": fbt, "exit_why": exit_why,
                "held": held_m, "peak": t["peak_unr"],
                "add_triggered": t["add_triggered"], "add_entry": t["add_entry"],
                "shares_full": t["shares_full"],
            })
            trades_today[sym] += 1
            in_trade   = False
            open_trade = None
        continue   # done managing — don't look for new entries this bar

    # ── No open position — scan all symbols for a signal ─────────────────────
    if (bt.hour > LAST_ENTRY_HOUR or
            (bt.hour == LAST_ENTRY_HOUR and bt.minute >= LAST_ENTRY_MINUTE)):
        continue

    # Collect ALL valid signals this bar, then pick highest ATR
    candidates = []
    for sym in sym_data:
        b3, b10, today3, shares, pmh, pml, df1_today = sym_data[sym]
        sup = PLAN[sym]["support"]; res = PLAN[sym]["resistance"]
        if trades_today.get(sym, 0) >= MAX_TRADES_PER_DAY:
            continue

        bar = next((b for b in today3 if pd.Timestamp(b.date)==bt), None)
        if bar is None: continue

        b10s = [b for b in b10 if pd.Timestamp(b.date)<=bt]
        b3s  = b3[:b3.index(bar)+1]
        if len(b10s)<5 or len(b3s)<MIN_BARS_3M: continue

        df10e = compute_emas(b10s)
        df3e  = compute_emas(b3s)
        trend = get_trend_10m(df10e)

        if ENABLE_GAP_ENTRIES:
            sig, stop_px, reason = get_open_cloud_break_signal_3m(
                df3e, bar_time=bt.to_pydatetime())
            if sig == "none":
                sig, stop_px, reason = get_gap_signal_3m(
                    df3e, bar_time=bt.to_pydatetime(), pmh=pmh,
                    support=sup, resistance=res)
        else:
            sig, stop_px, reason = "none", None, ""
        is_gap_entry = sig != "none"
        if sig == "none":
            # cloud_cont_crash is a crash-day starter — allowed before FIRST_ENTRY_MINUTE
            # (same treatment as gap signals).  All other signals wait until 09:40.
            is_crash_starter = False
            if bt.hour == 9 and bt.minute < FIRST_ENTRY_MINUTE:
                # Peek at the signal without committing — only crash starters pass early
                _sig, _stop, _rsn = get_entry_signal_3m(
                    df3e, trend, bar_time=bt.to_pydatetime(),
                    support=sup, resistance=res)
                if _rsn == "cloud_cont_crash":
                    sig, stop_px, reason = _sig, _stop, _rsn
                    is_crash_starter = True
                else:
                    continue   # too early for non-crash signals
            else:
                sig, stop_px, reason = get_entry_signal_3m(
                    df3e, trend, bar_time=bt.to_pydatetime(),
                    support=sup, resistance=res)
        if sig == "none": continue

        # Shares: full intended size; starter enters at STARTER_RATIO of that
        shares_full  = shares   # already set per-symbol above
        shares_start = max(1, int(shares_full * STARTER_RATIO))
        shares_add   = shares_full - shares_start

        entry     = df3e.iloc[-1].close
        stop_dist = abs(entry - stop_px)
        # Risk check uses STARTER shares only (starter is the max we risk initially)
        risk      = stop_dist * shares_start
        risk_cap  = MAX_RISK_DOLLARS_HIGH if entry >= HIGH_PRICE_THRESHOLD else MAX_RISK_DOLLARS
        if risk > risk_cap: continue
        # cloud_cont signals are by definition for stocks already past the key level —
        # applying the proximity filter on top would block every valid continuation entry.
        # Gap entries are also exempt (is_gap_entry).
        is_cont_signal = (reason in ("cloud_cont", "cloud_cont_crash") or
                          reason.startswith("cloud_512_flip"))
        if not is_gap_entry and not is_cont_signal and sig=="long"  and res and entry > res*(1+LEVEL_PROX_LONG):  continue
        if not is_gap_entry and not is_cont_signal and sig=="short" and sup and entry < sup*(1-LEVEL_PROX_SHORT): continue

        candidates.append({
            "sym": sym, "sig": sig, "entry": entry, "entry_t": bt,
            "original_entry": entry,
            "init_stop": stop_px, "stop_cur": stop_px, "dist": stop_dist,
            "reason": reason, "trend": trend,
            "shares": shares_start, "remaining": shares_start,
            "shares_full": shares_full, "add_shares": shares_add,
            "add_triggered": False, "add_entry": None,
            "best_unr": 0.0, "peak_unr": 0.0,
            "half_done": False, "half_pnl": 0.0,
            "atr": sym_atr.get(sym, 0),
        })

    if candidates:
        # Pick highest ATR — biggest mover wins
        best = max(candidates, key=lambda c: c["atr"])
        in_trade   = True
        open_trade = best

# Close anything still open at EOD
if in_trade and open_trade:
    t    = open_trade
    sym  = t["sym"]
    b3, b10, today3, shares, pmh, pml, df1_today = sym_data[sym]
    last = today3[-1]
    exit_px = last.close; fbt = pd.Timestamp(last.date)
    runner  = ((exit_px-t["entry"])*t["remaining"]) if t["sig"]=="long" else ((t["entry"]-exit_px)*t["remaining"])
    total   = t["half_pnl"] + runner
    held_m  = int((fbt - t["entry_t"]).total_seconds()//60)
    all_trades.append({
        "sym": sym, "time": t["entry_t"], "sig": t["sig"].upper(),
        "entry": t.get("original_entry", t["entry"]), "avg_entry": t["entry"],
        "stop": t["init_stop"],
        "dist": t["dist"], "shares": t["shares"], "risk": t["dist"]*t["shares"],
        "reason": t["reason"], "trend": t["trend"],
        "half_pnl": t["half_pnl"], "runner_pnl": runner, "total_pnl": total,
        "exit_px": exit_px, "exit_t": fbt, "exit_why": "eod_close",
        "held": held_m, "peak": t["peak_unr"],
        "add_triggered": t["add_triggered"], "add_entry": t["add_entry"],
        "shares_full": t["shares_full"],
    })

# ── Print ─────────────────────────────────────────────────────────────────────
HDR = (f"  {'#':<2}  {'TIME':<5}  {'SYM':<5}  {'DIR':<5}  {'ENTRY':>7}  "
       f"{'STOP':>7}  {'SH':>4}  {'PEAK/SH':>7}  {'P&L':>7}  "
       f"{'HELD':>4}  {'EXIT':<13}  SIGNAL")
print(HDR)
print(f"  {'-'*115}")

for n, r in enumerate(all_trades, 1):
    sign = "+" if r["total_pnl"]>=0 else "-"
    pk   = f"+${r['peak']:.2f}" if r["peak"]>=0 else f"-${abs(r['peak']):.2f}"
    tag  = "WIN" if r["total_pnl"]>0 else ("SCR" if abs(r["total_pnl"])<20 else "LOS")
    # Shares display: "25+25=50" when add triggered, "25(+25)" when not yet.
    # r["shares"] = starter count (never overwritten by add).
    # r["shares_full"] = full intended position.
    s_start = r["shares"]
    s_full  = r.get("shares_full", s_start)
    s_add   = s_full - s_start
    sh_str  = f"{s_start}+{s_add}={s_full}" if r.get("add_triggered") else f"{s_start}(+{s_add})"
    add_note = ""
    if r.get("add_triggered") and r.get("add_entry"):
        add_note = f"  ADD@${r['add_entry']:.2f} avg=${r.get('avg_entry', r['entry']):.2f}"
    extra = ""
    if r["half_pnl"]:
        rs = "+" if r["runner_pnl"]>=0 else "-"
        extra = (f"  (half +${r['half_pnl']:.0f}  "
                 f"runner {rs}${abs(r['runner_pnl']):.0f})")
    print(
        f"  {n:<2}  {r['time'].strftime('%H:%M')}  {r['sym']:<5}  {r['sig']:<5}  "
        f"${r['entry']:>6.2f}  ${r['stop']:>6.2f}  {sh_str:<10}  "
        f"{pk:>7}  {sign}${abs(r['total_pnl']):>5.0f}  {r['held']:>4}m  "
        f"{r['exit_why']:<13}  [{r['reason']}] 10m={r['trend']}  {tag}{add_note}{extra}"
    )

print(f"  {'-'*112}")
if all_trades:
    total = sum(r["total_pnl"] for r in all_trades)
    wins  = sum(1 for r in all_trades if r["total_pnl"]>0)
    loss  = sum(1 for r in all_trades if r["total_pnl"]<=0)
    print(f"\n  {len(all_trades)} trade(s)   {wins}W / {loss}L   TOTAL P&L: ${total:+.0f}\n")

    # Show what was skipped due to position being occupied
    print("  NOTE: signals below were blocked because bot was already in a trade:")
    occupied_windows = []
    for r in all_trades:
        occupied_windows.append((r["time"], r["exit_t"], r["sym"]))
    # (informational — not computing skipped signals here)
    print("  (run per-symbol version to see all individual symbol opportunities)\n")
else:
    print("  No trades fired.\n")
