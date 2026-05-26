"""
post_session_analyzer.py

Runs automatically at session end (called from main.py shutdown).
For each day it:
  1. Parses trades_YYYY-MM-DD.log  (entries, exits, blocked, skips)
  2. Fetches today's 3-min bars from yfinance
  3. Replays every BLOCKED/SKIP signal — was the filter right or wrong?
  4. Scores each filter by money saved vs money cost
  5. Writes logs/analysis_YYYY-MM-DD.json

Claude reads this file automatically at the start of every session
and presents specific parameter suggestions before trading begins.
"""

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# How many bars forward to evaluate a blocked signal
EVAL_BARS = 10
# Approximate bot data lag in minutes (IBKR paper ~15-18 min)
DATA_LAG_MIN = 18

LOG_DIR = Path("logs")


# ── Log parsing ────────────────────────────────────────────────────────────

def _f(s: str) -> float:
    """Parse a float from a string that may contain $, +, spaces."""
    return float(re.sub(r"[^0-9.\-]", "", s))


def parse_trade_log(log_path: Path) -> dict:
    entries, exits, blocked, skips = [], [], [], []

    # Patterns — all anchored to the tlog format: "HH:MM:SS  MESSAGE"
    p_entry = re.compile(
        r"(\d{2}:\d{2}:\d{2})\s+ENTRY\s+\[\d+/\d+\]\s+(\w+)\s+(\w+)"
        r"\s+x(\d+)sh\s+\$([0-9.]+)\s+stop=\$([0-9.]+)"
        r".*?\[([^\]]+)\].*?trend=(\S+)"
    )
    p_exit = re.compile(
        r"(\d{2}:\d{2}:\d{2})\s+EXIT\s+(\w+)\s+(\w+)\s+(\w+)"
        r"\s+x(\d+)sh\s+\$([0-9.]+)->\$([0-9.]+)"
        r"\s+pnl=([+\-$0-9.]+)\s+\[([^\]]+)\]\s+held=(\d+)min\s+signal=(\S+)"
    )
    p_blocked = re.compile(
        r"(\d{2}:\d{2}:\d{2})\s+BLOCKED\s+(\w+)\s+(\w+)\s+\[([^\]]+)\]"
        r"(?:\s+px=\$([0-9.]+))?(?:\s+e50=\$([0-9.]+))?"
    )
    p_skip = re.compile(
        r"(\d{2}:\d{2}:\d{2})\s+SKIP\s+(\w+)\s+(\w+)\s+\$([0-9.]+)"
        r".*?(?:risk=\$([0-9.]+).*?\[([^\]]+)\]|DTR\s+([0-9]+)%)"
    )

    with open(log_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            m = p_entry.search(line)
            if m:
                entries.append({
                    "time": m.group(1), "direction": m.group(2).lower(),
                    "symbol": m.group(3), "shares": int(m.group(4)),
                    "entry_price": float(m.group(5)), "stop_price": float(m.group(6)),
                    "signal": m.group(7), "trend": m.group(8),
                })
                continue

            m = p_exit.search(line)
            if m:
                exits.append({
                    "time": m.group(1), "result": m.group(2),
                    "direction": m.group(3).lower(), "symbol": m.group(4),
                    "shares": int(m.group(5)),
                    "entry_price": float(m.group(6)), "exit_price": float(m.group(7)),
                    "pnl": _f(m.group(8)), "reason": m.group(9),
                    "held_min": int(m.group(10)), "signal": m.group(11),
                })
                continue

            m = p_blocked.search(line)
            if m:
                reason = m.group(4)
                # Classify the filter that caused the block
                if "vol" in reason.lower():
                    filter_type = "volume"
                elif "C3" in reason:
                    filter_type = "c3_direction"
                elif "trend" in reason.lower():
                    filter_type = "trend"
                else:
                    filter_type = "other"
                blocked.append({
                    "time": m.group(1), "symbol": m.group(2),
                    "direction": m.group(3).lower(), "reason": reason,
                    "filter_type": filter_type,
                    "px": float(m.group(5)) if m.group(5) else None,
                    "e50": float(m.group(6)) if m.group(6) else None,
                })
                continue

            m = p_skip.search(line)
            if m:
                skip_reason = "dtr" if m.group(7) else "risk_cap"
                skips.append({
                    "time": m.group(1), "symbol": m.group(2),
                    "direction": m.group(3).lower(),
                    "price": float(m.group(4)),
                    "reason": skip_reason,
                })

    return {"entries": entries, "exits": exits, "blocked": blocked, "skips": skips}


# ── yfinance helpers ───────────────────────────────────────────────────────

def fetch_bars(symbol: str, target_date: date) -> pd.DataFrame:
    """3-min bars for target_date, Eastern time index."""
    if not HAS_YFINANCE:
        return pd.DataFrame()
    try:
        tk  = yf.Ticker(symbol)
        df  = tk.history(period="5d", interval="3m")
        if df.empty:
            return df
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("US/Eastern")
        else:
            df.index = df.index.tz_convert("US/Eastern")
        return df[df.index.date == target_date].copy()
    except Exception as e:
        print(f"  yfinance error for {symbol}: {e}")
        return pd.DataFrame()


def bars_after(df: pd.DataFrame, wall_time_str: str,
               target_date: date, n: int = EVAL_BARS) -> pd.DataFrame:
    """
    Given a wall-clock log time (HH:MM:SS), subtract the data lag to
    approximate the signal bar time, then return the next n bars.
    """
    if df.empty:
        return pd.DataFrame()
    try:
        wall = datetime.strptime(wall_time_str, "%H:%M:%S").replace(
            year=target_date.year, month=target_date.month, day=target_date.day
        )
        signal_approx = wall - timedelta(minutes=DATA_LAG_MIN)
        future = df[df.index >= pd.Timestamp(signal_approx, tz="US/Eastern")]
        return future.iloc[:n]
    except Exception:
        return pd.DataFrame()


# ── Signal replay ──────────────────────────────────────────────────────────

def evaluate_blocked(event: dict, df: pd.DataFrame,
                     target_date: date) -> dict:
    """
    Replay a blocked signal and decide if the filter was correct.

    Returns a dict with:
      verdict  — 'FILTER_CORRECT' | 'FILTER_WRONG' | 'INCONCLUSIVE'
      max_gain — best price move in the right direction (per share)
      max_loss — worst price move against the position (per share)
      missed_pnl — estimated $ missed (shares × max_gain) when FILTER_WRONG
    """
    px  = event.get("px")
    e50 = event.get("e50")
    direction = event["direction"]

    if px is None or df.empty:
        return {"verdict": "NO_DATA", "max_gain": 0, "max_loss": 0, "missed_pnl": 0}

    stop_dist = abs(px - e50) if e50 else px * 0.01  # fallback 1%
    future = bars_after(df, event["time"], target_date)
    if future.empty:
        return {"verdict": "NO_DATA", "max_gain": 0, "max_loss": 0, "missed_pnl": 0}

    highs = future["High"].values
    lows  = future["Low"].values

    if direction == "long":
        max_gain = float(max(highs) - px)
        max_loss = float(px - min(lows))
    else:
        max_gain = float(px - min(lows))
        max_loss = float(max(highs) - px)

    # Would stop have been hit before profit target?
    if max_loss >= stop_dist and max_gain < stop_dist:
        verdict = "FILTER_CORRECT"    # blocking saved a loss
        missed_pnl = 0.0
    elif max_gain >= stop_dist * 2 and max_loss < stop_dist:
        verdict = "FILTER_WRONG"      # blocking cost a winner
        missed_pnl = round(max_gain * 100, 0)  # 100 shares
    else:
        verdict = "INCONCLUSIVE"
        missed_pnl = 0.0

    return {
        "verdict":    verdict,
        "max_gain":   round(max_gain, 2),
        "max_loss":   round(max_loss, 2),
        "missed_pnl": missed_pnl,
    }


# ── Filter scoring ─────────────────────────────────────────────────────────

def score_filters(blocked: list, skips: list) -> dict:
    """
    Tally correct vs wrong blocks by filter type.
    Returns a dict used by Claude to decide which filters to tighten or loosen.
    """
    tally = {}
    for ev in blocked:
        ft = ev.get("filter_type", "other")
        analysis = ev.get("analysis", {})
        verdict  = analysis.get("verdict", "NO_DATA")
        if ft not in tally:
            tally[ft] = {"correct": 0, "wrong": 0, "no_data": 0,
                          "missed_pnl": 0.0, "saved_pnl": 0.0}
        if verdict == "FILTER_CORRECT":
            tally[ft]["correct"]   += 1
        elif verdict == "FILTER_WRONG":
            tally[ft]["wrong"]     += 1
            tally[ft]["missed_pnl"] += analysis.get("missed_pnl", 0)
        else:
            tally[ft]["no_data"]   += 1

    # DTR/risk skips — harder to evaluate without full replay, flag for info
    dtr_skips  = sum(1 for s in skips if s["reason"] == "dtr")
    risk_skips = sum(1 for s in skips if s["reason"] == "risk_cap")
    tally["dtr_skip"]      = {"count": dtr_skips}
    tally["risk_cap_skip"] = {"count": risk_skips}

    return tally


# ── Suggestions ────────────────────────────────────────────────────────────

def _load_config_value(key: str) -> str:
    """Read current value of a config variable directly from config.py."""
    try:
        cfg = Path("config.py").read_text(encoding="utf-8")
        m = re.search(rf"^{key}\s*=\s*([^\s#]+)", cfg, re.MULTILINE)
        return m.group(1) if m else "?"
    except Exception:
        return "?"


def generate_suggestions(events: dict, filter_scores: dict) -> list[str]:
    """
    Rule-based parameter suggestions Claude uses as a starting point.
    Claude then applies additional reasoning when reading this report.
    """
    suggestions = []
    vol_mult = _load_config_value("VOLUME_CONFIRM_MULT")
    dtr_pct  = _load_config_value("DTR_MAX_PCT")

    # Volume filter analysis
    vol = filter_scores.get("volume", {})
    if vol.get("wrong", 0) > vol.get("correct", 0):
        suggestions.append(
            f"LOWER VOLUME_CONFIRM_MULT: blocked {vol['wrong']} winners vs "
            f"{vol.get('correct',0)} losers (missed ~${vol.get('missed_pnl',0):.0f}). "
            f"Current={vol_mult}. Try {max(0.5, float(vol_mult) - 0.1):.1f}."
        )
    elif vol.get("correct", 0) > vol.get("wrong", 0) * 2:
        suggestions.append(
            f"KEEP VOLUME_CONFIRM_MULT at {vol_mult} — correctly blocked "
            f"{vol.get('correct',0)} losers today."
        )

    # C3 direction filter — should almost always be correct (don't fight C3)
    c3 = filter_scores.get("c3_direction", {})
    if c3.get("wrong", 0) > 0:
        suggestions.append(
            f"REVIEW C3 FILTER: {c3['wrong']} 'wrong' blocks where C3 mismatched. "
            f"These may need manual review — check if C3 flipped right after the block."
        )

    # DTR skips
    dtr = filter_scores.get("dtr_skip", {})
    if dtr.get("count", 0) >= 3:
        suggestions.append(
            f"HIGH DTR SKIP COUNT ({dtr['count']} today). "
            f"Current DTR_MAX_PCT={dtr_pct}. If most of these were valid setups, "
            f"consider raising to {min(0.85, float(dtr_pct) + 0.05):.0%}."
        )

    # Win rate
    exits = events.get("exits", [])
    if exits:
        wins     = sum(1 for e in exits if e["pnl"] > 0)
        win_rate = wins / len(exits)
        total    = sum(e["pnl"] for e in exits)
        if win_rate < 0.40:
            suggestions.append(
                f"LOW WIN RATE ({win_rate:.0%}, {wins}/{len(exits)} wins, ${total:+.0f}). "
                f"Entries may be too aggressive — review signal types in trades."
            )
        elif win_rate > 0.65:
            suggestions.append(
                f"STRONG WIN RATE ({win_rate:.0%}) — strategy working well today. "
                f"No filter changes recommended based on today alone."
            )

    # Signal type breakdown
    sig_stats = {}
    for e in exits:
        sig = e.get("signal", "unknown")
        if sig not in sig_stats:
            sig_stats[sig] = {"wins": 0, "losses": 0, "pnl": 0}
        if e["pnl"] > 0:
            sig_stats[sig]["wins"] += 1
        else:
            sig_stats[sig]["losses"] += 1
        sig_stats[sig]["pnl"] += e["pnl"]
    for sig, s in sig_stats.items():
        total_sig = s["wins"] + s["losses"]
        if total_sig >= 2:
            wr = s["wins"] / total_sig
            suggestions.append(
                f"SIGNAL [{sig}]: {s['wins']}W/{s['losses']}L  "
                f"win={wr:.0%}  pnl=${s['pnl']:+.0f}"
            )

    if not suggestions:
        suggestions.append("Not enough data today for confident suggestions. "
                           "Keep current params and gather more trades.")
    return suggestions


# ── Main ───────────────────────────────────────────────────────────────────

def analyze_session(target_date: date = None) -> dict:
    if target_date is None:
        target_date = date.today()

    date_str    = target_date.strftime("%Y-%m-%d")
    log_path    = LOG_DIR / f"trades_{date_str}.log"
    report_path = LOG_DIR / f"analysis_{date_str}.json"

    if not log_path.exists():
        print(f"\n[analyzer] No trade log for {date_str} — skipping.")
        return {}

    print(f"\n{'='*60}")
    print(f"  POST-SESSION ANALYSIS  {date_str}")
    print(f"{'='*60}")

    events = parse_trade_log(log_path)

    # Fetch bar data for every symbol that appeared today
    symbols = set()
    for lst in (events["entries"], events["blocked"], events["skips"]):
        for ev in lst:
            symbols.add(ev["symbol"])

    print(f"  Fetching intraday bars for: {', '.join(sorted(symbols))}")
    bars = {sym: fetch_bars(sym, target_date) for sym in symbols}

    # Replay blocked signals
    for ev in events["blocked"]:
        ev["analysis"] = evaluate_blocked(ev, bars.get(ev["symbol"], pd.DataFrame()), target_date)

    filter_scores = score_filters(events["blocked"], events["skips"])
    suggestions   = generate_suggestions(events, filter_scores)

    # ── Print summary ─────────────────────────────────────────────────
    entries = events["entries"]
    exits   = events["exits"]
    blocked = events["blocked"]
    skips   = events["skips"]

    total_pnl = sum(e["pnl"] for e in exits)
    wins      = sum(1 for e in exits if e["pnl"] > 0)

    print(f"\n  TRADES:  {len(exits)} exits  {wins}W/{len(exits)-wins}L  "
          f"total=${total_pnl:+.0f}")

    print(f"\n  BLOCKED SIGNALS ({len(blocked)}):")
    for ev in blocked:
        a = ev.get("analysis", {})
        verdict = a.get("verdict", "NO_DATA")
        tag = {"FILTER_WRONG": "❌ WRONG", "FILTER_CORRECT": "✓ CORRECT",
               "INCONCLUSIVE": "? UNCLEAR", "NO_DATA": "— NO DATA"}.get(verdict, verdict)
        print(f"    {ev['time']}  {ev['symbol']:6s} {ev['direction'].upper():5s} "
              f"[{ev['filter_type']}]  {tag}  "
              f"gain={a.get('max_gain',0):+.2f}  loss={a.get('max_loss',0):.2f}")

    print(f"\n  SKIPPED ({len(skips)}):")
    for ev in skips:
        print(f"    {ev['time']}  {ev['symbol']:6s}  [{ev['reason']}]")

    print(f"\n  SUGGESTIONS FOR TOMORROW:")
    for i, s in enumerate(suggestions, 1):
        print(f"    {i}. {s}")

    print(f"\n  Report: {report_path}")
    print("=" * 60 + "\n")

    # ── Save JSON report ───────────────────────────────────────────────
    report = {
        "date":           date_str,
        "params": {
            "VOLUME_CONFIRM_MULT": _load_config_value("VOLUME_CONFIRM_MULT"),
            "DTR_MAX_PCT":         _load_config_value("DTR_MAX_PCT"),
            "MAX_STOP_PCT":        _load_config_value("MAX_STOP_PCT"),
            "RATCHET_START":       _load_config_value("RATCHET_START"),
            "RATCHET_GIVEBACK":    _load_config_value("RATCHET_GIVEBACK"),
            "FIRST_ENTRY_MINUTE":  _load_config_value("FIRST_ENTRY_MINUTE"),
        },
        "summary": {
            "total_trades": len(exits),
            "wins": wins,
            "losses": len(exits) - wins,
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(wins / len(exits), 3) if exits else 0,
        },
        "trades":         exits,
        "entries":        entries,
        "blocked":        blocked,
        "skips":          skips,
        "filter_scores":  filter_scores,
        "suggestions":    suggestions,
    }

    LOG_DIR.mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report


if __name__ == "__main__":
    import sys
    d = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    analyze_session(d)
