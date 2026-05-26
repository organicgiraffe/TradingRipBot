"""
main.py — launch the Ripster Cloud live trading bot.

Usage:
    python main.py

Pre-requisites:
    1. Trader Workstation (TWS) is open and logged in to PAPER account
    2. TWS API enabled: File -> Global Configuration -> API -> Settings
         - Enable ActiveX and Socket Clients: ON
         - Socket port: 7497
         - Allow connections from localhost only: ON
         - Read-Only API: OFF  (bot needs to place orders)
    3. Python packages: ib_insync, pandas, yfinance
"""
import collections
import logging
import os
import signal
import sys
from datetime import date

import yfinance as yf
import pandas as pd

from config import (FIXED_SHARES, FIXED_SHARES_HIGH, HIGH_PRICE_THRESHOLD,
                    MAX_RISK_DOLLARS, MAX_RISK_DOLLARS_HIGH, MIN_DAILY_RANGE,
                    FIRST_ENTRY_MINUTE, MAX_SIMULTANEOUS_POSITIONS)
from ibkr_client import TradingBot
from post_session_analyzer import analyze_session

# ── Logging: console + bot activity log ───────────────────────────────────
os.makedirs("logs", exist_ok=True)
today_str = date.today().strftime("%Y-%m-%d")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/bot_{today_str}.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def _compute_atr(sym: str) -> float | None:
    """5-day avg daily range via yfinance 1-min bars. Returns None on failure."""
    try:
        df = yf.download(sym, period="5d", interval="1m",
                         auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.index = (df.index.tz_convert("America/New_York")
                    if df.index.tz else df.index.tz_localize("America/New_York"))
        df = df.between_time("09:30", "15:59")
        hi = collections.defaultdict(float)
        lo = collections.defaultdict(lambda: float("inf"))
        for ts, row in df.iterrows():
            d = ts.date()
            hi[d] = max(hi[d], row["high"])
            lo[d] = min(lo[d], row["low"])
        days = sorted(hi)[-5:]
        return round(sum(hi[d] - lo[d] for d in days) / len(days), 2) if days else None
    except Exception:
        return None


def get_startup_inputs() -> tuple[list[str], dict]:
    """Gather today's symbols, auto-filter by ATR, then ask for Rip's levels."""
    print()
    print("=" * 62)
    print("   RIPSTER CLOUD TRADING BOT  —  Paper Trading")
    print("=" * 62)

    raw = input("\nSymbols (comma-separated, e.g. TSLA, NVDA, AMD, MU): ")
    all_syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not all_syms:
        print("No symbols entered. Exiting.")
        sys.exit(1)

    # ── ATR filter ────────────────────────────────────────────────────
    print(f"\n  Checking 5-day ATR (min=${MIN_DAILY_RANGE:.0f}) ...\n")
    passed, dropped = [], []
    atr_map: dict[str, float] = {}
    for sym in all_syms:
        atr = _compute_atr(sym)
        atr_map[sym] = atr or 0.0
        if atr is None:
            print(f"    {sym:<6}  (no data — skipped)")
            dropped.append(sym)
        elif atr < MIN_DAILY_RANGE:
            print(f"    {sym:<6}  ATR=${atr:.2f}  SKIP  (< ${MIN_DAILY_RANGE:.0f})")
            dropped.append(sym)
        else:
            sh = FIXED_SHARES_HIGH if atr_map[sym] > HIGH_PRICE_THRESHOLD else FIXED_SHARES
            print(f"    {sym:<6}  ATR=${atr:.2f}  OK")
            passed.append(sym)

    if not passed:
        print("\n  No symbols passed the ATR filter. Exiting.")
        sys.exit(1)

    # Sort passed symbols by ATR descending — highest movers evaluated first
    passed.sort(key=lambda s: atr_map[s], reverse=True)

    if dropped:
        print(f"\n  Dropped {len(dropped)}: {', '.join(dropped)}")
    print(f"  Active  {len(passed)}: {', '.join(passed)}  (sorted by ATR, highest first)\n")

    # ── Rip's levels — only for filtered symbols ──────────────────────
    print("Enter Rip's levels for each symbol.")
    print("  (press Enter to skip — runs rules-only with no level filter)\n")

    plan: dict = {}
    for sym in passed:
        try:
            sup_s = input(f"  {sym}  support  ($): ").strip()
            res_s = input(f"  {sym}  resistance ($): ").strip()
            plan[sym] = {
                "support":    float(sup_s) if sup_s else None,
                "resistance": float(res_s) if res_s else None,
            }
        except ValueError:
            print(f"  Invalid number for {sym} — defaulting to rules-only.")
            plan[sym] = {"support": None, "resistance": None}

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print("-" * 62)
    print("  TODAY'S PLAN")
    print("-" * 62)
    for sym in passed:
        p    = plan[sym]
        atr  = atr_map[sym]
        sup_s = f"${p['support']:.2f}" if p["support"] else "—"
        res_s = f"${p['resistance']:.2f}" if p["resistance"] else "—"
        risk_cap = MAX_RISK_DOLLARS_HIGH if atr >= HIGH_PRICE_THRESHOLD else MAX_RISK_DOLLARS
        print(f"  {sym:6s}  ATR=${atr:.0f}  sup={sup_s:<10} res={res_s:<10} risk_cap=${risk_cap}")
    print()
    print(f"  Shares   : {FIXED_SHARES} (< ${HIGH_PRICE_THRESHOLD:.0f})  "
          f"/ {FIXED_SHARES_HIGH} (>= ${HIGH_PRICE_THRESHOLD:.0f})")
    print(f"  ATR min  : ${MIN_DAILY_RANGE:.0f}/day")
    print(f"  Entry    : no trades before 09:{FIRST_ENTRY_MINUTE:02d} ET")
    print(f"  Slots    : {MAX_SIMULTANEOUS_POSITIONS} simultaneous position(s)")
    print(f"  Log file : logs/trades_{today_str}.log")
    print("-" * 62)

    confirm = input("\nStart bot? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        sys.exit(0)

    return passed, plan


def main():
    symbols, plan = get_startup_inputs()
    bot = TradingBot(symbols, plan)

    def shutdown(sig, frame):
        log.info("Shutdown signal received — stopping bot...")
        for sym, pos in list(bot.positions.items()):
            if pos.is_open:
                log.warning(
                    f"OPEN POSITION NOT AUTO-CLOSED: {pos.summary()}"
                    f" — close manually in TWS!"
                )
        bot.print_session_summary()
        bot.disconnect()
        analyze_session()   # replay blocked signals, score filters, write report
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        bot.run()   # connects, subscribes, then blocks in ib.run()
    except Exception as e:
        log.error(f"Bot crashed: {e}", exc_info=True)
        bot.print_session_summary()
        bot.disconnect()


if __name__ == "__main__":
    main()
