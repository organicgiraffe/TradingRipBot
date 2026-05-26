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
import asyncio
import logging
import os
import signal
import sys
from datetime import date

from ib_insync import util

from config import (FIXED_SHARES, FIXED_SHARES_HIGH, HIGH_PRICE_THRESHOLD,
                    MAX_RISK_DOLLARS, FIRST_ENTRY_MINUTE, MAX_SIMULTANEOUS_POSITIONS)
from ibkr_client import TradingBot

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


def get_startup_inputs() -> tuple[list[str], dict]:
    """Gather today's symbols and Rip's support/resistance levels."""
    print()
    print("=" * 58)
    print("   RIPSTER CLOUD TRADING BOT  —  Paper Trading")
    print("=" * 58)

    raw = input("\nSymbols (comma-separated, e.g. TSLA, NVDA, AMD): ")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not symbols:
        print("No symbols entered. Exiting.")
        sys.exit(1)

    print(f"\nEnter Rip's levels for each symbol.")
    print("  (press Enter to skip — symbol runs rules-only with no level filter)\n")

    plan: dict = {}
    for sym in symbols:
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
    print("-" * 58)
    print("  TODAY'S PLAN")
    print("-" * 58)
    for sym in symbols:
        p = plan[sym]
        sup_s = f"${p['support']:.2f}" if p["support"] else "—"
        res_s = f"${p['resistance']:.2f}" if p["resistance"] else "—"
        print(f"  {sym:6s}  support={sup_s:<10}  resistance={res_s}")
    print()
    print(f"  Shares   : {FIXED_SHARES} (< ${HIGH_PRICE_THRESHOLD:.0f})  "
          f"/ {FIXED_SHARES_HIGH} (>= ${HIGH_PRICE_THRESHOLD:.0f})")
    print(f"  Risk cap : ${MAX_RISK_DOLLARS}/trade")
    print(f"  Entry    : no trades before 09:{FIRST_ENTRY_MINUTE:02d} ET")
    print(f"  Slots    : {MAX_SIMULTANEOUS_POSITIONS} simultaneous position(s)")
    print(f"  Log file : logs/trades_{today_str}.log")
    print("-" * 58)

    confirm = input("\nStart bot? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        sys.exit(0)

    return symbols, plan


async def main():
    util.startLoop()   # ib_insync event loop integration

    symbols, plan = get_startup_inputs()
    bot = TradingBot(symbols, plan)

    loop = asyncio.get_event_loop()

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
        loop.stop()

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        await bot.run()
    except Exception as e:
        log.error(f"Bot crashed: {e}", exc_info=True)
        bot.print_session_summary()
        bot.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
