import asyncio
import logging
import signal
import sys

from ib_insync import util

from ibkr_client import TradingBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_startup_inputs() -> tuple[list[str], int]:
    print("\n============================")
    print("  RIPSTER CLOUD TRADING BOT ")
    print("============================\n")

    raw = input("Enter stock symbols (comma-separated, e.g. MU, NVDA, AMD): ")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]

    if not symbols:
        print("No symbols entered. Exiting.")
        sys.exit(1)

    while True:
        try:
            shares = int(input("Number of shares per trade: "))
            if shares > 0:
                break
            print("Must be greater than 0.")
        except ValueError:
            print("Please enter a whole number.")

    print(f"\nSymbols : {', '.join(symbols)}")
    print(f"Shares  : {shares}")
    confirm = input("Start bot? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        sys.exit(0)

    return symbols, shares


async def main():
    util.startLoop()   # ib_insync event loop integration

    symbols, shares = get_startup_inputs()
    bot = TradingBot(symbols, shares)

    loop = asyncio.get_event_loop()

    def shutdown(sig, frame):
        log.info("Shutdown signal received — closing positions and exiting...")
        if bot.position and bot.position.is_open:
            log.warning(f"Open position will NOT be auto-closed on exit: {bot.position.summary()}")
            log.warning("Close it manually in TWS if needed.")
        bot.print_session_summary()
        bot.disconnect()
        loop.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        await bot.run()
    except Exception as e:
        log.error(f"Bot crashed: {e}", exc_info=True)
        bot.print_session_summary()
        bot.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
