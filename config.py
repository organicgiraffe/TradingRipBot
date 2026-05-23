TWS_HOST = "127.0.0.1"
TWS_PORT = 7497        # 7497 = paper trading, 7496 = live
TWS_CLIENT_ID = 1

# Trading hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 40
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 50

# Ripster EMA Cloud periods — source = hl2 = (high+low)/2, matches PineScript exactly
# Cloud 1:  8 / 9    micro-trend  (dark green / dark pink)
# Cloud 2:  5 / 12   fast cloud   (bright green / red)   ← entry confirmation
# Cloud 3: 34 / 50   slow cloud   (blue / orange)        ← trend bias + stop level
# Cloud 5:      200  major trend filter (standalone, from Cloud 5: 180/200)
EMA_PERIODS = [5, 8, 9, 12, 34, 50, 200]

# Minimum bars before signals are trusted
MIN_BARS_10M = 210     # 200 EMA needs ~210 bars of history on 10-min
MIN_BARS_3M  = 60      # 50 EMA stable enough for 3-min entry

# Bar sizes
BAR_SIZE_10M = "10 mins"   # trend direction
BAR_SIZE_3M  = "3 mins"    # entry + trade management

# Quality trade filters
BREAKEVEN_TRIGGER    = 5.0    # once trade is $5 profitable, lock stop at entry
MAX_STOP_PCT         = 0.025  # skip trade if stop is farther than 2.5% of entry price
MAX_STOP_DISTANCE    = 5.00   # legacy dollar cap — now secondary to MAX_STOP_PCT
MAX_TRADES_PER_DAY   = 1      # one clean trade per symbol per day; second trades are consistent losers
VOLUME_CONFIRM_MULT  = 1.2    # signal candle volume must be > 1.2x 20-bar avg
MAX_RISK_PER_TRADE   = 200.0  # max $ at risk per trade — shares are sized dynamically
MIN_SHARES           = 1      # never go below 1 share
MIN_STOP_DIST        = 0.40   # minimum $ stop distance — below this the entry is degenerate
CLOUD_EXIT_BUFFER    = 0.10   # price must close at least $0.10 PAST ema34 to exit
                              # prevents false exits when price just grazes the cloud edge
GAP_THRESHOLD        = 0.020  # ema50 > 2.0% from price = Day1 gap-up/down catalyst play
                              # above this: use ema12 as tighter stop, allow TYPE 1b entry
