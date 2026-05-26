TWS_HOST = "127.0.0.1"
TWS_PORT = 7497        # 7497 = paper trading, 7496 = live
TWS_CLIENT_ID = 1

# Trading hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 33   # first 3-min bar closes at 09:33 — take the open move
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
BREAKEVEN_TRIGGER    = 5.0    # once trade is $5 profitable, lock stop at entry (legacy, kept for ref)
RATCHET_START        = 3.0    # once up $3/share, ratchet trailing stop activates
RATCHET_GIVEBACK     = 2.0    # max $ that can be given back once ratchet is active
                               # e.g. up $5 → stop floor = entry + $3 (locks $150 on runner)
                               #      up $7 → stop floor = entry + $5 (locks $250 on runner)
                               # tighter than $3 — works alongside the half-exit at level
MAX_STOP_PCT         = 0.025  # skip trade if stop is farther than 2.5% of entry price
MAX_STOP_DISTANCE    = 5.00   # legacy dollar cap — now secondary to MAX_STOP_PCT
MAX_TRADES_PER_DAY   = 1      # one clean trade per symbol per day; second trades are consistent losers
VOLUME_CONFIRM_MULT  = 0.8    # signal candle volume must be > 0.8x 20-bar avg (80% of average)
RVOL_EXIT_MULT       = 0.3    # exit when 3m bar volume drops below 30% of avg — truly dead
MAX_RISK_PER_TRADE   = 200.0  # max $ at risk per trade — shares are sized dynamically
MIN_SHARES           = 1      # never go below 1 share

# ── $500/day target parameters ────────────────────────────────────────────────
FIXED_SHARES         = 100    # flat 100 shares per position (stocks under HIGH_PRICE_THRESHOLD)
FIXED_SHARES_HIGH    = 50     # 50 shares for expensive stocks ($500+)
HIGH_PRICE_THRESHOLD = 500.0  # entry price at or above this → use FIXED_SHARES_HIGH
                               # catches META ~$610, CRWD ~$565, MU ~$800, SNDK ~$1000+
PROFIT_TARGET_SHARE  = 5.00   # exit at +$5/share = $500 profit on 100 shares
MIN_DAILY_RANGE      = 7.00   # skip stock if 5-day avg daily range < $7
                               # TSLA/AMD/META/CRWD pass; AAPL/AMZN/NFLX typically fail
MIN_STOP_DIST        = 0.40   # absolute floor (legacy) — superseded by MIN_STOP_PCT_LOWER below
LARGE_CANDLE_STOP    = 5.00   # prev candle range > this → use entry ± range/2 as stop
                               # prevents oversized risk on gap/news bars ($5 threshold)
MIN_STOP_PCT_LOWER   = 0.0025 # 0.25% of price — stops tighter than this are noise, not signal
                              # META $605 + $0.68 stop = 0.11% → blocked
                              # AMZN $264 + $0.97 stop = 0.37% → allowed
CLOUD_EXIT_BUFFER    = 0.10   # price must close at least $0.10 PAST ema34 to exit
                              # prevents false exits when price just grazes the cloud edge
GAP_THRESHOLD        = 0.020  # ema50 > 2.0% from price = Day1 gap-up/down catalyst play
                              # above this: use ema12 as tighter stop, allow TYPE 1b entry
FIRST_ENTRY_MINUTE   = 40     # no entries before 09:40 — skip the first 5-min chaos
                               # 9:40-9:44 setups are valid; $700 risk cap handles disasters
LAST_ENTRY_HOUR      = 15     # no new entries at or after this hour (ET)
LAST_ENTRY_MINUTE    = 0      # → 15:00  gives at least 50 min for trade to develop
FRIDAY_OPEN_MINUTE   = 45     # Lotto Friday: hold extra 5 min — first entry 09:45
MAX_RISK_DOLLARS     = 700    # skip any trade where stop_dist × shares > $700
                               # TSLA normal risk ~$550 → allowed
                               # MU Mar-31 open trade $716 → blocked (plus time filter)
MAX_SIMULTANEOUS_POSITIONS = 1   # one quality trade at a time — two positions cancel each other out

# Level proximity — entry must be NEAR Rip's level, not chasing mid-range
# Long:  entry must be below resistance + 1.5%  (at support or fresh breakout only)
# Short: entry must be above support - 2.0%     (at resistance or fresh breakdown only)
LEVEL_PROX_LONG  = 0.015   # 1.5% above resistance = max allowed for a long entry
LEVEL_PROX_SHORT = 0.020   # 2.0% below support    = max allowed for a short entry

# DTR / ATR range filter (from Rip's sheet — e.g. "DTR: 6.21 vs ATR: 7.59  82%")
# Don't enter when today's range is already mostly spent.
# Example: if ATR = $7.59 and DTR so far = $6.21, ratio = 82% — skip the trade.
ATR_PERIODS  = 14     # trading days for ATR lookback
DTR_MAX_PCT  = 0.75   # skip entry when DTR >= 75% of ATR (move is largely done)

# ── Debug output ──────────────────────────────────────────────────────────────
DEBUG_SIGNALS = True   # print a status line every 3m bar + BLOCKED reasons
