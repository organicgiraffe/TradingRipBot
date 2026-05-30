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
# ── ATR-scaled ratchet ──────────────────────────────────────────────────
# Fixed $3/$2 are right for ~$150 stocks but absurd on $1000+ names (a $2
# giveback on SNDK @ $1677 is 0.12% — inside the noise → instant stop-out).
# When an ATR is supplied to compute_trailing_stop, the start/giveback become
#   max(fixed, mult × ATR)  — so small stocks keep current behaviour and only
# high-volatility names get the room they need.  ATR = 5-day avg daily range.
ATR_START_MULT       = 0.40   # ratchet activates after 0.40 × ATR of profit
ATR_GIVE_MULT        = 0.30   # lock stop within 0.30 × ATR of best price
MAX_STOP_PCT         = 0.025  # skip trade if stop is farther than 2.5% of entry price
MAX_STOP_DISTANCE    = 5.00   # legacy dollar cap — now secondary to MAX_STOP_PCT
MAX_TRADES_PER_DAY   = 2      # up to 2 entries per symbol per day — allows re-entry on a fresh setup
VOLUME_CONFIRM_MULT  = 0.8    # signal candle volume must be > 0.8x 20-bar avg (80% of average)
RVOL_EXIT_MULT       = 0.3    # exit when 3m bar volume drops below 30% of avg — truly dead
MAX_RISK_PER_TRADE   = 200.0  # max $ at risk per trade — shares are sized dynamically
MIN_SHARES           = 1      # never go below 1 share

# ── $500/day target parameters ────────────────────────────────────────────────
FIXED_SHARES         = 100    # full position size (stocks under HIGH_PRICE_THRESHOLD)
FIXED_SHARES_HIGH    = 50     # full position size for expensive stocks ($500+)
HIGH_PRICE_THRESHOLD = 500.0  # entry price at or above this → use FIXED_SHARES_HIGH
                               # catches META ~$610, CRWD ~$565, MU ~$800, SNDK ~$1000+
# ── Starter + add-in (pyramiding) ────────────────────────────────────────────
STARTER_RATIO        = 0.50   # first entry = 50% of full intended shares
                               # MU (50sh full) → 25 starter;  others (100sh) → 50 starter
ADD_TRIGGER_PROFIT   = 3.00   # add remaining shares when starter is up $3+/share
                               # matches RATCHET_START — trade has proven itself before we press it
# ─────────────────────────────────────────────────────────────────────────────
PROFIT_TARGET_SHARE  = 5.00   # exit at +$5/share = $500 profit on 100 shares
MIN_DAILY_RANGE      = 0.30   # absolute-dollar floor — penny-stock guard only ($0.30 min ATR)
                               # was $5/$10 — blocked ONDS-style movers (low price, huge %).
                               # Real filter is now percentage-based below.
MIN_DAILY_RANGE_PCT  = 0.025  # 2.5% — 5-day avg range must be ≥ 2.5% of stock price
                               # ONDS $12 with $1 ATR = 8.3% → allowed
                               # MU $900 with $76 ATR = 8.4% → allowed
                               # MSFT $420 with $11 ATR = 2.6% → borderline allowed
                               # AAPL $300 with $4 ATR = 1.3% → blocked (no intraday range)
MIN_STOP_DIST        = 0.05   # absolute floor — pennies only.  MIN_STOP_PCT_LOWER (0.25%) is the real filter.
                               # was $0.40 — blocked ONDS 09:36 flip with $0.399 stop dist by 0.1 cent.
MIN_STOP_PCT_LOWER   = 0.0025 # 0.25% of price — stops tighter than this are noise, not signal
                              # META $605 + $0.68 stop = 0.11% → blocked
                              # AMZN $264 + $0.97 stop = 0.37% → allowed
CLOUD_EXIT_BUFFER    = 0.10   # price must close at least $0.10 PAST ema34 to exit
CLOUD_CONT_MAX_DIST  = 0.025  # cloud_cont: max distance price can be from ema12
                               # before the signal is considered too extended to enter
                               # 2.5%: MU at $960 with ema12=$985 → 2.6% → blocked
                               #        MU at $975 with ema12=$985 → 1.0% → allowed
                              # prevents false exits when price just grazes the cloud edge
GAP_THRESHOLD        = 0.020  # ema50 > 2.0% from price = Day1 gap-up/down catalyst play
                              # above this: use ema12 as tighter stop, allow TYPE 1b entry
FIRST_ENTRY_MINUTE   = 33     # first completed 3-min bar — trade the 5/12 cloud flip from the open
                               # 9:40-9:44 setups are valid; $700 risk cap handles disasters
GAP_ENTRY_START_HOUR   = 9     # Gap & Go / Gap & Crap opening-drive window
GAP_ENTRY_START_MINUTE = 33    # first completed 3-min bar
GAP_ENTRY_END_HOUR     = 10
GAP_ENTRY_END_MINUTE   = 0     # inclusive: allow the 10:00 bar close
OPEN_CLOUD_BREAK_BODY_PCT = 0.004  # opening cloud break: candle body >= 0.4%
OPEN_CLOUD_BREAK_RANGE_PCT = 0.008 # opening cloud break: full range >= 0.8%
OPEN_CLOUD_BREAK_VOL_MULT = 1.2    # opening cloud break: volume expansion vs 20-bar avg
LAST_ENTRY_HOUR      = 15     # no new entries at or after this hour (ET)
LAST_ENTRY_MINUTE    = 0      # → 15:00  gives at least 50 min for trade to develop
FRIDAY_OPEN_MINUTE   = 33     # keep the same 5/12 cloud flip start on Fridays
MAX_RISK_DOLLARS     = 700    # skip any trade where stop_dist × shares > $700
                               # TSLA normal risk ~$550 → allowed
                               # MU Mar-31 open trade $716 → blocked (plus time filter)
MAX_RISK_DOLLARS_HIGH = 900   # separate cap for stocks >= HIGH_PRICE_THRESHOLD ($500+)
                               # MU $886 entry, stop $872 → $14.65 × 50sh = $732 → allowed
                               # prevents blocking big movers just because their $ stop is wide
MAX_SIMULTANEOUS_POSITIONS = 2   # allow 2 concurrent trades — second slot catches HOOD/ZS while MU/SNDK runs

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
DTR_EXEMPT_ATR = 5.0  # stocks with 5-day ATR above this skip the DTR filter entirely
                       # matched to MIN_DAILY_RANGE so any stock passing the ATR minimum
                       # is also DTR-exempt — on catalyst/news days stocks routinely run
                       # 2x-3x their average range; DTR filter was blocking the best trades
                       # (was 30.0 — only SNDK/MU exempt; AMD/META/TSLA/AVGO were blocked)

# ── Paper account data delay ──────────────────────────────────────────────────
# IBKR paper accounts serve market data 15 minutes delayed.
# Setting this to 15 shifts all entry-gate and signal-window comparisons back
# so they align with the actual bar times we're seeing.
# On live accounts (port 7496) this value is ignored — no offset applied.
PAPER_DATA_DELAY_MINUTES = 0    # set to 15 if paper account loses real-time data sharing

# ── Debug output ──────────────────────────────────────────────────────────────
DEBUG_SIGNALS = True   # print a status line every 3m bar + BLOCKED reasons
