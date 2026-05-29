# Trading Bot Session Review — 2026-05-28

## Live trading result
- **Live bot P&L today: −$41** (2 trades, 0W/2L)
- All entries blocked by C3 filter; only AVGO short fired and got chopped twice
- Crash STP race condition caused unintended LONG 100sh AVGO position

## What was fixed today

### Code changes (in order)
| File | Change | Why |
|---|---|---|
| `ibkr_client.py` | `_entry_order` — MKT for paper AND live (was MIDPRICE on live) | Live orders need guaranteed fills |
| `ibkr_client.py` | `_close_position` — cancel crash STP BEFORE placing exit MKT | Race condition created unintended LONG 100 AVGO |
| `ibkr_client.py` | Fallback polling now calls `_refresh_bars()` every 3 min | Stalled `keepUpToDate=True` stream → 6-hour dead window after 11:06 |
| `ibkr_client.py` | `subscribe_bars` — iterate `list(self.symbols)` | Mutating list during iteration silently dropped next symbol |
| `ibkr_client.py` | `subscribe_bars` — handle empty `b10`/`b3`, drop symbol gracefully | Data subscription errors crashed startup |
| `ibkr_client.py` | Main loop — reconnect logic (3 attempts) on disconnect | TWS disconnects no longer kill bot |
| `ibkr_client.py` | Per-symbol try/except on `_on_new_bar_3m`, `_on_new_bar_10m`, `_refresh_bars_1m`, `ib.sleep` | One bad bar can't kill loop anymore |
| `ibkr_client.py` | BLOCKED diagnostic shows real reason (stop pct/distance), not C3 | Diagnostic was lying about C3 still blocking |
| `ema_engine.py` | Removed C3 filter, 10m trend filter, volume filter from 5/12 flip | User: "trade on the C2 flip regardless of 34/50" |
| `ema_engine.py` | Removed $15 hard dollar stop cap from `_stop_ok` | Was silently blocking MU/SNDK/META trades |
| `config.py` | `MIN_STOP_DIST` $0.40 → $0.05 | Was blocking ONDS-style small-cap entries by pennies |
| `config.py` | `MIN_DAILY_RANGE` $5 → $0.30 + new `MIN_DAILY_RANGE_PCT` 2.5% | Range filter now %-based, low-priced stocks no longer auto-blocked |
| `config.py` | `MAX_SIMULTANEOUS_POSITIONS` 1 → 2 | +$3,749/week from 2nd slot vs 1 slot |
| `config.py` | `MAX_STOP_PCT` tested 4% then reverted to 2.5% | User wanted original stop ceiling |
| `week_backtest.py` | DTR exemption now %-based: `vol_pct >= 3.0%` OR ATR ≥ $10 | Was blocking PLTR (4.27% vol_pct) at 5% threshold |

## Backtest results matrix

### Week (5 days, last 5 trading sessions)
| Date | Day | MU/SNDK/HOOD/ZS |
|---|---|---|
| 5/21 Thu | +$1,039 |
| 5/22 Fri | −$936 (Lotto Friday chop) |
| 5/26 Tue | +$1,009 |
| 5/27 Wed | +$3,433 (MU+ZS shorts parallel) |
| 5/28 Thu | +$456 |
| **TOTAL** | **+$5,002 / 5 days = ~$1,000/day avg** |

### Today (5/28) — multiple baskets
| Basket | Result | Notes |
|---|---|---|
| MU/CRM/HOOD/AVGO/LLY/SNDK | **+$559** | User's real pick — MU $718, LLY $432, others mixed |
| ONDS/MU/SNDK/TSLA/MSFT | +$437 | ONDS late entries, MU stars |
| DELL/SNOW/NOW/CRM | +$367 | Even spread, CRM $480 winner |
| MU/SNDK/HOOD/ZS | +$456 | HOOD entered late |

### Yesterday (5/27)
| Symbols | Result |
|---|---|
| COIN/NVDA/UBER/BABA/ABNB/SHOP | **+$336** (SHOP +$240, COIN +$187) |
| MU alone | **+$466** (one $861 winner, one −$395 chase) |

### Solo tests (proving signals fire)
| Symbol | Result | Trade |
|---|---|---|
| HOOD alone (5/28) | **+$734** | $75.35 → $84.35 held EOD |
| PLTR alone (5/28) | +$122 | After DTR threshold lowered to 3% |
| ONDS alone (5/28) | +$0 net | Gap-go-pmh fired at 09:33 (proves signal works) |
| MU alone (5/27) | **+$466** | Workhorse confirmed |

## Key findings

### What works
1. **MU is the bot's edge.** +$1,184 across 5/27 + 5/28 alone. Always include MU.
2. **5/12 flip without filters fires cleanly.** MU short at 09:42 today (+$449) was textbook.
3. **2 slots is the sweet spot.** 1 slot leaves $3,749/week on the table.
4. **R:R 4-9:1.** Win rate is 33-62% across days, but winners are 4-9x bigger than losers.
5. **Strategy is consistent across watchlists.** All 4 baskets tested today were profitable.

### What doesn't work / known limitations
1. **HOOD always loses to higher-ATR stocks.** Slot priority by ATR locks HOOD out when paired with MU/SNDK/LLY. Solo HOOD +$734, in basket $0.
2. **Friday chop kills the day.** −$936 on 5/22. Either skip Fridays or trade half-size.
3. **Bot logs "ENTRY" before fill confirmation.** Misleading when orders cancel (Error 354 today).
4. **Momentum-priority slot ordering broke things.** Tested it, lost $1,190. Reverted.
5. **Cloud_cont 2.5% extension limit** prevents chasing big gap-and-go moves like ONDS.

### Data subscription history today
- 09:21-10:03: Error 10089/10168/354 — subscription not active, orders cancelled
- 10:06+: Subscription registered, AVGO short FILLED at $418.74
- 11:02 restart: warnings persist but orders fill
- **Confirmed working from 10:06 onwards**

## Pre-flight checklist for tomorrow

1. [ ] TWS open and logged in to paper account BEFORE 09:25
2. [ ] Paper trading disclaimer accepted (the popup)
3. [ ] Click any watchlist stock in TWS → confirm LIVE prices update (not "Delayed")
4. [ ] No other clientId=1 active (close Excel RTD, prior bot sessions)
5. [ ] API settings (verified today via screenshot):
   - ✓ Enable ActiveX and Socket Clients
   - ✓ Read-Only API OFF
   - ✓ Socket port 7497
   - ✓ Allow connections from localhost only
6. [ ] Configure → Order Presets — "Automatically Transmit Orders" ON
7. [ ] Restart bot at 09:25 ET

## Current config state (going into tomorrow)

```
MAX_SIMULTANEOUS_POSITIONS = 2
MAX_STOP_PCT               = 0.025   (2.5%)
MIN_STOP_DIST              = $0.05
MIN_DAILY_RANGE            = $0.30   (penny-stock floor)
MIN_DAILY_RANGE_PCT        = 2.5%
DTR_EXEMPT_ATR             = $10  OR  vol_pct ≥ 3.0%
FIRST_ENTRY_MINUTE         = 09:33
LAST_ENTRY_HOUR/MINUTE     = 15:00
MAX_RISK_DOLLARS           = $700  (<$500 stock) / $900 (≥$500)
Orders                     = MKT only (paper + live)
5/12 flip filters          = NONE (no C3, no 10m trend, no volume)
```

## Items to discuss tomorrow

1. **Slot priority refactor** — momentum-based was wrong direction; what about prioritizing by signal type (gap > flip > cont)?
2. **HOOD problem** — keep it, drop it, or tier the watchlist?
3. **Friday strategy** — skip, half-size, or trust the rules?
4. **ENTRY logging** — should wait for fill confirmation before writing to trade log
5. **PLTR threshold** — 3.0% caught it, but barely. Other stocks that need this?
6. **Add-in trigger** — does the +$3/share scale-up actually work in live?

## Tomorrow's suggested watchlist

**Core (must-have):**
- MU (workhorse, +$1,184 across last 2 days)
- SNDK (high ATR, catches big swings)

**Rotation (pick 2-3 from Rip's sheet based on bias):**
- LLY (caught $432 today on cloud_cont)
- CRM (caught $480 today on open_cloud_break)
- AVGO (chopped today but trades clean on directional days)
- COIN (consistent yesterday, $187)
- HOOD (only if catalyst — drop another name if including)

**Don't put in basket together:**
- HOOD + SNDK + MU (HOOD will lose every slot fight)
- 4+ low-priced stocks (slots fight, none get good entries)

## Pending tasks (background)

1. Build Tier 1 bank upgrade/downgrade scanner
2. Build morning HTML parser (auto-extract Rip's sheet)
3. Multi-symbol live bot improvements (slot priority, tiering)
4. Wire Rip bias + levels into live bot morning startup
5. Separate Day1 news plays from Day2/Day3 continuation plays
6. Fix "ENTRY" being logged before fill confirmation
