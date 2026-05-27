"""
add_symbol.py — hot-add a symbol to the running bot mid-session.

Usage:
    python add_symbol.py NVDA
    python add_symbol.py NVDA 120.00 125.00

Arguments:
    SYMBOL      ticker to add (required)
    support     Rip's support level  (optional — omit for rules-only)
    resistance  Rip's resistance level (optional — omit for rules-only)

The bot picks it up within one second and loads historical bars,
subscribes real-time data, and starts watching for signals immediately.
ATR filter still applies — symbols below $10/day are skipped.
"""
import json
import sys

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

sym = sys.argv[1].upper().strip()
sup = float(sys.argv[2]) if len(sys.argv) > 2 else None
res = float(sys.argv[3]) if len(sys.argv) > 3 else None

payload = {"symbol": sym, "support": sup, "resistance": res}

with open("hot_add.json", "w") as f:
    json.dump(payload, f)

sup_s = f"${sup:.2f}" if sup else "—"
res_s = f"${res:.2f}" if res else "—"
print(f"Queued {sym}  sup={sup_s}  res={res_s}")
print("Bot picks it up within the next second.")
