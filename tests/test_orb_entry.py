"""
test_orb_entry.py — scenario tests for the PRODUCTION ORB entry path
(`TradingBot._try_orb_entry` / `_orb_range_from_bars` in ibkr_client.py) and
the backtest ORB branch (`today_backtest.run_multi_today(..., orb_entry=True)`).

The strategy ORB is a STOP-STYLE breakout, NOT candle-close confirmation:
  * OR  = the EXACT 09:30:00-09:40:00 range, from 1-min bars only
  * long trigger = OR_high + 0.01,  short trigger = OR_low - 0.01
  * entry/risk use the TRIGGER price, never the breakout candle's close

These tests assert that exact contract.  In particular `test_orb_long_*` and
`test_orb_short_*` FAIL on the old code (which used `cur.close`) and PASS on the
fixed code (which uses the trigger).

Run:  python tests/test_orb_entry.py
"""
from __future__ import annotations

import sys
import pathlib
import contextlib
import io
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tests.mock_ib import MockIB, MockTicker, MockContract, FillScript
from ibkr_client import TradingBot

DAY = datetime(2026, 6, 1)
TODAY = DAY.date()


@dataclass
class Bar:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 100_000


def make_bot(symbols=None):
    symbols = symbols or ["NVDA"]
    plan = {s: {"support": None, "resistance": None} for s in symbols}
    bot = TradingBot(symbols, plan, ib=MockIB())
    bot.orb_enabled = True
    return bot, bot.ib


def seed_range(bot, symbol, high, low):
    """Pre-populate the cached OR so _try_orb_entry skips the 1-min fetch
    (which would hit reqHistoricalData → MockIB returns [])."""
    bot._orb_state[symbol] = {"date": TODAY, "high": high, "low": low, "fired": False}


def set_live(bot, symbol, px):
    bot.tickers[symbol] = MockTicker(contract=MockContract(symbol), last=px)


def cur_bar(high, low, close, open_=None):
    return pd.Series({"high": high, "low": low, "close": close,
                      "open": open_ if open_ is not None else close})


def spy_open(bot):
    """Replace _open_position with a recorder so we can inspect the entry
    estimate the strategy passes (must be the trigger, not the close)."""
    rec = {}
    def _spy(symbol, direction, entry_price, stop_price, shares, time, **kw):
        rec.update(symbol=symbol, direction=direction, entry_price=entry_price,
                   stop_price=stop_price, shares=shares, kw=kw)
    bot._open_position = _spy
    return rec


def _now(minute):
    return DAY.replace(hour=9, minute=minute)


# ---------------------------------------------------------------------------
# 1 — LONG entry estimate is the TRIGGER (OR_high+0.01), not the candle close
# ---------------------------------------------------------------------------
def test_orb_long_entry_is_trigger_not_close():
    bot, ib = make_bot(["NVDA"])
    seed_range(bot, "NVDA", high=101.00, low=99.00)   # range 2.00
    set_live(bot, "NVDA", 101.01)
    rec = spy_open(bot)
    # Breakout candle pierces the trigger but CLOSES at 101.30 (the old-bug value)
    cur = cur_bar(high=101.50, low=100.40, close=101.30)

    fired = bot._try_orb_entry("NVDA", cur, _now(42), _now(42))

    assert fired is True
    assert rec["direction"] == "long"
    assert abs(rec["entry_price"] - 101.01) < 1e-9, (
        f"entry must be the trigger 101.01, got {rec['entry_price']} "
        f"(101.30 means it used the candle close — the bug)")
    assert rec["entry_price"] != 101.30
    # third_from_break stop: 101.00 - 2.00/3 = 100.33
    assert abs(rec["stop_price"] - 100.33) < 0.01
    print("  PASS  test_orb_long_entry_is_trigger_not_close")


# ---------------------------------------------------------------------------
# 2 — SHORT entry estimate is the TRIGGER (OR_low-0.01), not the candle close
# ---------------------------------------------------------------------------
def test_orb_short_entry_is_trigger_not_close():
    bot, ib = make_bot(["NVDA"])
    seed_range(bot, "NVDA", high=101.00, low=99.00)
    set_live(bot, "NVDA", 98.99)
    rec = spy_open(bot)
    cur = cur_bar(high=99.60, low=98.50, close=98.70)   # closes 98.70 (old-bug value)

    fired = bot._try_orb_entry("NVDA", cur, _now(42), _now(42))

    assert fired is True
    assert rec["direction"] == "short"
    assert abs(rec["entry_price"] - 98.99) < 1e-9, (
        f"entry must be the trigger 98.99, got {rec['entry_price']}")
    assert rec["entry_price"] != 98.70
    # third_from_break (short): 99.00 + 2.00/3 = 99.67
    assert abs(rec["stop_price"] - 99.67) < 0.01
    print("  PASS  test_orb_short_entry_is_trigger_not_close")


# ---------------------------------------------------------------------------
# 3 — OR range is exact 09:30-09:40 from 1-min bars; a late (09:42) bar
#     does NOT pollute it
# ---------------------------------------------------------------------------
def test_orb_range_excludes_late_3min_bar():
    bot, ib = make_bot(["NVDA"])
    bars = []
    for m in range(30, 40):                      # 1-min bars 09:30..09:39
        hi = 101.0 if m == 31 else 100.5
        lo = 99.0  if m == 33 else 99.5
        bars.append(Bar(DAY.replace(hour=9, minute=m), 100, hi, lo, 100))
    # the "09:39-09:42 3-min bar" — labeled 09:42, WILD range — must be excluded
    bars.append(Bar(DAY.replace(hour=9, minute=42), 150, 200.0, 50.0, 150))

    rng = bot._orb_range_from_bars(bars, TODAY)
    assert rng == (101.0, 99.0), (
        f"OR must be the exact 09:30-09:40 range [99,101]; got {rng} "
        f"(the 09:42 bar polluted it)")
    print("  PASS  test_orb_range_excludes_late_3min_bar")


# ---------------------------------------------------------------------------
# 4 — Both sides crossed in one candle = ambiguous: no trade, shot preserved
# ---------------------------------------------------------------------------
def test_orb_ambiguous_both_sides_skips():
    bot, ib = make_bot(["NVDA"])
    seed_range(bot, "NVDA", high=101.00, low=99.00)
    set_live(bot, "NVDA", 100.00)
    rec = spy_open(bot)
    cur = cur_bar(high=101.50, low=98.50, close=100.00)   # crosses BOTH triggers

    fired = bot._try_orb_entry("NVDA", cur, _now(42), _now(42))

    assert fired is False
    assert rec == {}, "ambiguous bar must not place an order"
    assert bot._orb_state["NVDA"]["fired"] is False, (
        "ambiguous bar must NOT consume the one-shot")
    print("  PASS  test_orb_ambiguous_both_sides_skips")


# ---------------------------------------------------------------------------
# 5 — No entry before the OR window closes (09:40)
# ---------------------------------------------------------------------------
def test_orb_waits_for_window_close():
    bot, ib = make_bot(["NVDA"])
    seed_range(bot, "NVDA", high=101.00, low=99.00)
    rec = spy_open(bot)
    cur = cur_bar(high=101.50, low=100.40, close=101.30)

    fired = bot._try_orb_entry("NVDA", cur, _now(39), _now(39))   # 09:39 < 09:40
    assert fired is False
    assert rec == {}
    print("  PASS  test_orb_waits_for_window_close")


# ---------------------------------------------------------------------------
# 6 — One shot per symbol per day
# ---------------------------------------------------------------------------
def test_orb_one_shot():
    bot, ib = make_bot(["NVDA"])
    seed_range(bot, "NVDA", high=101.00, low=99.00)
    set_live(bot, "NVDA", 101.01)
    rec = spy_open(bot)
    cur = cur_bar(high=101.50, low=100.40, close=101.30)

    assert bot._try_orb_entry("NVDA", cur, _now(42), _now(42)) is True
    rec.clear()
    again = bot._try_orb_entry("NVDA", cur, _now(45), _now(45))
    assert again is False, "second call same day must be a no-op"
    assert rec == {}
    print("  PASS  test_orb_one_shot")


# ---------------------------------------------------------------------------
# 7 — Risk cap blocks an over-wide ORB and consumes the shot (no retry)
# ---------------------------------------------------------------------------
def test_orb_risk_cap_blocks():
    bot, ib = make_bot(["NVDA"])
    seed_range(bot, "NVDA", high=125.00, low=100.00)   # range 25 → big stop dist
    set_live(bot, "NVDA", 125.01)
    rec = spy_open(bot)
    cur = cur_bar(high=126.00, low=124.00, close=125.50)

    fired = bot._try_orb_entry("NVDA", cur, _now(42), _now(42))
    # trigger 125.01, third stop 125-25/3=116.67 → dist 8.34 × 100 = $834 > $700
    assert fired is False
    assert rec == {}, "over-cap risk must not place an order"
    assert bot._orb_state["NVDA"]["fired"] is True, "blocked ORB must not retry all day"
    print("  PASS  test_orb_risk_cap_blocks")


# ---------------------------------------------------------------------------
# 8 — Full integration: real _open_position + fill → managed Position
# ---------------------------------------------------------------------------
def test_orb_full_integration_creates_position():
    bot, ib = make_bot(["NVDA"])
    seed_range(bot, "NVDA", high=101.00, low=99.00)
    set_live(bot, "NVDA", 101.01)
    ib.set_fill_script("NVDA", FillScript(delay_seconds=1.0, fill_price=101.01))
    cur = cur_bar(high=101.50, low=100.40, close=101.30)

    fired = bot._try_orb_entry("NVDA", cur, _now(42), _now(42))
    ib.sleep(1.0)   # fire the delayed fill → _on_fill creates the Position

    assert fired is True
    assert "NVDA" in bot.positions
    assert bot.positions["NVDA"].direction == "long"
    assert abs(bot.positions["NVDA"].stop_price - 100.33) < 0.01
    print("  PASS  test_orb_full_integration_creates_position")


# ---------------------------------------------------------------------------
# 9 — Production: ORB is NOT decided in the 3-min handler
# ---------------------------------------------------------------------------
def test_orb_not_decided_in_3min_handler():
    import inspect
    src_3m = inspect.getsource(TradingBot._on_new_bar_3m)
    assert "_try_orb_entry" not in src_3m, (
        "_on_new_bar_3m must NOT make ORB entry decisions.")
    # and the live software-stop path DOES invoke it
    assert "_try_orb_entry" in inspect.getsource(TradingBot._refresh_orb_live)
    print("  PASS  test_orb_not_decided_in_3min_handler")


# ---------------------------------------------------------------------------
# 10 — Production: _refresh_orb_5m evaluates ORB on the 5-MINUTE bar high/low
# ---------------------------------------------------------------------------
def test_refresh_orb_5m_uses_5min_bar():
    bot, ib = make_bot(["NVDA"])
    seed_range(bot, "NVDA", high=101.00, low=99.00)
    # 5-min bars served by reqHistoricalData: a CLOSED breakout bar + a forming
    # bar. IBKR historical bars are labelled by START time: 09:40 covers
    # 09:40-09:45 and is closed at 09:46; 09:45 is still forming.
    closed_breakout = Bar(DAY.replace(hour=9, minute=40), 102.8, 103.50, 102.70, 103.0)
    forming         = Bar(DAY.replace(hour=9, minute=45), 103.0, 103.10, 102.90, 103.0)
    bot.ib.reqHistoricalData = lambda *a, **k: [closed_breakout, forming]

    captured = {}
    def spy(symbol, cur, now, effective_now):
        captured.update(symbol=symbol, cur_high=float(cur.high), cur_low=float(cur.low))
        return True
    bot._try_orb_entry = spy

    bot._refresh_orb_5m(_now(46))

    assert captured.get("symbol") == "NVDA", "ORB must be evaluated via the 5-min path"
    assert captured["cur_high"] == 103.50, (
        "ORB detection must use the 5-min bar high/low (the last CLOSED 5-min bar)")
    print("  PASS  test_refresh_orb_5m_uses_5min_bar")


# ---------------------------------------------------------------------------
# 11 — Production: live ORB uses real-time price as software stop trigger
# ---------------------------------------------------------------------------
def test_refresh_orb_live_uses_live_price_trigger():
    bot, ib = make_bot(["NVDA"])
    seed_range(bot, "NVDA", high=101.00, low=99.00)
    set_live(bot, "NVDA", 101.01)
    rec = spy_open(bot)

    bot._refresh_orb_live(_now(40))

    assert rec.get("symbol") == "NVDA"
    assert rec["direction"] == "long"
    assert abs(rec["entry_price"] - 101.01) < 1e-9
    assert rec["kw"]["entry_meta"]["trigger_source"] == "live_price"
    print("  PASS  test_refresh_orb_live_uses_live_price_trigger")


# ---------------------------------------------------------------------------
# 12 — Production: live ORB refuses stale bar fallback prices
# ---------------------------------------------------------------------------
def test_refresh_orb_live_ignores_stale_bar_fallback():
    bot, ib = make_bot(["NVDA"])
    seed_range(bot, "NVDA", high=101.00, low=99.00)
    # This would trigger if _refresh_orb_live used _rt_price()'s last-3m-bar
    # fallback. ORB entries must require actual ticker last/mid data.
    bot.bars_3m["NVDA"] = [Bar(DAY.replace(hour=9, minute=39),
                               101.0, 102.0, 100.5, 101.50)]
    rec = spy_open(bot)

    bot._refresh_orb_live(_now(40))

    assert rec == {}, "ORB live trigger must not use stale 3-min bar fallback"
    print("  PASS  test_refresh_orb_live_ignores_stale_bar_fallback")


# ---------------------------------------------------------------------------
# 13 — Backtest: orb_entry REQUIRES entry_freq='5min' (rejects 3-min loudly)
# ---------------------------------------------------------------------------
def test_backtest_orb_rejects_3min():
    from today_backtest import run_multi_today
    raised = False
    try:
        run_multi_today({"X": {"support": None, "resistance": None}},
                        date_str="2026-06-01", interval="1m", period="7d",
                        entry_freq="3min", orb_entry=True, prefetched={}, quiet=True)
    except ValueError as e:
        raised = "5min" in str(e)
    assert raised, "ORB backtest with entry_freq='3min' must raise a clear ValueError"
    print("  PASS  test_backtest_orb_rejects_3min")


# ---------------------------------------------------------------------------
# 14 — Backtest: orb_entry rejects coarse 5-min OR source data
# ---------------------------------------------------------------------------
def test_backtest_orb_rejects_coarse_or_data():
    from today_backtest import run_multi_today, _resample
    idx = pd.date_range("2026-06-01 09:30:00", "2026-06-01 15:55:00",
                        freq="5min", tz="US/Eastern")
    raw = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                        "Close": 100.0, "Volume": 50_000.0}, index=idx)
    coarse = _resample(raw, "5min")
    sd = {"TEST": {"df_5m": coarse, "df_10m": _resample(raw, "10min"),
                   "df_1m": coarse, "pmh_by": {}, "pml_by": {}, "atr": 5.0}}
    raised = False
    try:
        run_multi_today({"TEST": {"support": None, "resistance": None}},
                        date_str="2026-06-01", interval="1m", period="7d",
                        entry_freq="5min", orb_entry=True,
                        prefetched=sd, quiet=True)
    except ValueError as e:
        raised = "1-minute" in str(e)
    assert raised, "ORB must reject prefetched df_1m that is really 5-minute data"
    print("  PASS  test_backtest_orb_rejects_coarse_or_data")


# ---------------------------------------------------------------------------
# 15 — Backtest: orb_entry on 5-min creates an ORB trade at the TRIGGER price,
#      OR sourced from 1-min, under DEFAULT sizing ("risk")
# ---------------------------------------------------------------------------
def test_backtest_orb_creates_trades_5min_default_sizing():
    import numpy as np
    from datetime import time as _t
    from today_backtest import run_multi_today, _resample

    idx = pd.date_range("2026-06-01 09:30:00", "2026-06-01 15:59:00",
                        freq="1min", tz="US/Eastern")
    n = len(idx)
    o = np.full(n, 100.0); h = np.full(n, 100.2)
    l = np.full(n, 99.8);  c = np.full(n, 100.0); v = np.full(n, 50_000.0)
    for i, ts in enumerate(idx):
        tm = ts.time()
        if tm == _t(9, 31):    h[i] = 101.0; c[i] = 100.6     # OR high (09:30-09:39)
        elif tm == _t(9, 33):  l[i] = 99.0;  c[i] = 99.4      # OR low
        elif tm > _t(9, 40):   o[i] = 102.8; h[i] = 103.2; l[i] = 102.7; c[i] = 103.0  # break AFTER window
    raw = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx)

    sd = {"TEST": {"df_5m": _resample(raw, "5min"),     # 5-MIN entry frame
                   "df_10m": _resample(raw, "10min"),
                   "df_1m": raw.rename(columns=str.lower),
                   "pmh_by": {}, "pml_by": {}, "atr": 5.0}}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        trades = run_multi_today(
            {"TEST": {"support": None, "resistance": None}},
            date_str="2026-06-01", interval="1m", period="7d",
            entry_freq="5min", stale_secs=10**9, sizing="risk",   # DEFAULT sizing
            orb_entry=True, prefetched=sd, quiet=True)

    orb_trades = [t for t in trades
                  if "orb" in str(t.get("entry_signal", "")).lower()]
    assert orb_trades, (
        f"orb_entry (5-min, default sizing) must create an ORB trade; "
        f"got {[(t.get('entry_signal'), round(t.get('pnl', 0), 1)) for t in trades]}")
    # OR from 1-min (high 101) → trigger 101.01, NOT the breakout candle close 103
    assert abs(orb_trades[0]["entry"] - 101.01) < 0.02, (
        f"backtest ORB entry must be the trigger 101.01, got {orb_trades[0]['entry']}")
    print("  PASS  test_backtest_orb_creates_trades_5min_default_sizing")


# ---------------------------------------------------------------------------
# 16 — Velez experiment: ORB source must be exact 1-minute data
# ---------------------------------------------------------------------------
def test_velez_orb_requires_exact_1min_source():
    import numpy as np
    import velez_experiment as ve

    prev_allow = ve.ALLOW_COARSE_ORB_SOURCE
    try:
        ve.ALLOW_COARSE_ORB_SOURCE = False
        idx = pd.date_range("2026-06-01 09:30:00", "2026-06-01 10:00:00",
                            freq="1min", tz="US/Eastern")
        raw = pd.DataFrame(
            {
                "Open": np.full(len(idx), 100.0),
                "High": np.full(len(idx), 100.2),
                "Low": np.full(len(idx), 99.8),
                "Close": np.full(len(idx), 100.0),
                "Volume": np.full(len(idx), 50_000.0),
            },
            index=idx,
        )
        raw.loc[pd.Timestamp("2026-06-01 09:31:00", tz="US/Eastern"), "High"] = 101.0
        raw.loc[pd.Timestamp("2026-06-01 09:33:00", tz="US/Eastern"), "Low"] = 99.0

        bars = ve._resample_with_indicators(raw, "5min")
        one_min = raw.rename(columns=str.lower)
        order = ve._orb_order(
            bars.between_time("09:30", "16:00"),
            pd.Timestamp("2026-06-01 09:40:00", tz="US/Eastern"),
            orb_frame=one_min,
        )
        assert order["long_trigger"] == 101.01
        assert order["short_trigger"] == 98.99

        raised = False
        try:
            ve._orb_order(
                bars.between_time("09:30", "16:00"),
                pd.Timestamp("2026-06-01 09:40:00", tz="US/Eastern"),
                orb_frame=bars,
            )
        except ValueError as e:
            raised = "1-minute" in str(e)
        assert raised, "velez_experiment ORB must block coarse 5-min OR source data"
    finally:
        ve.ALLOW_COARSE_ORB_SOURCE = prev_allow
    print("  PASS  test_velez_orb_requires_exact_1min_source")


if __name__ == "__main__":
    tests = [
        test_orb_long_entry_is_trigger_not_close,
        test_orb_short_entry_is_trigger_not_close,
        test_orb_range_excludes_late_3min_bar,
        test_orb_ambiguous_both_sides_skips,
        test_orb_waits_for_window_close,
        test_orb_one_shot,
        test_orb_risk_cap_blocks,
        test_orb_full_integration_creates_position,
        test_orb_not_decided_in_3min_handler,
        test_refresh_orb_5m_uses_5min_bar,
        test_refresh_orb_live_uses_live_price_trigger,
        test_refresh_orb_live_ignores_stale_bar_fallback,
        test_backtest_orb_rejects_3min,
        test_backtest_orb_rejects_coarse_or_data,
        test_backtest_orb_creates_trades_5min_default_sizing,
        test_velez_orb_requires_exact_1min_source,
    ]
    passed = 0
    failed = []
    for t in tests:
        try:
            t(); passed += 1
        except AssertionError as e:
            failed.append((t.__name__, str(e))); print(f"  FAIL  {t.__name__}\n        {e}")
        except Exception as e:
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {t.__name__}")
            import traceback; traceback.print_exc()
    print("\n" + "=" * 60)
    print(f"  {passed}/{len(tests)} passed")
    if failed:
        for name, msg in failed:
            print(f"    - {name}: {msg}")
        sys.exit(1)
