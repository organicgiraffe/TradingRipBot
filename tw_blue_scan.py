"""
Scan daily candles for the blue trigger used in
pine_scripts/trade_whisperer_style_smart_candles.pine.

This is for watchlist generation, not financial advice.

Usage:
    python tw_blue_scan.py
    python tw_blue_scan.py TSLA NVDA AMD PLTR
"""

from __future__ import annotations

import sys
import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "AMD", "NFLX", "PLTR", "COIN", "MSTR", "HOOD", "CRWD", "ARM",
    "APP", "SMCI", "MU", "MRVL", "TSM", "ORCL", "DELL", "SHOP",
    "UBER", "SOFI", "RDDT", "IONQ", "RGTI", "QBTS", "ACHR", "JOBY",
    "BBAI", "AI", "OPEN", "DAVE", "VRT", "OKLO", "SMR", "NVTS",
    "SOUN", "APLD", "ASTS", "RKLB", "HIMS", "SE", "TTD", "NET",
]

BB_LEN = 20
VOL_LEN = 20
PEAK_LEN = 120
TRIGGER_SCORE = 3.0


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False,
                        min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False,
                        min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return (100 - (100 / (1 + rs))).fillna(50)


def volume_peak_proxy(df: pd.DataFrame, lookback: int = PEAK_LEN) -> pd.Series:
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    peaks = []
    for i in range(len(df)):
        start = max(0, i - lookback + 1)
        window = df.iloc[start:i + 1]
        latest_peak_pos = window["volume"].to_numpy().argmax()
        peaks.append(float(hlc3.iloc[start + latest_peak_pos]))
    return pd.Series(peaks, index=df.index)


def add_signal_columns(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    df["ema8"] = df["close"].ewm(span=8, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["bb_mid"] = df["close"].rolling(BB_LEN).mean()
    df["rsi14"] = rsi(df["close"])
    df["rvol"] = df["volume"] / df["volume"].rolling(VOL_LEN).mean()
    df["volume_peak"] = volume_peak_proxy(df)
    df["mid_slope"] = df["bb_mid"] - df["bb_mid"].shift(5)

    up_candle = df["close"] >= df["open"]
    down_candle = df["close"] <= df["open"]

    bull = pd.Series(0.0, index=df.index)
    bull += (df["close"] > df["bb_mid"]).astype(float) * 1.0
    bull += (df["ema8"] > df["ema21"]).astype(float) * 1.0
    bull += (df["close"] > df["sma50"]).astype(float) * 1.0
    bull += (df["close"] > df["sma200"]).astype(float) * 0.5
    bull += (df["close"] > df["volume_peak"]).astype(float) * 1.0
    bull += (df["rsi14"] > 52).astype(float) * 1.0
    bull += (df["mid_slope"] > 0).astype(float) * 0.5
    bull += ((df["rvol"] > 1.1) & up_candle).astype(float) * 0.75
    bull += (df["close"] > df["close"].shift(1)).astype(float) * 0.5

    bear = pd.Series(0.0, index=df.index)
    bear += (df["close"] < df["bb_mid"]).astype(float) * 1.0
    bear += (df["ema8"] < df["ema21"]).astype(float) * 1.0
    bear += (df["close"] < df["sma50"]).astype(float) * 1.0
    bear += (df["close"] < df["sma200"]).astype(float) * 0.5
    bear += (df["close"] < df["volume_peak"]).astype(float) * 1.0
    bear += (df["rsi14"] < 48).astype(float) * 1.0
    bear += (df["mid_slope"] < 0).astype(float) * 0.5
    bear += ((df["rvol"] > 1.1) & down_candle).astype(float) * 0.75
    bear += (df["close"] < df["close"].shift(1)).astype(float) * 0.5

    df["score"] = bull - bear
    cross_up_mid = (df["close"] > df["bb_mid"]) & (
        df["close"].shift(1) <= df["bb_mid"].shift(1))
    cross_up_peak = (df["close"] > df["volume_peak"]) & (
        df["close"].shift(1) <= df["volume_peak"].shift(1))
    cross_up = cross_up_mid | cross_up_peak
    ready = pd.Series(range(len(df)), index=df.index) > max(PEAK_LEN, 200)
    df["blue"] = (
        ready
        & (df["score"] >= TRIGGER_SCORE)
        & ((df["score"].shift(1) < TRIGGER_SCORE) | cross_up)
    )
    return df


def scan(symbols: list[str]) -> list[dict]:
    rows = []
    for symbol in symbols:
        try:
            raw = yf.download(symbol, period="2y", interval="1d",
                              auto_adjust=True, progress=False)
            if raw.empty:
                print(f"{symbol:<6} no data")
                continue
            df = add_signal_columns(raw)
            if len(df) < 205:
                print(f"{symbol:<6} not enough history")
                continue
            last = df.iloc[-1]
            color = "BLUE" if bool(last["blue"]) else "not blue"
            print(f"{symbol:<6} {color:<8} {last.name.date()} "
                  f"close={last.close:.2f} score={last.score:.2f}")
            if bool(last["blue"]):
                rows.append({
                    "symbol": symbol,
                    "date": last.name.date().isoformat(),
                    "close": round(float(last.close), 2),
                    "score": round(float(last.score), 2),
                    "rsi14": round(float(last.rsi14), 1),
                    "rvol": round(float(last.rvol), 2),
                })
        except Exception as exc:
            print(f"{symbol:<6} error: {exc}")
    return rows


def main() -> None:
    symbols = [s.upper() for s in sys.argv[1:]] or DEFAULT_SYMBOLS
    blue = scan(symbols)
    print("\nLatest daily BLUE candles:")
    if blue:
        for row in blue:
            print(f"  {row['symbol']:<6} {row['date']} "
                  f"close={row['close']:<8} score={row['score']:<4} "
                  f"rsi={row['rsi14']:<5} rvol={row['rvol']}")
    else:
        print("  None in this symbol list.")


if __name__ == "__main__":
    main()
