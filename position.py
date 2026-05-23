from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Position:
    symbol:      str
    direction:   str        # 'long' or 'short'
    shares:      int
    entry_price: float
    entry_time:  datetime
    stop_price:  float      # computed from previous 3-min candle at entry
    ibkr_order_id: Optional[int] = None
    exit_price:  Optional[float] = None
    exit_time:   Optional[datetime] = None
    exit_reason: str = ""

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def pnl(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        if self.direction == "long":
            return (self.exit_price - self.entry_price) * self.shares
        return (self.entry_price - self.exit_price) * self.shares

    @property
    def risk(self) -> float:
        """Max dollar risk based on stop."""
        if self.direction == "long":
            return (self.entry_price - self.stop_price) * self.shares
        return (self.stop_price - self.entry_price) * self.shares

    def update_stop(self, new_stop: float):
        """Tighten trailing stop — never loosens."""
        if self.direction == "long":
            self.stop_price = max(self.stop_price, new_stop)
        else:
            self.stop_price = min(self.stop_price, new_stop)

    def close(self, price: float, time: datetime, reason: str = ""):
        self.exit_price  = price
        self.exit_time   = time
        self.exit_reason = reason

    def summary(self) -> str:
        risk_str = f"risk=${self.risk:.2f}"
        if self.is_open:
            return (f"{self.direction.upper()} {self.shares}x {self.symbol} "
                    f"@ {self.entry_price:.2f}  stop={self.stop_price:.2f}  [{risk_str}  OPEN]")
        return (f"{self.direction.upper()} {self.shares}x {self.symbol} "
                f"@ {self.entry_price:.2f} → {self.exit_price:.2f}  "
                f"pnl=${self.pnl:+.2f}  [{self.exit_reason}]")
