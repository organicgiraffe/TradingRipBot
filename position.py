from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Position:
    symbol:      str
    direction:   str        # 'long' or 'short'
    shares:      int        # current remaining shares (drops by half after half-exit)
    entry_price: float
    entry_time:  datetime
    stop_price:  float
    ibkr_order_id: Optional[int] = None
    exit_price:  Optional[float] = None
    exit_time:   Optional[datetime] = None
    exit_reason: str = ""

    # ── Entry metadata ──────────────────────────────────────────────────
    entry_signal:    str   = ""    # 'cloud_flip' | 'pmh_breakout' | 'pml_breakdown'
    original_shares: int   = 0    # shares at entry (set in __post_init__)
    level_res: Optional[float] = None   # Rip's resistance — half-exit target (longs)
    level_sup: Optional[float] = None   # Rip's support    — half-exit target (shorts)

    # ── Trade management state ──────────────────────────────────────────
    half_exited:     bool  = False  # True once 50% of shares exited at Rip's level
    best_unrealised: float = 0.0   # HWM per-share profit (intrabar-safe ratchet)

    def __post_init__(self):
        # Capture original share count before any partial exits
        if self.original_shares == 0:
            self.original_shares = self.shares

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
        """Current dollar risk based on stop and remaining shares."""
        if self.direction == "long":
            return (self.entry_price - self.stop_price) * self.shares
        return (self.stop_price - self.entry_price) * self.shares

    def update_stop(self, new_stop: float):
        """Tighten trailing stop — never loosens it."""
        if self.direction == "long":
            self.stop_price = max(self.stop_price, new_stop)
        else:
            self.stop_price = min(self.stop_price, new_stop)

    def close(self, price: float, time: datetime, reason: str = ""):
        self.exit_price  = price
        self.exit_time   = time
        self.exit_reason = reason

    def summary(self) -> str:
        if self.is_open:
            return (f"{self.direction.upper()} {self.shares}x {self.symbol} "
                    f"@ {self.entry_price:.2f}  stop={self.stop_price:.2f}  "
                    f"[risk=${self.risk:.0f}  OPEN]")
        sign = "+" if (self.pnl or 0) >= 0 else ""
        return (f"{self.direction.upper()} {self.shares}x {self.symbol} "
                f"@ {self.entry_price:.2f} -> {self.exit_price:.2f}  "
                f"pnl={sign}${self.pnl:.0f}  [{self.exit_reason}]")
