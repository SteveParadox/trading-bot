from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from forex_agent.data.schemas import TradeRecord


@dataclass
class TradeIssue:
    trade_id: str
    field: str
    severity: str
    message: str


@dataclass
class DataQualityReport:
    total_trades: int = 0
    issues: list[TradeIssue] = field(default_factory=list)
    clean_trades: int = 0

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


def validate_trades(trades: list[TradeRecord]) -> DataQualityReport:
    report = DataQualityReport(total_trades=len(trades))

    seen_ids: dict[str, int] = {}
    now = datetime.utcnow()

    for trade in trades:
        seen_ids[trade.trade_id] = seen_ids.get(trade.trade_id, 0) + 1

        if trade.exit_price is None:
            report.issues.append(
                TradeIssue(
                    trade_id=trade.trade_id,
                    field="exit_price",
                    severity="warning",
                    message="Trade has no exit price (possibly still open)",
                )
            )

        if trade.direction == "LONG" and trade.exit_price is not None:
            if trade.take_profit and trade.exit_price > trade.take_profit:
                report.issues.append(
                    TradeIssue(
                        trade_id=trade.trade_id,
                        field="exit_price",
                        severity="error",
                        message=(
                            f"Exit price {trade.exit_price} exceeds take profit "
                            f"{trade.take_profit} for LONG trade"
                        ),
                    )
                )

        if trade.direction == "SHORT" and trade.exit_price is not None:
            if trade.take_profit and trade.exit_price < trade.take_profit:
                report.issues.append(
                    TradeIssue(
                        trade_id=trade.trade_id,
                        field="exit_price",
                        severity="error",
                        message=(
                            f"Exit price {trade.exit_price} below take profit "
                            f"{trade.take_profit} for SHORT trade"
                        ),
                    )
                )

        if trade.entry_price <= 0:
            report.issues.append(
                TradeIssue(
                    trade_id=trade.trade_id,
                    field="entry_price",
                    severity="error",
                    message=f"Invalid entry price: {trade.entry_price}",
                )
            )

        if trade.entry_time > now:
            report.issues.append(
                TradeIssue(
                    trade_id=trade.trade_id,
                    field="entry_time",
                    severity="error",
                    message=f"Entry time is in the future: {trade.entry_time}",
                )
            )

        if trade.exit_time and trade.exit_time < trade.entry_time:
            report.issues.append(
                TradeIssue(
                    trade_id=trade.trade_id,
                    field="exit_time",
                    severity="error",
                    message="Exit time is before entry time",
                )
            )

        if trade.risk_amount < 0:
            report.issues.append(
                TradeIssue(
                    trade_id=trade.trade_id,
                    field="risk_amount",
                    severity="error",
                    message=f"Negative risk amount: {trade.risk_amount}",
                )
            )

    for tid, count in seen_ids.items():
        if count > 1:
            report.issues.append(
                TradeIssue(
                    trade_id=tid,
                    field="trade_id",
                    severity="error",
                    message=f"Duplicate trade ID: {tid} appears {count} times",
                )
            )

    report.clean_trades = report.total_trades - report.error_count
    return report
