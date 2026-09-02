from __future__ import annotations

import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from fxbot.journal import StructuredJournal
from fxbot.models import BotRunState


class StructuredJournalTests(unittest.TestCase):
    def test_order_reservation_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                first, created_first = journal.reserve_order(
                    client_order_id="client-1",
                    timestamp=datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc),
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    order_type="MARKET",
                    risk_amount=2.0,
                    payload={"leg": "full"},
                )
                second, created_second = journal.reserve_order(
                    client_order_id="client-1",
                    timestamp=datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc),
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    order_type="MARKET",
                    risk_amount=2.0,
                    payload={"leg": "full"},
                )

                self.assertTrue(created_first)
                self.assertFalse(created_second)
                self.assertEqual(first.id, second.id)

    def test_order_status_update_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                journal.reserve_order(
                    client_order_id="client-2",
                    timestamp=datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc),
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    order_type="MARKET",
                    risk_amount=2.0,
                    payload={},
                )

                journal.update_order("client-2", status="filled", broker_order_id="10", broker_trade_id="11")
                row = journal.find_order("client-2")

                self.assertIsNotNone(row)
                self.assertEqual(row.status, "filled")
                self.assertEqual(row.broker_trade_id, "11")

    def test_current_positions_are_marked_closed_when_absent_from_latest_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                timestamp = datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc)
                journal.record_position_snapshot(
                    timestamp=timestamp,
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    avg_price=1.1,
                    unrealized_pl=3.0,
                    margin_used=40.0,
                    price=1.103,
                    estimated_daily_financing=-0.02,
                    payload={},
                )

                self.assertEqual(len(journal.current_positions()), 1)

                journal.mark_current_positions_closed(set(), timestamp)

                self.assertEqual(journal.current_positions(), [])

    def test_daily_loss_halt_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                timestamp = datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc)
                self.assertIsNone(
                    journal.update_protection_state(
                        timestamp=timestamp,
                        equity=10_000,
                        max_daily_loss_pct=0.02,
                        max_drawdown_pct=0.10,
                    )
                )

                reason = journal.update_protection_state(
                    timestamp=timestamp,
                    equity=9_750,
                    max_daily_loss_pct=0.02,
                    max_drawdown_pct=0.10,
                )

                self.assertEqual(reason, "daily_loss_halt")
                self.assertEqual(journal.get_state().state, BotRunState.HALTED.value)


if __name__ == "__main__":
    unittest.main()
