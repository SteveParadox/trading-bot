from __future__ import annotations

import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fxbot.journal import StructuredJournal, row_to_dict
from fxbot.models import BotRunState
from fxbot.config import BrokerSettings, FxBotSettings, RuntimeSettings
from fxbot.forward import ForwardTestWorker
from fxbot.instruments import FxInstrument, PriceSnapshot
from fxbot.outcome_tracker import CandidateOutcomeTracker

class StructuredJournalTests(unittest.TestCase):
    def test_equity_history_is_chronological_and_keeps_date_range_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                start = datetime(2026, 1, 1, tzinfo=timezone.utc)
                for index in range(20):
                    timestamp = start + timedelta(hours=index)
                    journal.record_equity(timestamp=timestamp, payload={"equity": 1000 + index, "balance": 1000})
                rows = journal.equity_history(start=start + timedelta(hours=4), end=start + timedelta(hours=15), limit=4)
                actual = [row.timestamp.replace(tzinfo=timezone.utc) for row in rows]
                self.assertEqual(actual[0], start + timedelta(hours=4))
                self.assertEqual(actual[-1], start + timedelta(hours=15))
                self.assertEqual(actual, sorted(actual))
                self.assertEqual(len(actual), 4)
                serialized = row_to_dict(rows[0])
                self.assertTrue(serialized["timestamp"].endswith("+00:00"))

    def test_forward_sync_records_closed_history_and_updates_existing_trade(self) -> None:
        class HistoryClient:
            def __init__(self):
                self.since = None

            def closed_trades_since(self, since, until):
                self.since = since
                return [{
                    "broker_trade_id": "mt5-closed-1", "instrument": "EUR_USD", "side": "LONG",
                    "units": 1000, "exit_time": datetime(2026, 1, 6, 14, 5, tzinfo=timezone.utc),
                    "exit_price": 1.103, "realized_pl": 2.75, "financing": -0.1,
                    "exit_reason": "mt5_history_deal",
                }]

        with tempfile.TemporaryDirectory() as tmp:
            settings = FxBotSettings(
                broker=BrokerSettings(),
                runtime=RuntimeSettings(database_url=f"sqlite:///{Path(tmp) / 'journal.db'}", log_jsonl_path=None),
            )
            with closing(StructuredJournal(settings.runtime.database_url)) as journal:
                journal.upsert_trade(
                    broker_trade_id="mt5-closed-1", instrument="EUR_USD", side="LONG", units=1000,
                    state="open", entry_time=datetime(2026, 1, 6, 14, tzinfo=timezone.utc),
                    entry_price=1.1, payload={"strategy_context": {"signal_score": 71}},
                )
                client = HistoryClient()
                worker = ForwardTestWorker(settings, client=client, journal=journal)
                worker._sync_trade_history(datetime(2026, 1, 6, 14, 10, tzinfo=timezone.utc))
                closed = journal.find_trade("mt5-closed-1")
                self.assertIsNotNone(closed)
                self.assertEqual(closed.state, "closed")
                self.assertAlmostEqual(closed.realized_pl, 2.75)
                self.assertAlmostEqual(closed.financing, -0.1)
                self.assertEqual(closed.payload["strategy_context"]["signal_score"], 71)
                self.assertEqual(client.since, datetime(2025, 12, 7, 14, 10, tzinfo=timezone.utc))
                worker.close()

    def test_old_order_ticket_row_is_archived_when_position_ticket_closes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                journal.upsert_trade(broker_trade_id="opening-order-42", instrument="EUR_USD", side="LONG", units=1000, state="open")
                journal.upsert_trade(broker_trade_id="position-9001", instrument="EUR_USD", side="LONG", units=1000, state="closed", exit_time=datetime(2026, 1, 6, tzinfo=timezone.utc), realized_pl=3.0)
                alias = journal.reconcile_duplicate_open_trade(canonical_trade_id="position-9001", instrument="EUR_USD", units=1000, exit_time=datetime(2026, 1, 6, tzinfo=timezone.utc), exit_price=1.103)
                self.assertEqual(alias, "opening-order-42")
                self.assertEqual(journal.find_trade("opening-order-42").state, "reconciled_alias")
                self.assertEqual([row.broker_trade_id for row in journal.filtered_trades()], ["position-9001"])

    def test_forward_targeted_recovery_repairs_legacy_open_row(self) -> None:
        class LegacyHistoryClient:
            def __init__(self):
                self.references = None

            def closed_trades_since(self, since, until):
                return []

            def closed_trade_for_references(self, references, since, until):
                self.references = references
                return {
                    "broker_trade_id": "9001", "instrument": "EUR_USD", "side": "LONG",
                    "units": 500, "exit_time": datetime(2026, 1, 6, 14, 5, tzinfo=timezone.utc),
                    "exit_price": 1.103, "realized_pl": 2.75, "financing": -0.1,
                    "exit_reason": "mt5_history_deal",
                }

        with tempfile.TemporaryDirectory() as tmp:
            settings = FxBotSettings(
                broker=BrokerSettings(),
                runtime=RuntimeSettings(database_url=f"sqlite:///{Path(tmp) / 'journal.db'}", log_jsonl_path=None),
            )
            with closing(StructuredJournal(settings.runtime.database_url)) as journal:
                journal.upsert_trade(
                    broker_trade_id="42", instrument="EUR_USD", side="LONG", units=1000,
                    state="open", entry_time=datetime(2026, 1, 6, 14, tzinfo=timezone.utc),
                    payload={"mt5": {"order": 42, "deal": 84}},
                )
                client = LegacyHistoryClient()
                worker = ForwardTestWorker(settings, client=client, journal=journal)
                worker._sync_trade_history(datetime(2026, 1, 6, 14, 10, tzinfo=timezone.utc))

                self.assertEqual(client.references, ["42", "84"])
                self.assertEqual(journal.find_trade("42").state, "reconciled_alias")
                self.assertEqual(journal.find_trade("9001").state, "closed")
                self.assertEqual([row.broker_trade_id for row in journal.filtered_trades()], ["9001"])
                worker.close()
    def test_candidate_journal_is_idempotent_and_tracks_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                timestamp = datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc)
                first, created_first = journal.record_candidate(
                    candidate_id="fxsig-EURUSD-test",
                    timestamp=timestamp,
                    symbol="EUR_USD",
                    direction="LONG",
                    entry=1.1010,
                    stop_loss=1.0990,
                    take_profit=1.1050,
                    spread=0.0002,
                    atr=0.0010,
                    momentum=0.7,
                    trend_strength=28.0,
                    news_risk={"risk_level": "MEDIUM"},
                    strategy_signal="signal_confirmed",
                    payload={"first_observation": True},
                )
                second, created_second = journal.record_candidate(
                    candidate_id="fxsig-EURUSD-test",
                    timestamp=timestamp + timedelta(seconds=30),
                    symbol="EUR_USD",
                    direction="LONG",
                    entry=1.1020,
                    spread=0.0004,
                    strategy_signal="signal_confirmed",
                    payload={"first_observation": False},
                )

                self.assertTrue(created_first)
                self.assertFalse(created_second)
                self.assertEqual(first.candidate_id, second.candidate_id)
                self.assertEqual(second.entry, 1.1010)
                self.assertEqual(second.payload["first_observation"], True)

                journal.update_candidate(
                    "fxsig-EURUSD-test",
                    status="rejected",
                    executed=False,
                    rejection_reason="spread_to_atr_filter",
                )
                rejected = journal.find_candidate("fxsig-EURUSD-test")
                self.assertIsNotNone(rejected)
                self.assertFalse(rejected.executed)
                self.assertEqual(rejected.rejection_reason, "spread_to_atr_filter")

                journal.update_candidate(
                    "fxsig-EURUSD-test",
                    status="executed",
                    executed=True,
                    rejection_reason=None,
                    stop_loss=1.0992,
                    take_profit=1.1052,
                )
                executed = journal.find_candidate("fxsig-EURUSD-test")
                self.assertTrue(executed.executed)
                self.assertEqual(executed.status, "executed")
                self.assertIsNone(executed.rejection_reason)
                self.assertEqual(executed.stop_loss, 1.0992)
                self.assertEqual(executed.take_profit, 1.1052)
                self.assertEqual(len(journal.recent_candidates()), 1)


    def test_candidate_outcomes_capture_future_returns_waits_and_first_touch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                start = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)
                candidate, _ = journal.record_candidate(
                    candidate_id="fxsig-outcome-long", timestamp=start, symbol="EUR_USD", direction="LONG",
                    entry=1.1002, stop_loss=1.0992, take_profit=1.1012, spread=0.0002,
                    atr=0.001, momentum=0.5, trend_strength=25.0, news_risk={},
                    strategy_signal="signal_confirmed",
                )
                tracker = CandidateOutcomeTracker(journal, observation_lag_tolerance_seconds=2000)
                instrument = FxInstrument("EUR_USD")
                tracker.seed(
                    candidate=candidate,
                    price=PriceSnapshot("EUR_USD", bid=1.1000, ask=1.1002, time=start),
                    instrument=instrument,
                    observed_at=start,
                )
                for seconds, bid, ask in [
                    (30, 1.0998, 1.1000), (60, 1.1006, 1.1008), (90, 1.1013, 1.1015),
                    (180, 1.1004, 1.1006), (300, 1.1009, 1.1011),
                    (900, 1.1010, 1.1012), (1800, 1.0997, 1.0999),
                ]:
                    observed = start + timedelta(seconds=seconds)
                    tracker.observe(
                        candidate=candidate,
                        price=PriceSnapshot("EUR_USD", bid=bid, ask=ask, time=observed),
                        instrument=instrument,
                        observed_at=observed,
                    )
                outcome = journal.find_candidate_outcome(candidate.candidate_id)
                self.assertEqual(outcome.status, "complete")
                self.assertEqual(outcome.first_touch, "TP")
                self.assertTrue(outcome.tp_before_sl)
                self.assertAlmostEqual(outcome.mfe_pips, 11.0)
                self.assertAlmostEqual(outcome.mae_pips, 5.0)
                self.assertAlmostEqual(outcome.return_1m_pips, 4.0)
                self.assertAlmostEqual(outcome.return_3m_pips, 2.0)
                self.assertAlmostEqual(outcome.return_5m_pips, 7.0)
                self.assertAlmostEqual(outcome.return_15m_pips, 8.0)
                self.assertAlmostEqual(outcome.return_30m_pips, -5.0)
                self.assertAlmostEqual(outcome.wait_30s_improvement_pips, 2.0)
                self.assertAlmostEqual(outcome.wait_1m_improvement_pips, -6.0)
                self.assertAlmostEqual(outcome.wait_3m_improvement_pips, -4.0)
                self.assertAlmostEqual(outcome.wait_5m_improvement_pips, -9.0)
                self.assertTrue(outcome.payload["spread_included_in_returns"])
                self.assertFalse(outcome.payload["commission_and_slippage_included"])

    def test_candidate_outcomes_do_not_backfill_missed_horizons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                start = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)
                candidate, _ = journal.record_candidate(
                    candidate_id="fxsig-outcome-gap", timestamp=start, symbol="EUR_USD", direction="LONG",
                    entry=1.1002, stop_loss=1.0992, take_profit=1.1012, spread=0.0002,
                    strategy_signal="signal_confirmed",
                )
                tracker = CandidateOutcomeTracker(journal, observation_lag_tolerance_seconds=30)
                instrument = FxInstrument("EUR_USD")
                tracker.seed(
                    candidate=candidate,
                    price=PriceSnapshot("EUR_USD", bid=1.1000, ask=1.1002, time=start),
                    instrument=instrument,
                    observed_at=start,
                )
                late = start + timedelta(seconds=100)
                tracker.observe(
                    candidate=candidate,
                    price=PriceSnapshot("EUR_USD", bid=1.1010, ask=1.1012, time=late),
                    instrument=instrument,
                    observed_at=late,
                )
                outcome = journal.find_candidate_outcome(candidate.candidate_id)
                self.assertIsNone(outcome.return_1m_pips)
                self.assertIn("return_1m_pips", outcome.payload["missed_horizons"])
                self.assertEqual(outcome.data_quality, "degraded")
                much_later = start + timedelta(seconds=1900)
                tracker.observe(
                    candidate=candidate,
                    price=PriceSnapshot("EUR_USD", bid=1.1020, ask=1.1022, time=much_later),
                    instrument=instrument,
                    observed_at=much_later,
                )
                outcome = journal.find_candidate_outcome(candidate.candidate_id)
                self.assertEqual(outcome.status, "incomplete")
                self.assertIsNone(outcome.return_30m_pips)
                self.assertEqual(outcome.payload["incomplete_reason"], "missed_30m_horizon")

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

    def test_filtered_trade_outcome_uses_realized_pnl_and_financing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                timestamp = datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc)
                journal.upsert_trade(
                    broker_trade_id="winner-after-financing",
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    state="closed",
                    entry_time=timestamp,
                    realized_pl=2.0,
                    financing=-1.0,
                )
                journal.upsert_trade(
                    broker_trade_id="loser-after-financing",
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    state="closed",
                    entry_time=timestamp,
                    realized_pl=-2.0,
                    financing=1.0,
                )

                wins = journal.filtered_trades(outcome="win")
                losses = journal.filtered_trades(outcome="loss")

                self.assertEqual([row.broker_trade_id for row in wins], ["winner-after-financing"])
                self.assertEqual([row.broker_trade_id for row in losses], ["loser-after-financing"])

    def test_trade_close_preserves_initial_strategy_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                timestamp = datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc)
                journal.upsert_trade(
                    broker_trade_id="context-preserved",
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    state="open",
                    entry_time=timestamp,
                    payload={"strategy_context": {"signal_score": 73.5}},
                )
                closed = journal.upsert_trade(
                    broker_trade_id="context-preserved",
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    state="closed",
                    exit_time=timestamp,
                    realized_pl=5.0,
                    payload={"exit_source": "mt5_history"},
                )

                self.assertEqual(closed.payload["strategy_context"]["signal_score"], 73.5)
                self.assertEqual(closed.payload["exit_source"], "mt5_history")


if __name__ == "__main__":
    unittest.main()
