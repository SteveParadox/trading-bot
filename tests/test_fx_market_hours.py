from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fxbot.config import StrategySettings
from fxbot.market_hours import active_sessions, is_fx_market_open, trading_allowed_now


class FxMarketHoursTests(unittest.TestCase):
    def test_market_closed_between_friday_and_sunday_new_york_close(self) -> None:
        self.assertFalse(is_fx_market_open(datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)))
        self.assertFalse(is_fx_market_open(datetime(2026, 1, 11, 20, 0, tzinfo=timezone.utc)))
        self.assertTrue(is_fx_market_open(datetime(2026, 1, 11, 23, 0, tzinfo=timezone.utc)))

    def test_london_new_york_overlap_session_detection(self) -> None:
        sessions = active_sessions(datetime(2026, 1, 6, 13, 0, tzinfo=timezone.utc))

        self.assertIn("london", sessions)
        self.assertIn("new_york", sessions)
        self.assertIn("overlap", sessions)

    def test_trading_guard_blocks_asian_session_by_default(self) -> None:
        allowed, reason = trading_allowed_now(
            "EUR_USD",
            StrategySettings(),
            [],
            datetime(2026, 1, 6, 2, 0, tzinfo=timezone.utc),
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "session_filter")


if __name__ == "__main__":
    unittest.main()

