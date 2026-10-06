from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from fxbot.config import BrokerSettings
from fxbot.instruments import FxInstrument, PriceSnapshot, normalize_instrument_name
from fxbot.mt5 import Mt5Client, _mt5_comment


class Mt5ConfigAndInstrumentTests(unittest.TestCase):
    def test_symbol_map_keeps_strategy_names_and_broker_symbols_separate(self) -> None:
        settings = BrokerSettings(symbol_map={"eurusd": "EURUSD.a"})

        self.assertEqual(settings.broker_symbol_for("EUR_USD"), "EURUSD.a")
        self.assertEqual(settings.strategy_symbol_for("EURUSD.a"), "EUR_USD")
        self.assertEqual(settings.broker_symbol_for("USD_JPY"), "USDJPY")

    def test_mt5_instrument_uses_contract_size_and_lot_step_for_units(self) -> None:
        info = SimpleNamespace(
            name="EURUSD.a",
            digits=5,
            point=0.00001,
            trade_contract_size=100_000,
            volume_min=0.01,
            volume_max=50,
            volume_step=0.01,
            trade_stops_level=20,
        )

        instrument = FxInstrument.from_mt5("EUR_USD", info, account_leverage=30, broker_symbol="EURUSD.a")

        self.assertEqual(instrument.name, "EUR_USD")
        self.assertEqual(instrument.broker_symbol, "EURUSD.a")
        self.assertEqual(instrument.minimum_trade_size, 1_000)
        self.assertEqual(instrument.round_units(2_450), 2_000)
        self.assertEqual(instrument.units_to_volume(2_450), 0.02)
        self.assertAlmostEqual(instrument.margin_rate, 1 / 30)
        self.assertAlmostEqual(instrument.minimum_stop_distance, 0.0002)

    def test_mt5_tick_normalizes_broker_symbol_to_strategy_pair(self) -> None:
        tick = SimpleNamespace(bid=150.12, ask=150.13, time=1_767_225_600)

        snapshot = PriceSnapshot.from_mt5("USDJPY", tick)

        self.assertEqual(snapshot.instrument, "USD_JPY")
        self.assertEqual(snapshot.mid, 150.125)

    def test_mt5_timestamp_offset_normalizes_quotes_and_candles(self) -> None:
        class FakeMt5:
            TIMEFRAME_M15 = 15

            def initialize(self, *args, **kwargs): return True
            def account_info(self): return SimpleNamespace(currency="USD", trade_mode=0)
            def symbol_info(self, symbol): return SimpleNamespace(name=symbol, visible=True)
            def symbol_info_tick(self, symbol):
                return SimpleNamespace(bid=1.1, ask=1.1001, time=1_700_010_800, time_msc=1_700_010_800_000)
            def copy_rates_from_pos(self, symbol, timeframe, start, count):
                return [{"time": 1_700_010_800, "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15, "tick_volume": 10}]
            def last_error(self): return "ok"

        client = Mt5Client(BrokerSettings(time_offset_seconds=-10_800), module=FakeMt5())

        quote = client.pricing(["EUR_USD"]).prices["EUR_USD"]
        candles = client.candles("EUR_USD", "15m", 1)

        expected = datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)
        self.assertEqual(quote.time, expected)
        self.assertEqual(candles.index[0].to_pydatetime(), expected)

    def test_mt5_historical_ranges_apply_same_timestamp_offset_and_preserve_spread(self) -> None:
        class FakeMt5:
            TIMEFRAME_M15 = 15
            COPY_TICKS_INFO = 1

            def initialize(self, *args, **kwargs): return True
            def account_info(self): return SimpleNamespace(currency="USD", trade_mode=0)
            def symbol_info(self, symbol): return SimpleNamespace(name=symbol, visible=True)
            def copy_rates_range(self, symbol, timeframe, start, end):
                return [{
                    "time": 1_700_010_800,
                    "open": 1.1,
                    "high": 1.2,
                    "low": 1.0,
                    "close": 1.15,
                    "tick_volume": 10,
                    "spread": 12,
                }]
            def copy_ticks_range(self, symbol, start, end, flags):
                return [
                    {"time_msc": 1_700_010_800_000, "bid": 1.1000, "ask": 1.1002},
                    {"time_msc": 1_700_010_801_000, "bid": 1.1001, "ask": 1.1003},
                ]
            def last_error(self): return "ok"

        client = Mt5Client(BrokerSettings(time_offset_seconds=-10_800), module=FakeMt5())
        start = datetime.fromtimestamp(1_699_999_000, tz=timezone.utc)
        end = datetime.fromtimestamp(1_700_001_000, tz=timezone.utc)

        candles = client.historical_candles("EUR_USD", "15m", start, end)
        ticks = client.historical_ticks("EUR_USD", start, end)

        expected = datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)
        self.assertEqual(candles.index[0].to_pydatetime(), expected)
        self.assertEqual(candles.iloc[0]["spread_points"], 12)
        self.assertEqual(ticks.iloc[0]["timestamp"].to_pydatetime(), expected)
        self.assertEqual(ticks.iloc[0]["instrument"], "EUR_USD")
        self.assertAlmostEqual(ticks.iloc[0]["bid"], 1.1000)
        self.assertAlmostEqual(ticks.iloc[0]["ask"], 1.1002)


    def test_mt5_historical_candles_chunk_large_ranges(self) -> None:
        class FakeMt5:
            TIMEFRAME_M15 = 15

            def __init__(self) -> None:
                self.calls: list[tuple[datetime, datetime]] = []

            def initialize(self, *args, **kwargs): return True
            def account_info(self): return SimpleNamespace(currency="USD", trade_mode=0)
            def symbol_info(self, symbol): return SimpleNamespace(name=symbol, visible=True)
            def copy_rates_range(self, symbol, timeframe, start, end):
                self.calls.append((start, end))
                return [{
                    "time": int(start.timestamp()),
                    "open": 1.1,
                    "high": 1.2,
                    "low": 1.0,
                    "close": 1.15,
                    "tick_volume": 10,
                    "spread": 12,
                }]
            def last_error(self): return "ok"

        fake = FakeMt5()
        client = Mt5Client(BrokerSettings(), module=fake)
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 3, 7, tzinfo=timezone.utc)

        candles = client.historical_candles("EUR_USD", "15m", start, end)

        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(len(candles), 3)
        self.assertLessEqual(
            max((call_end - call_start).days for call_start, call_end in fake.calls),
            30,
        )

    def test_mt5_historical_ticks_prime_old_history_and_retry_range(self) -> None:
        class FakeMt5:
            COPY_TICKS_INFO = 1
            COPY_TICKS_ALL = 2

            def __init__(self) -> None:
                self.range_calls = 0
                self.from_calls: list[tuple[int, int]] = []
                self.error = (-1, "Terminal: Call failed")

            def initialize(self, *args, **kwargs): return True
            def account_info(self): return SimpleNamespace(currency="USD", trade_mode=0)
            def symbol_info(self, symbol): return SimpleNamespace(name=symbol, visible=True)
            def copy_ticks_range(self, symbol, start, end, flags):
                self.range_calls += 1
                if self.range_calls == 1:
                    return None
                return [
                    {"time_msc": 1_700_000_000_000, "bid": 1.1000, "ask": 1.1002},
                    {"time_msc": 1_700_000_001_000, "bid": 1.1001, "ask": 1.1003},
                ]
            def copy_ticks_from(self, symbol, start, count, flags):
                self.from_calls.append((count, flags))
                return [{"time_msc": 1_700_000_000_000, "bid": 1.1000, "ask": 1.1002}]
            def last_error(self): return self.error

        fake = FakeMt5()
        client = Mt5Client(BrokerSettings(), module=fake)
        start = datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)
        end = datetime.fromtimestamp(1_700_000_001, tz=timezone.utc)

        ticks = client.historical_ticks("EUR_USD", start, end)

        self.assertEqual(fake.range_calls, 2)
        self.assertEqual(fake.from_calls, [(1_000, fake.COPY_TICKS_ALL)])
        self.assertEqual(len(ticks), 2)
        self.assertAlmostEqual(ticks.iloc[0]["bid"], 1.1000)

    def test_mt5_historical_ticks_fallback_to_copy_ticks_from_window(self) -> None:
        class FakeMt5:
            COPY_TICKS_INFO = 1
            COPY_TICKS_ALL = 2

            def __init__(self) -> None:
                self.range_calls = 0
                self.from_flags: list[int] = []

            def initialize(self, *args, **kwargs): return True
            def account_info(self): return SimpleNamespace(currency="USD", trade_mode=0)
            def symbol_info(self, symbol): return SimpleNamespace(name=symbol, visible=True)
            def copy_ticks_range(self, symbol, start, end, flags):
                self.range_calls += 1
                return None
            def copy_ticks_from(self, symbol, start, count, flags):
                self.from_flags.append(flags)
                if flags == self.COPY_TICKS_ALL:
                    return [
                        {"time_msc": 1_700_000_000_000, "bid": 1.1000, "ask": 1.1002},
                    ]
                return [
                    {"time_msc": 1_700_000_000_000, "bid": 1.1000, "ask": 1.1002},
                    {"time_msc": 1_700_000_001_000, "bid": 1.1001, "ask": 1.1003},
                ]
            def last_error(self): return (-1, "Terminal: Call failed")

        fake = FakeMt5()
        client = Mt5Client(BrokerSettings(), module=fake)
        start = datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)
        end = datetime.fromtimestamp(1_700_000_001, tz=timezone.utc)

        ticks = client.historical_ticks("EUR_USD", start, end)

        self.assertEqual(fake.range_calls, 2)
        self.assertEqual(fake.from_flags[0], fake.COPY_TICKS_ALL)
        self.assertIn(fake.COPY_TICKS_INFO, fake.from_flags[1:])
        self.assertEqual(len(ticks), 2)
        self.assertAlmostEqual(ticks.iloc[-1]["ask"], 1.1003)

    def test_normalize_rejects_malformed_fx_symbol(self) -> None:
        with self.assertRaises(ValueError):
            normalize_instrument_name("EUR")

    def test_mt5_comment_is_short_safe_marker(self) -> None:
        comment = _mt5_comment("fxft-GBPUSD-full-1234567890abcdef", "LONG GBP_USD full")

        self.assertLessEqual(len(comment), 20)
        self.assertEqual(comment, "fxft-GBPUSD-full-123")

    def test_mt5_uses_order_check_to_select_accepted_fill_mode_for_demo_symbols(self) -> None:
        class FakeMt5:
            ORDER_FILLING_RETURN = 2
            ORDER_FILLING_IOC = 1
            ORDER_FILLING_FOK = 0
            ORDER_TYPE_BUY = 0
            ORDER_TYPE_SELL = 1
            ORDER_TIME_GTC = 0
            TRADE_ACTION_DEAL = 1
            TRADE_RETCODE_DONE = 10009
            TRADE_RETCODE_PLACED = 10008
            TRADE_RETCODE_DONE_PARTIAL = 10010

            def __init__(self) -> None:
                self.checked: list[dict[str, object]] = []
                self.sent: list[dict[str, object]] = []
                self.connected = False

            def initialize(self, *args, **kwargs):
                self.connected = True
                return True

            def account_info(self):
                return SimpleNamespace(currency="USD", trade_mode=0, login=12345678, server="Demo")

            def symbol_info(self, symbol: str):
                return SimpleNamespace(
                    name=symbol,
                    visible=True,
                    digits=5,
                    point=0.00001,
                    trade_contract_size=100_000,
                    volume_min=0.01,
                    volume_max=50,
                    volume_step=0.01,
                    trade_stops_level=20,
                )

            def symbol_info_tick(self, symbol: str):
                return SimpleNamespace(bid=1.1000, ask=1.1001, time=1_700_000_000)

            def order_check(self, request: dict[str, object]):
                self.checked.append(request.copy())
                if request.get("type_filling") == self.ORDER_FILLING_IOC:
                    return {"retcode": self.TRADE_RETCODE_DONE}
                return {"retcode": 10015}

            def order_send(self, request: dict[str, object]):
                self.sent.append(request.copy())
                return {"retcode": self.TRADE_RETCODE_DONE, "order": 42, "deal": 84, "price": 1.1001}

            def last_error(self) -> str:
                return "demo symbol only accepts IOC fill"

        settings = BrokerSettings(order_filling="RETURN")
        client = Mt5Client(settings=settings, module=FakeMt5())
        instrument = FxInstrument.from_mt5(
            "EUR_USD",
            SimpleNamespace(
                name="EURUSD",
                digits=5,
                point=0.00001,
                trade_contract_size=100_000,
                volume_min=0.01,
                volume_max=50,
                volume_step=0.01,
                trade_stops_level=20,
            ),
            account_leverage=30,
            broker_symbol="EURUSD",
        )

        response = client.create_market_order(
            instrument=instrument,
            signed_units=1000,
            stop_loss=1.0990,
            take_profit=1.1020,
            client_order_id="demo-fill-mode",
            comment="unit test",
        )

        self.assertEqual(response["mt5"]["order"], 42)
        self.assertEqual(client._module().checked[0]["type_filling"], client._module().ORDER_FILLING_RETURN)
        self.assertEqual(client._module().checked[1]["type_filling"], client._module().ORDER_FILLING_IOC)
        self.assertEqual(client._module().sent[0]["type_filling"], client._module().ORDER_FILLING_IOC)

    def test_mt5_journals_position_ticket_not_opening_order_ticket(self) -> None:
        class FakeMt5:
            ORDER_FILLING_RETURN = 2
            ORDER_TYPE_BUY = 0
            ORDER_TIME_GTC = 0
            TRADE_ACTION_DEAL = 1
            TRADE_RETCODE_DONE = 10009

            def initialize(self, *args, **kwargs): return True
            def account_info(self): return SimpleNamespace(currency="USD", trade_mode=0)
            def symbol_info(self, symbol): return SimpleNamespace(name=symbol, visible=True, digits=5, point=0.00001, trade_contract_size=100000, volume_min=0.01, volume_max=50, volume_step=0.01, trade_stops_level=20)
            def symbol_info_tick(self, symbol): return SimpleNamespace(bid=1.1, ask=1.1001, time=1700000000)
            def order_check(self, request): return {"retcode": self.TRADE_RETCODE_DONE}
            def order_send(self, request): return {"retcode": self.TRADE_RETCODE_DONE, "order": 42, "deal": 84, "price": 1.1001}
            def history_deals_get(self, *, ticket): return [SimpleNamespace(position_id=9001)]
            def last_error(self): return "ok"

        client = Mt5Client(BrokerSettings(), module=FakeMt5())
        instrument = FxInstrument.from_mt5("EUR_USD", SimpleNamespace(name="EURUSD", digits=5, point=0.00001, trade_contract_size=100000, volume_min=0.01, volume_max=50, volume_step=0.01, trade_stops_level=20), account_leverage=30, broker_symbol="EURUSD")
        response = client.create_market_order(instrument=instrument, signed_units=1000, stop_loss=1.099, take_profit=1.102, client_order_id="position-id-test", comment="test")
        self.assertEqual(response["orderFillTransaction"]["tradeOpened"]["tradeID"], "9001")

    def test_mt5_recovers_closed_position_from_legacy_order_ticket(self) -> None:
        class FakeMt5:
            DEAL_ENTRY_IN = 0
            DEAL_ENTRY_OUT = 1
            DEAL_ENTRY_INOUT = 2
            DEAL_ENTRY_OUT_BY = 3
            DEAL_TYPE_BUY = 0
            DEAL_TYPE_SELL = 1

            def initialize(self, *args, **kwargs): return True
            def account_info(self): return SimpleNamespace(currency="USD", trade_mode=0)
            def history_orders_get(self, *, ticket):
                return [SimpleNamespace(position_id=9001)] if ticket == 42 else []
            def history_deals_get(self, *args, **kwargs):
                if "ticket" in kwargs:
                    return []
                if kwargs.get("position") == 9001:
                    return [
                        SimpleNamespace(position_id=9001, entry=0, symbol="EURUSD", type=0, volume=0.01, time=1767708000, price=1.1, profit=0, commission=-0.1, fee=0, swap=0),
                        SimpleNamespace(position_id=9001, entry=1, symbol="EURUSD", type=1, volume=0.01, time=1767708300, price=1.101, profit=10, commission=-0.1, fee=0, swap=-0.2),
                    ]
                return []
            def symbol_info(self, symbol): return SimpleNamespace(trade_contract_size=100000)
            def last_error(self): return "ok"

        client = Mt5Client(BrokerSettings(), module=FakeMt5())
        event = client.closed_trade_for_references(
            ["42"],
            datetime.fromtimestamp(1767707000, tz=timezone.utc),
            datetime.fromtimestamp(1767709000, tz=timezone.utc),
        )

        self.assertIsNotNone(event)
        self.assertEqual(event["broker_trade_id"], "9001")
        self.assertEqual(event["side"], "LONG")
        self.assertEqual(event["units"], 1000)
        self.assertAlmostEqual(event["realized_pl"], 9.8)

    def test_mt5_accepts_order_check_retcodes_for_valid_demo_fill_mode(self) -> None:
        class FakeMt5:
            ORDER_FILLING_RETURN = 2
            ORDER_FILLING_IOC = 1
            ORDER_FILLING_FOK = 0
            ORDER_TYPE_BUY = 0
            ORDER_TIME_GTC = 0
            TRADE_ACTION_DEAL = 1
            TRADE_RETCODE_DONE = 10009
            TRADE_RETCODE_PLACED = 10008
            TRADE_RETCODE_DONE_PARTIAL = 10010

            def initialize(self, *args, **kwargs):
                return True

            def account_info(self):
                return SimpleNamespace(currency="USD", trade_mode=0, login=12345678, server="Demo")

            def symbol_info(self, symbol: str):
                return SimpleNamespace(
                    name=symbol,
                    visible=True,
                    digits=5,
                    point=0.00001,
                    trade_contract_size=100_000,
                    volume_min=0.01,
                    volume_max=50,
                    volume_step=0.01,
                    trade_stops_level=20,
                )

            def symbol_info_tick(self, symbol: str):
                return SimpleNamespace(bid=1.1000, ask=1.1001, time=1_700_000_000)

            def order_check(self, request: dict[str, object]):
                if request.get("type_filling") == self.ORDER_FILLING_FOK:
                    return {"retcode": 0}
                return {"retcode": 10015}

            def order_send(self, request: dict[str, object]):
                return {"retcode": 10009, "order": 7, "deal": 9, "price": 1.1001}

            def last_error(self) -> str:
                return "Success"

        settings = BrokerSettings(order_filling="RETURN")
        client = Mt5Client(settings=settings, module=FakeMt5())
        instrument = FxInstrument.from_mt5(
            "EUR_USD",
            SimpleNamespace(
                name="EURUSD",
                digits=5,
                point=0.00001,
                trade_contract_size=100_000,
                volume_min=0.01,
                volume_max=50,
                volume_step=0.01,
                trade_stops_level=20,
            ),
            account_leverage=30,
            broker_symbol="EURUSD",
        )

        result = client.create_market_order(
            instrument=instrument,
            signed_units=1000,
            stop_loss=1.0990,
            take_profit=1.1020,
            client_order_id="demo-fok",
            comment="unit test",
        )

        self.assertEqual(result["mt5"]["order"], 7)
        self.assertEqual(result["mt5"]["retcode"], 10009)


if __name__ == "__main__":
    unittest.main()
