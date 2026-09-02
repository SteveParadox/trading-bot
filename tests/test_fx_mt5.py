from __future__ import annotations

import unittest
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
