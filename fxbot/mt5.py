"""MetaTrader 5 terminal client for FX forward testing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from fxbot.config import BrokerSettings
from fxbot.instruments import FxInstrument, PriceSnapshot, normalize_instrument_name, split_instrument_name
from fxbot.models import Side

try:  # The package is Windows-terminal backed, so keep imports lazy for tests/docs.
    import MetaTrader5 as _MT5_MODULE
except ImportError:  # pragma: no cover - exercised only on hosts without MT5.
    _MT5_MODULE = None


TIMEFRAME_TO_MT5 = {
    "5m": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
}


class Mt5Error(RuntimeError):
    pass


class Mt5CredentialsMissing(Mt5Error):
    pass


@dataclass(frozen=True)
class PricingResponse:
    prices: dict[str, PriceSnapshot]
    conversion_rates: dict[str, float]
    raw: dict[str, Any]


class Mt5Client:
    def __init__(self, settings: BrokerSettings, *, module: Any | None = None) -> None:
        self.settings = settings
        self._mt5 = module
        self._connected = False

    @property
    def configured(self) -> bool:
        return self.settings.configured

    def account_summary(self) -> dict[str, Any]:
        self._ensure_connected()
        mt5 = self._module()
        account = mt5.account_info()
        if account is None:
            raise Mt5CredentialsMissing(f"MT5 account_info failed: {mt5.last_error()}")
        payload = _as_dict(account)
        positions = self._positions()
        return {
            "NAV": _safe_float(payload.get("equity"), _safe_float(payload.get("balance"))),
            "balance": _safe_float(payload.get("balance")),
            "marginUsed": _safe_float(payload.get("margin")),
            "openPositionCount": len(positions),
            "currency": str(payload.get("currency") or "USD").upper(),
            "positionValue": 0.0,
            "lastTransactionID": "",
            "login": str(payload.get("login") or ""),
            "server": str(payload.get("server") or self.settings.server),
            "trade_mode": payload.get("trade_mode"),
            "leverage": payload.get("leverage"),
            "margin_free": payload.get("margin_free"),
            "profit": payload.get("profit"),
        }

    def instruments(self, names: list[str]) -> dict[str, FxInstrument]:
        self._ensure_connected()
        leverage = _safe_float(_as_dict(self._module().account_info()).get("leverage"), 30.0)
        instruments: dict[str, FxInstrument] = {}
        for raw_name in names:
            name = normalize_instrument_name(raw_name)
            broker_symbol = self.settings.broker_symbol_for(name)
            symbol_info = self._select_symbol(broker_symbol)
            instruments[name] = FxInstrument.from_mt5(
                name,
                symbol_info,
                account_leverage=leverage,
                broker_symbol=broker_symbol,
            )
        return instruments

    def pricing(self, instruments: list[str]) -> PricingResponse:
        self._ensure_connected()
        prices: dict[str, PriceSnapshot] = {}
        raw_prices: list[dict[str, Any]] = []
        for raw_name in instruments:
            name = normalize_instrument_name(raw_name)
            snapshot = self._price_snapshot(name)
            prices[name] = snapshot
            raw_prices.append({"instrument": name, "broker_symbol": self.settings.broker_symbol_for(name), **_snapshot_payload(snapshot)})
        conversions = self._conversion_rates(prices)
        return PricingResponse(prices=prices, conversion_rates=conversions, raw={"prices": raw_prices, "conversion_rates": conversions})

    def candles(self, instrument: str, timeframe: str, count: int) -> pd.DataFrame:
        self._ensure_connected()
        mt5 = self._module()
        timeframe_value = self._timeframe(timeframe)
        name = normalize_instrument_name(instrument)
        symbol = self.settings.broker_symbol_for(name)
        self._select_symbol(symbol)
        rates = mt5.copy_rates_from_pos(symbol, timeframe_value, 1, count)
        if rates is None:
            raise Mt5Error(f"MT5 copy_rates_from_pos failed for {symbol}: {mt5.last_error()}")
        frame = pd.DataFrame(rates)
        if frame.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame["timestamp"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        volume_column = "tick_volume" if "tick_volume" in frame.columns else "real_volume"
        frame["volume"] = pd.to_numeric(frame.get(volume_column, 0), errors="coerce")
        for column in ["open", "high", "low", "close"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=["open", "high", "low", "close"]).set_index("timestamp")[
            ["open", "high", "low", "close", "volume"]
        ].sort_index()

    def open_positions(self) -> list[dict[str, Any]]:
        self._ensure_connected()
        grouped: dict[str, dict[str, Any]] = {}
        for position in self._positions():
            raw = _as_dict(position)
            symbol = str(raw.get("symbol") or "")
            if not symbol:
                continue
            name = self.settings.strategy_symbol_for(symbol)
            contract_size = self._contract_size(symbol)
            signed_units = self._position_signed_units(raw, contract_size)
            if signed_units == 0:
                continue
            bucket = grouped.setdefault(
                name,
                {
                    "instrument": name,
                    "broker_symbol": symbol,
                    "long": {"units": 0.0, "averagePrice": 0.0},
                    "short": {"units": 0.0, "averagePrice": 0.0},
                    "unrealizedPL": 0.0,
                    "marginUsed": 0.0,
                    "positions": [],
                },
            )
            side_key = "long" if signed_units > 0 else "short"
            _merge_side(bucket[side_key], signed_units, _safe_float(raw.get("price_open")))
            bucket["unrealizedPL"] += _safe_float(raw.get("profit"))
            bucket["marginUsed"] += self._position_margin(raw)
            bucket["positions"].append(raw)
        return list(grouped.values())

    def open_trades(self) -> list[dict[str, Any]]:
        self._ensure_connected()
        trades: list[dict[str, Any]] = []
        for position in self._positions():
            raw = _as_dict(position)
            symbol = str(raw.get("symbol") or "")
            if not symbol:
                continue
            name = self.settings.strategy_symbol_for(symbol)
            contract_size = self._contract_size(symbol)
            signed_units = self._position_signed_units(raw, contract_size)
            if signed_units == 0:
                continue
            stop_loss = _safe_float(raw.get("sl"))
            take_profit = _safe_float(raw.get("tp"))
            trades.append(
                {
                    "id": self._position_id(raw),
                    "instrument": name,
                    "broker_symbol": symbol,
                    "currentUnits": signed_units,
                    "initialUnits": signed_units,
                    "openTime": _mt5_time_to_datetime(raw.get("time")),
                    "price": _safe_float(raw.get("price_open")),
                    "realizedPL": 0.0,
                    "financing": _safe_float(raw.get("swap")),
                    "unrealizedPL": _safe_float(raw.get("profit")),
                    "stopLossOrder": {"price": stop_loss} if stop_loss > 0 else None,
                    "takeProfitOrder": {"price": take_profit} if take_profit > 0 else None,
                    "mt5": raw,
                }
            )
        return trades

    def closed_trades_since(self, since: datetime, until: datetime | None = None) -> list[dict[str, Any]]:
        self._ensure_connected()
        mt5 = self._module()
        end = until or datetime.now(timezone.utc)
        deals = mt5.history_deals_get(_naive_utc(since), _naive_utc(end))
        if deals is None:
            raise Mt5Error(f"MT5 history_deals_get failed: {mt5.last_error()}")
        return self._closed_trade_events(deals)

    def order_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        self._ensure_connected()
        mt5 = self._module()
        marker = _mt5_comment(client_order_id, "")
        for getter_name in ("positions_get", "orders_get"):
            getter = getattr(mt5, getter_name, None)
            if getter is None:
                continue
            rows = getter()
            if not rows:
                continue
            for row in rows:
                payload = _as_dict(row)
                if str(payload.get("comment") or "").startswith(marker):
                    return {"id": str(payload.get("ticket") or payload.get("order") or ""), "mt5": payload}
        return None

    def create_market_order(
        self,
        *,
        instrument: FxInstrument,
        signed_units: float,
        stop_loss: float,
        take_profit: float,
        client_order_id: str,
        comment: str,
    ) -> dict[str, Any]:
        self._ensure_connected()
        mt5 = self._module()
        symbol = instrument.broker_symbol or self.settings.broker_symbol_for(instrument.name)
        self._select_symbol(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise Mt5Error(f"MT5 symbol_info_tick failed for {symbol}: {mt5.last_error()}")
        tick_payload = _as_dict(tick)
        side = Side.LONG if signed_units > 0 else Side.SHORT
        price = _safe_float(tick_payload.get("ask" if side is Side.LONG else "bid"))
        volume = instrument.units_to_volume(abs(signed_units))
        if volume <= 0 or (instrument.volume_min and volume < instrument.volume_min):
            raise Mt5Error(f"{instrument.name} MT5 volume {volume} is below broker minimum")
        request = {
            "action": _constant(mt5, "TRADE_ACTION_DEAL", 1),
            "symbol": symbol,
            "volume": volume,
            "type": _constant(mt5, "ORDER_TYPE_BUY", 0) if side is Side.LONG else _constant(mt5, "ORDER_TYPE_SELL", 1),
            "price": price,
            "sl": instrument.round_price(stop_loss),
            "tp": instrument.round_price(take_profit),
            "deviation": self.settings.deviation_points,
            "magic": self.settings.magic_number,
            "comment": _mt5_comment(client_order_id, comment),
            "type_time": _constant(mt5, "ORDER_TIME_GTC", 0),
        }
        request["type_filling"] = self._validated_order_filling(request)
        result = mt5.order_send(request)
        result_payload = self._checked_result(result, "order_send")
        return _market_order_response(
            client_order_id=client_order_id,
            instrument=instrument.name,
            signed_units=signed_units,
            price=_safe_float(result_payload.get("price"), price),
            order_id=str(result_payload.get("order") or ""),
            deal_id=str(result_payload.get("deal") or ""),
            result=result_payload,
        )

    def set_trade_dependent_orders(
        self,
        *,
        trade_id: str,
        instrument: FxInstrument,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        self._ensure_connected()
        mt5 = self._module()
        symbol = instrument.broker_symbol or self.settings.broker_symbol_for(instrument.name)
        request = {
            "action": _constant(mt5, "TRADE_ACTION_SLTP", 6),
            "position": int(trade_id),
            "symbol": symbol,
            "magic": self.settings.magic_number,
            "comment": "fxft-sltp",
        }
        if stop_loss is not None:
            request["sl"] = instrument.round_price(stop_loss)
        if take_profit is not None:
            request["tp"] = instrument.round_price(take_profit)
        result = mt5.order_send(request)
        return {"mt5": self._checked_result(result, "order_send SLTP"), "request": request}

    def shutdown(self) -> None:
        if self._connected:
            self._module().shutdown()
            self._connected = False

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        mt5 = self._module()
        kwargs: dict[str, Any] = {"timeout": self.settings.timeout_ms, "portable": self.settings.portable}
        if self.settings.login is not None:
            kwargs["login"] = int(self.settings.login)
        if self.settings.password:
            kwargs["password"] = self.settings.password
        if self.settings.server:
            kwargs["server"] = self.settings.server
        ok = mt5.initialize(self.settings.terminal_path, **kwargs) if self.settings.terminal_path else mt5.initialize(**kwargs)
        if not ok:
            raise Mt5CredentialsMissing(f"MT5 initialize failed: {mt5.last_error()}")
        account = mt5.account_info()
        if account is None:
            raise Mt5CredentialsMissing(f"MT5 account_info failed after initialize: {mt5.last_error()}")
        self._assert_demo_account(account)
        self._connected = True

    def _assert_demo_account(self, account: Any) -> None:
        if not self.settings.demo_only:
            return
        mt5 = self._module()
        trade_mode = _safe_int(_as_dict(account).get("trade_mode"), -1)
        real_mode = _constant(mt5, "ACCOUNT_TRADE_MODE_REAL", 2)
        if trade_mode == real_mode:
            raise Mt5CredentialsMissing("MT5_DEMO_ONLY=true refuses to run against a real MT5 account")

    def _select_symbol(self, symbol: str) -> Any:
        mt5 = self._module()
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise Mt5Error(f"MT5 symbol {symbol!r} was not found")
        payload = _as_dict(symbol_info)
        if not payload.get("visible", True) and not mt5.symbol_select(symbol, True):
            raise Mt5Error(f"MT5 symbol_select failed for {symbol}: {mt5.last_error()}")
        return mt5.symbol_info(symbol) or symbol_info

    def _price_snapshot(self, instrument: str) -> PriceSnapshot:
        mt5 = self._module()
        symbol = self.settings.broker_symbol_for(instrument)
        self._select_symbol(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise Mt5Error(f"MT5 symbol_info_tick failed for {symbol}: {mt5.last_error()}")
        return PriceSnapshot.from_mt5(instrument, tick)

    def _conversion_rates(self, prices: dict[str, PriceSnapshot]) -> dict[str, float]:
        account = self._account_currency()
        conversions: dict[str, float] = {}
        for name, snapshot in prices.items():
            base, quote = split_instrument_name(name)
            if quote == account:
                conversions[base] = snapshot.mid
            elif base == account and snapshot.mid > 0:
                conversions[quote] = 1.0 / snapshot.mid
        needed = {
            split_instrument_name(name)[1]
            for name in prices
            if account not in split_instrument_name(name)
        }
        for currency in needed:
            if currency == account or currency in conversions:
                continue
            direct = self._mid_for_pair(currency, account)
            if direct:
                conversions[currency] = direct
                continue
            reverse = self._mid_for_pair(account, currency)
            if reverse:
                conversions[currency] = 1.0 / reverse
        return conversions

    def _mid_for_pair(self, base: str, quote: str) -> float | None:
        try:
            snapshot = self._price_snapshot(f"{base}_{quote}")
        except (Mt5Error, ValueError):
            return None
        return snapshot.mid if snapshot.mid > 0 else None

    def _positions(self) -> tuple[Any, ...]:
        mt5 = self._module()
        positions = mt5.positions_get()
        if positions is None:
            raise Mt5Error(f"MT5 positions_get failed: {mt5.last_error()}")
        return tuple(positions)

    def _position_signed_units(self, payload: dict[str, Any], contract_size: float) -> float:
        mt5 = self._module()
        position_type = _safe_int(payload.get("type"), _constant(mt5, "POSITION_TYPE_BUY", 0))
        is_buy = position_type == _constant(mt5, "POSITION_TYPE_BUY", 0)
        units = _safe_float(payload.get("volume")) * contract_size
        return units if is_buy else -units

    def _position_margin(self, payload: dict[str, Any]) -> float:
        mt5 = self._module()
        symbol = str(payload.get("symbol") or "")
        volume = _safe_float(payload.get("volume"))
        price = _safe_float(payload.get("price_open") or payload.get("price_current"))
        if not symbol or volume <= 0 or price <= 0:
            return 0.0
        order_type = _constant(mt5, "ORDER_TYPE_BUY", 0) if self._position_signed_units(payload, 1.0) > 0 else _constant(mt5, "ORDER_TYPE_SELL", 1)
        try:
            margin = mt5.order_calc_margin(order_type, symbol, volume, price)
        except Exception:
            return 0.0
        return _safe_float(margin)

    def _contract_size(self, symbol: str) -> float:
        symbol_info = self._select_symbol(symbol)
        return _safe_float(_as_dict(symbol_info).get("trade_contract_size"), 100_000.0)

    def _position_id(self, payload: dict[str, Any]) -> str:
        return str(payload.get("ticket") or payload.get("identifier") or "")

    def _timeframe(self, timeframe: str) -> int:
        mt5 = self._module()
        constant_name = TIMEFRAME_TO_MT5.get(timeframe)
        if constant_name is None:
            raise ValueError(f"unsupported MT5 candle timeframe: {timeframe}")
        return _constant(mt5, constant_name, None)

    def _order_filling(self) -> int:
        return _constant(self._module(), f"ORDER_FILLING_{self.settings.order_filling}", _constant(self._module(), "ORDER_FILLING_RETURN", 2))

    def _validated_order_filling(self, request: dict[str, Any]) -> int:
        mt5 = self._module()
        ok_codes = {
            _constant(mt5, "TRADE_RETCODE_DONE", 10009),
            _constant(mt5, "TRADE_RETCODE_PLACED", 10008),
            _constant(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010),
            0,
        }
        candidates = []
        for value in (
            self._order_filling(),
            _constant(mt5, "ORDER_FILLING_IOC", 1),
            _constant(mt5, "ORDER_FILLING_FOK", 0),
            _constant(mt5, "ORDER_FILLING_RETURN", 2),
        ):
            if value not in candidates:
                candidates.append(value)
        for filling in candidates:
            request["type_filling"] = filling
            result = mt5.order_check(request)
            if result is None:
                continue
            payload = _as_dict(result)
            retcode = _safe_int(payload.get("retcode"), -1)
            if retcode in ok_codes:
                return filling
        raise Mt5Error(
            f"MT5 order_check rejected all supported fill modes for {request.get('symbol')!r}: "
            f"{candidates}"
        )

    def _account_currency(self) -> str:
        account = self._module().account_info()
        if account is None:
            return "USD"
        return str(_as_dict(account).get("currency") or "USD").upper()

    def _checked_result(self, result: Any, operation: str) -> dict[str, Any]:
        if result is None:
            raise Mt5Error(f"MT5 {operation} returned no result: {self._module().last_error()}")
        payload = _as_dict(result)
        retcode = _safe_int(payload.get("retcode"), -1)
        ok_codes = {
            _constant(self._module(), "TRADE_RETCODE_DONE", 10009),
            _constant(self._module(), "TRADE_RETCODE_PLACED", 10008),
            _constant(self._module(), "TRADE_RETCODE_DONE_PARTIAL", 10010),
        }
        if retcode not in ok_codes:
            raise Mt5Error(f"MT5 {operation} failed with retcode {retcode}: {payload}")
        return payload

    def _closed_trade_events(self, deals: tuple[Any, ...]) -> list[dict[str, Any]]:
        mt5 = self._module()
        out_entries = {
            _constant(mt5, "DEAL_ENTRY_OUT", 1),
            _constant(mt5, "DEAL_ENTRY_INOUT", 2),
            _constant(mt5, "DEAL_ENTRY_OUT_BY", 3),
        }
        buy_deal = _constant(mt5, "DEAL_TYPE_BUY", 0)
        grouped: dict[str, dict[str, Any]] = {}
        for deal in deals:
            payload = _as_dict(deal)
            if _safe_int(payload.get("entry"), -1) not in out_entries:
                continue
            symbol = str(payload.get("symbol") or "")
            if not symbol:
                continue
            name = self.settings.strategy_symbol_for(symbol)
            contract_size = self._contract_size(symbol)
            trade_id = str(payload.get("position_id") or payload.get("order") or payload.get("ticket") or "")
            if not trade_id:
                continue
            event = grouped.setdefault(
                trade_id,
                {
                    "broker_trade_id": trade_id,
                    "instrument": name,
                    "side": Side.SHORT.value if _safe_int(payload.get("type"), buy_deal) == buy_deal else Side.LONG.value,
                    "units": 0.0,
                    "exit_time": _mt5_time_to_datetime(payload.get("time")),
                    "exit_price": _safe_float(payload.get("price")),
                    "realized_pl": 0.0,
                    "financing": 0.0,
                    "exit_reason": "mt5_history_deal",
                    "deals": [],
                },
            )
            event["units"] += _safe_float(payload.get("volume")) * contract_size
            event["realized_pl"] += (
                _safe_float(payload.get("profit"))
                + _safe_float(payload.get("commission"))
                + _safe_float(payload.get("fee"))
            )
            event["financing"] += _safe_float(payload.get("swap"))
            deal_time = _mt5_time_to_datetime(payload.get("time"))
            if deal_time and (event["exit_time"] is None or deal_time > event["exit_time"]):
                event["exit_time"] = deal_time
                event["exit_price"] = _safe_float(payload.get("price"), event["exit_price"])
            event["deals"].append(payload)
        return list(grouped.values())

    def _module(self) -> Any:
        if self._mt5 is not None:
            return self._mt5
        if _MT5_MODULE is None:
            raise Mt5CredentialsMissing("MetaTrader5 package is not installed; run pip install -r requirements.txt")
        self._mt5 = _MT5_MODULE
        return self._mt5


def extract_order_ids(response: dict[str, Any]) -> tuple[str | None, str | None]:
    fill = response.get("orderFillTransaction") or {}
    create = response.get("orderCreateTransaction") or {}
    trade_opened = fill.get("tradeOpened") or {}
    order_id = str(create.get("id") or fill.get("orderID") or "") or None
    trade_id = str(trade_opened.get("tradeID") or "") or None
    return order_id, trade_id


def _market_order_response(
    *,
    client_order_id: str,
    instrument: str,
    signed_units: float,
    price: float,
    order_id: str,
    deal_id: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    trade_id = order_id or deal_id
    now = datetime.now(timezone.utc).isoformat()
    return {
        "orderCreateTransaction": {
            "id": order_id,
            "clientExtensions": {"id": client_order_id},
        },
        "orderFillTransaction": {
            "orderID": order_id,
            "id": deal_id,
            "instrument": instrument,
            "units": signed_units,
            "price": price,
            "time": now,
            "tradeOpened": {"tradeID": trade_id, "units": signed_units},
        },
        "mt5": result,
    }


def _merge_side(side_payload: dict[str, Any], signed_units: float, price: float) -> None:
    existing_units = _safe_float(side_payload.get("units"))
    existing_abs = abs(existing_units)
    added_abs = abs(signed_units)
    total_abs = existing_abs + added_abs
    if total_abs <= 0:
        return
    side_payload["averagePrice"] = (
        (_safe_float(side_payload.get("averagePrice")) * existing_abs) + (price * added_abs)
    ) / total_abs
    side_payload["units"] = existing_units + signed_units


def _snapshot_payload(snapshot: PriceSnapshot) -> dict[str, Any]:
    return {
        "bid": snapshot.bid,
        "ask": snapshot.ask,
        "time": snapshot.time.isoformat(),
        "quote_to_home_factor": snapshot.quote_to_home_factor,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "_asdict"):
        return {str(key): _jsonable(item) for key, item in value._asdict().items()}
    return {
        name: _jsonable(getattr(value, name))
        for name in dir(value)
        if not name.startswith("_") and not callable(getattr(value, name))
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "_asdict"):
        return _as_dict(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _constant(mt5: Any, name: str, default: int | None) -> int:
    value = getattr(mt5, name, default)
    if value is None:
        raise Mt5Error(f"MetaTrader5 constant {name} is unavailable")
    return int(value)


def _mt5_comment(client_order_id: str, comment: str) -> str:
    raw = client_order_id or comment or "fxft"
    safe = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
    return (safe or "fxft")[:20]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mt5_time_to_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)
