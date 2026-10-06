"""MetaTrader 5 terminal client for FX forward testing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
import math

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


class Mt5RejectedError(Mt5Error):
    """Order was definitively declined by the broker terminal (not placed)."""


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
            self._mark_disconnected()
            raise Mt5Error(f"MT5 account_info failed: {mt5.last_error()}")
        self._assert_demo_account(account)
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
            "hedging_enabled": payload.get("margin_mode") == _constant(mt5, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", 2),
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
            self._mark_disconnected()
            raise Mt5Error(f"MT5 copy_rates_from_pos failed for {symbol}: {mt5.last_error()}")
        frame = pd.DataFrame(rates)
        if frame.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame["timestamp"] = pd.to_datetime(
            frame["time"] + self.settings.time_offset_seconds,
            unit="s",
            utc=True,
        )
        volume_column = "tick_volume" if "tick_volume" in frame.columns else "real_volume"
        frame["volume"] = pd.to_numeric(frame.get(volume_column, 0), errors="coerce")
        for column in ["open", "high", "low", "close"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=["open", "high", "low", "close"]).set_index("timestamp")[
            ["open", "high", "low", "close", "volume"]
        ].sort_index()


    def historical_candles(
        self,
        instrument: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch historical MT5 bars for a corrected UTC interval.

        MT5 can reject very large copy_rates_range requests even when every
        smaller sub-range is valid. Fetch bounded chunks and de-duplicate the
        inclusive chunk boundaries before applying the configured broker-time
        correction.
        """

        self._ensure_connected()
        mt5 = self._module()
        timeframe_value = self._timeframe(timeframe)
        name = normalize_instrument_name(instrument)
        symbol = self.settings.broker_symbol_for(name)
        self._select_symbol(symbol)
        start_utc = _coerce_utc(start)
        end_utc = _coerce_utc(end)
        if end_utc <= start_utc:
            raise ValueError("historical candle end must be after start")

        offset = timedelta(seconds=self.settings.time_offset_seconds)
        request_start = start_utc - offset
        request_end = end_utc - offset
        chunk_size = timedelta(days=30)
        cursor = request_start
        frames: list[pd.DataFrame] = []

        while cursor < request_end:
            chunk_end = min(cursor + chunk_size, request_end)
            rates = mt5.copy_rates_range(symbol, timeframe_value, cursor, chunk_end)
            if rates is None:
                error = mt5.last_error()
                self._mark_disconnected()
                raise Mt5Error(
                    "MT5 copy_rates_range failed for "
                    f"{symbol} [{timeframe}] "
                    f"{cursor.isoformat()} -> {chunk_end.isoformat()}: {error}"
                )
            if len(rates):
                frames.append(pd.DataFrame(rates))
            cursor = chunk_end

        if not frames:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "spread_points"]
            )

        frame = pd.concat(frames, ignore_index=True)
        if "time" in frame.columns:
            frame = (
                frame.drop_duplicates(subset=["time"], keep="last")
                .sort_values("time")
                .reset_index(drop=True)
            )
        frame["timestamp"] = pd.to_datetime(
            frame["time"] + self.settings.time_offset_seconds,
            unit="s",
            utc=True,
        )
        volume_column = "tick_volume" if "tick_volume" in frame.columns else "real_volume"
        frame["volume"] = pd.to_numeric(frame.get(volume_column, 0), errors="coerce")
        frame["spread_points"] = pd.to_numeric(frame.get("spread", 0), errors="coerce")
        for column in ["open", "high", "low", "close"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        result = frame.dropna(subset=["open", "high", "low", "close"]).set_index("timestamp")[
            ["open", "high", "low", "close", "volume", "spread_points"]
        ].sort_index()
        return result.loc[
            (result.index >= pd.Timestamp(start_utc))
            & (result.index <= pd.Timestamp(end_utc))
        ]

    def historical_ticks(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch historical executable bid/ask ticks for one corrected UTC interval.

        Older MT5 tick history may exist on the broker server without being
        synchronized into the terminal's local tick database. A bounded range
        request can fail in that state even though copy_ticks_from succeeds.
        Prime the old history, retry the exact range, then fall back to bounded
        copy_ticks_from pagination before treating the request as failed.
        """

        self._ensure_connected()
        mt5 = self._module()
        name = normalize_instrument_name(instrument)
        symbol = self.settings.broker_symbol_for(name)
        self._select_symbol(symbol)
        start_utc = _coerce_utc(start)
        end_utc = _coerce_utc(end)
        if end_utc <= start_utc:
            raise ValueError("historical tick end must be after start")

        offset = timedelta(seconds=self.settings.time_offset_seconds)
        request_start = start_utc - offset
        request_end = end_utc - offset
        info_flags = _constant(
            mt5,
            "COPY_TICKS_INFO",
            _constant(mt5, "COPY_TICKS_ALL", 0),
        )
        all_flags = _constant(mt5, "COPY_TICKS_ALL", info_flags)

        ticks = mt5.copy_ticks_range(
            symbol,
            request_start,
            request_end,
            info_flags,
        )
        initial_error: Any | None = None
        retry_error: Any | None = None
        fallback_error: Any | None = None
        prime_count: int | None = None

        if ticks is None:
            initial_error = mt5.last_error()
            copy_ticks_from = getattr(mt5, "copy_ticks_from", None)

            if copy_ticks_from is not None:
                prime = copy_ticks_from(
                    symbol,
                    request_start,
                    1_000,
                    all_flags,
                )
                if prime is not None:
                    prime_count = len(prime)

            ticks = mt5.copy_ticks_range(
                symbol,
                request_start,
                request_end,
                info_flags,
            )

            if ticks is None:
                retry_error = mt5.last_error()
                ticks, fallback_error = self._copy_ticks_from_window(
                    symbol=symbol,
                    start=request_start,
                    end=request_end,
                    flags=info_flags,
                )

            if ticks is None:
                self._mark_disconnected()
                raise Mt5Error(
                    "MT5 historical tick request failed after synchronization "
                    f"for {symbol} "
                    f"{start_utc.isoformat()} -> {end_utc.isoformat()}; "
                    f"initial_error={initial_error}; "
                    f"retry_error={retry_error}; "
                    f"fallback_error={fallback_error}; "
                    f"prime_count={prime_count}"
                )

        frame = pd.DataFrame(ticks)
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "instrument", "bid", "ask"])
        if "time_msc" in frame.columns:
            raw_time = (
                pd.to_numeric(frame["time_msc"], errors="coerce")
                + self.settings.time_offset_seconds * 1000
            )
            frame["timestamp"] = pd.to_datetime(raw_time, unit="ms", utc=True)
        else:
            raw_time = (
                pd.to_numeric(frame["time"], errors="coerce")
                + self.settings.time_offset_seconds
            )
            frame["timestamp"] = pd.to_datetime(raw_time, unit="s", utc=True)
        frame["bid"] = pd.to_numeric(frame.get("bid"), errors="coerce")
        frame["ask"] = pd.to_numeric(frame.get("ask"), errors="coerce")
        frame["instrument"] = name
        result = frame.dropna(subset=["timestamp", "bid", "ask"])[
            ["timestamp", "instrument", "bid", "ask"]
        ]
        result = result[(result["bid"] > 0) & (result["ask"] > result["bid"])]
        result = result.sort_values("timestamp").drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        return result.loc[
            (result["timestamp"] >= pd.Timestamp(start_utc))
            & (result["timestamp"] <= pd.Timestamp(end_utc))
        ].reset_index(drop=True)

    def _copy_ticks_from_window(
        self,
        *,
        symbol: str,
        start: datetime,
        end: datetime,
        flags: int,
        batch_size: int = 50_000,
        max_batches: int = 32,
    ) -> tuple[pd.DataFrame | None, Any | None]:
        """Bounded fallback for MT5 terminals that reject old range requests."""

        mt5 = self._module()
        copy_ticks_from = getattr(mt5, "copy_ticks_from", None)
        if copy_ticks_from is None:
            return None, "copy_ticks_from unavailable"

        cursor = start
        frames: list[pd.DataFrame] = []

        for _ in range(max_batches):
            batch = copy_ticks_from(symbol, cursor, batch_size, flags)
            if batch is None:
                return None, mt5.last_error()

            frame = pd.DataFrame(batch)
            if frame.empty:
                break

            if "time_msc" in frame.columns:
                raw = pd.to_numeric(frame["time_msc"], errors="coerce")
                valid = raw.dropna()
                if valid.empty:
                    return None, "copy_ticks_from returned no valid time_msc values"
                raw_times = pd.to_datetime(raw, unit="ms", utc=True)
                last_raw = int(valid.iloc[-1])
                last_time = datetime.fromtimestamp(last_raw / 1000, tz=timezone.utc)
                next_cursor = datetime.fromtimestamp(
                    (last_raw + 1) / 1000,
                    tz=timezone.utc,
                )
            elif "time" in frame.columns:
                raw = pd.to_numeric(frame["time"], errors="coerce")
                valid = raw.dropna()
                if valid.empty:
                    return None, "copy_ticks_from returned no valid time values"
                raw_times = pd.to_datetime(raw, unit="s", utc=True)
                last_raw = int(valid.iloc[-1])
                last_time = datetime.fromtimestamp(last_raw, tz=timezone.utc)
                next_cursor = last_time + timedelta(seconds=1)
            else:
                return None, "copy_ticks_from returned ticks without time/time_msc"

            mask = (
                (raw_times >= pd.Timestamp(start))
                & (raw_times <= pd.Timestamp(end))
            )
            selected = frame.loc[mask]
            if not selected.empty:
                frames.append(selected)

            if last_time >= end:
                break
            if next_cursor <= cursor:
                return None, "copy_ticks_from pagination did not advance"
            cursor = next_cursor
        else:
            return None, (
                "copy_ticks_from exceeded bounded pagination "
                f"({max_batches} batches x {batch_size} ticks)"
            )

        if not frames:
            return pd.DataFrame(), None

        combined = pd.concat(frames, ignore_index=True)
        dedupe_columns = [
            column
            for column in ("time_msc", "time", "bid", "ask")
            if column in combined.columns
        ]
        if dedupe_columns:
            combined = combined.drop_duplicates(
                subset=dedupe_columns,
                keep="last",
            )
        return combined.reset_index(drop=True), None

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
                    "openTime": _mt5_time_to_datetime(
                        raw.get("time"), self.settings.time_offset_seconds
                    ),
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
            self._mark_disconnected()
            raise Mt5Error(f"MT5 history_deals_get failed: {mt5.last_error()}")
        events = self._closed_trade_events(deals)
        result = []
        for event in events:
            try:
                full_history = mt5.history_deals_get(position=int(event["broker_trade_id"]))
            except (TypeError, ValueError):
                full_history = None
            if full_history is not None:
                matching = [row for row in self._closed_trade_events(full_history)
                            if row["broker_trade_id"] == event["broker_trade_id"]]
                if matching:
                    event = matching[0]
            event["costs_complete"] = full_history is not None
            result.append(event)
        return result

    def closed_trade_for_references(
        self,
        references: list[str],
        since: datetime,
        until: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Recover one closed position from legacy order/deal identifiers.

        Worker versions prior to the position-ID fix sometimes persisted an
        opening order ticket as ``broker_trade_id``.  MT5 close deals are keyed
        by ``position_id``, so a normal time-window scan cannot relate those
        rows when the close-history window is missing or has moved on.  Resolve
        each known order/deal reference to its position, then ask MT5 directly
        for every deal belonging to that position.
        """
        self._ensure_connected()
        mt5 = self._module()
        position_ids: set[str] = set()
        for raw_reference in references:
            try:
                reference = int(str(raw_reference))
            except (TypeError, ValueError):
                continue
            orders_get = getattr(mt5, "history_orders_get", None)
            if orders_get is not None:
                try:
                    orders = orders_get(ticket=reference)
                except (TypeError, ValueError, AttributeError):
                    orders = ()
                for order in orders or ():
                    payload = _as_dict(order)
                    position_id = payload.get("position_id") or payload.get("position")
                    if position_id:
                        position_ids.add(str(position_id))
            deals_get = getattr(mt5, "history_deals_get", None)
            if deals_get is not None:
                try:
                    referenced_deals = deals_get(ticket=reference)
                except (TypeError, ValueError, AttributeError):
                    referenced_deals = ()
                for deal in referenced_deals or ():
                    position_id = _as_dict(deal).get("position_id")
                    if position_id:
                        position_ids.add(str(position_id))

        deals_get = getattr(mt5, "history_deals_get", None)
        if deals_get is None:
            return None
        end = until or datetime.now(timezone.utc)
        for position_id in position_ids:
            try:
                deals = deals_get(position=int(position_id))
            except (TypeError, ValueError, AttributeError):
                deals = None
            if deals is None:
                # Some terminal builds do not expose the named ``position``
                # selector. Retain a bounded date-range fallback.
                deals = deals_get(_naive_utc(since), _naive_utc(end))
                if deals is not None:
                    deals = tuple(
                        deal for deal in deals
                        if str(_as_dict(deal).get("position_id") or "") == position_id
                    )
            if deals is None:
                continue
            for event in self._closed_trade_events(tuple(deals)):
                if str(event.get("broker_trade_id") or "") == position_id:
                    return event
        return None

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
                    identifier = str(payload.get("ticket") or payload.get("order") or "")
                    return {"id": identifier, "mt5": payload,
                            "state": "filled" if getter_name == "positions_get" else "submitted",
                            "trade_id": identifier if getter_name == "positions_get" else None}
        return None

    def create_market_order(
        self,
        *,
        instrument: FxInstrument,
        signed_units: float,
        stop_loss: float,
        take_profit: float | None,
        client_order_id: str,
        comment: str,
        approved_entry_price: float | None = None,
        max_spread_price: float | None = None,
        max_quote_age_seconds: float | None = None,
    ) -> dict[str, Any]:
        self._ensure_connected()
        mt5 = self._module()
        account = mt5.account_info()
        if account is None:
            raise Mt5Error("MT5 account unavailable before order")
        self._assert_demo_account(account)
        if _as_dict(account).get("trade_allowed") is False or _as_dict(account).get("trade_expert") is False:
            raise Mt5RejectedError("MT5 account trading disabled")
        symbol = instrument.broker_symbol or self.settings.broker_symbol_for(instrument.name)
        self._select_symbol(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self._mark_disconnected()
            raise Mt5Error(f"MT5 symbol_info_tick failed for {symbol}: {mt5.last_error()}")
        tick_payload = _as_dict(tick)
        if max_quote_age_seconds is not None:
            snapshot = PriceSnapshot.from_mt5(instrument.name, tick, time_offset_seconds=self.settings.time_offset_seconds)
            age = (datetime.now(timezone.utc) - snapshot.time).total_seconds()
            if not 0 <= age <= max_quote_age_seconds:
                raise Mt5RejectedError("Stale or future MT5 order quote")
        side = Side.LONG if signed_units > 0 else Side.SHORT
        price = _safe_float(tick_payload.get("ask" if side is Side.LONG else "bid"))
        if not math.isfinite(stop_loss) or stop_loss <= 0 or (price - stop_loss) * side.sign <= 0:
            raise Mt5RejectedError("Required protective stop is invalid")
        if approved_entry_price is not None:
            bid, ask = _safe_float(tick_payload.get("bid")), _safe_float(tick_payload.get("ask"))
            if not all(math.isfinite(v) for v in (bid, ask, approved_entry_price)) or bid <= 0 or ask <= bid or (price - approved_entry_price) * side.sign > 0:
                raise Mt5RejectedError("Executable price moved beyond sniper risk approval")
            if max_spread_price is not None and ask - bid > max_spread_price:
                raise Mt5RejectedError("Spread moved beyond sniper execution approval")
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
            "tp": instrument.round_price(take_profit) if take_profit is not None else 0.0,
            "deviation": self.settings.deviation_points,
            "magic": self.settings.magic_number,
            "comment": _mt5_comment(client_order_id, comment),
            "type_time": _constant(mt5, "ORDER_TIME_GTC", 0),
        }
        request["type_filling"] = self._validated_order_filling(request)
        result = mt5.order_send(request)
        result_payload = self._checked_result(result, "order_send")
        position_id = self._position_id_from_fill(mt5, result_payload)
        return _market_order_response(
            client_order_id=client_order_id,
            instrument=instrument.name,
            signed_units=signed_units,
            price=_safe_float(result_payload.get("price"), price),
            order_id=str(result_payload.get("order") or ""),
            deal_id=str(result_payload.get("deal") or ""),
            position_id=position_id,
            result=result_payload,
        )

    def _position_id_from_fill(self, mt5: Any, result: dict[str, Any]) -> str:
        """Resolve the MT5 position ticket for a just-filled market order.

        ``MqlTradeResult.order`` is an order ticket, not a position ticket.
        Close history is keyed by ``DEAL_POSITION_ID``/``position_id``. Using
        the order ticket here creates a second open journal row that can never
        be closed by history reconciliation.
        """
        direct = result.get("position") or result.get("position_id")
        if direct:
            return str(direct)
        deal_id = result.get("deal")
        history = getattr(mt5, "history_deals_get", None)
        if not deal_id or history is None:
            return ""
        try:
            deals = history(ticket=int(deal_id))
        except (TypeError, ValueError, AttributeError):
            return ""
        for deal in deals or ():
            position_id = _as_dict(deal).get("position_id")
            if position_id:
                return str(position_id)
        return ""

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

    def close_position(
        self,
        *,
        trade_id: str,
        instrument: FxInstrument,
        signed_units: float,
        comment: str = "fxft-news-close",
    ) -> dict[str, Any]:
        """Close a position at market, mirroring the open-order request shape."""
        self._ensure_connected()
        mt5 = self._module()
        instrument_name = instrument.name
        symbol = instrument.broker_symbol or self.settings.broker_symbol_for(instrument_name)
        side = Side.LONG if signed_units > 0 else Side.SHORT
        tick_payload = self._price_snapshot(instrument_name)
        price = tick_payload.bid if side is Side.LONG else tick_payload.ask
        volume = instrument.units_to_volume(abs(signed_units))
        if volume <= 0 or (instrument.volume_min and volume < instrument.volume_min):
            raise Mt5Error(f"{instrument_name} MT5 volume {volume} is below broker minimum")
        request = {
            "action": _constant(mt5, "TRADE_ACTION_DEAL", 1),
            "symbol": symbol,
            "volume": volume,
            "type": _constant(mt5, "ORDER_TYPE_SELL", 1) if side is Side.LONG else _constant(mt5, "ORDER_TYPE_BUY", 0),
            "position": int(trade_id),
            "price": price,
            "deviation": self.settings.deviation_points,
            "magic": self.settings.magic_number,
            "comment": comment,
            "type_time": _constant(mt5, "ORDER_TIME_GTC", 0),
        }
        request["type_filling"] = self._validated_order_filling(request)
        result = mt5.order_send(request)
        result_payload = self._checked_result(result, "order_send close")
        return {
            "mt5": result_payload,
            "request": request,
            "price": _safe_float(result_payload.get("price"), price),
            "deal_id": str(result_payload.get("deal") or ""),
        }

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
            raise Mt5Error(f"MT5 initialize failed: {mt5.last_error()}")
        account = mt5.account_info()
        if account is None:
            raise Mt5Error(f"MT5 account_info failed after initialize: {mt5.last_error()}")
        self._assert_demo_account(account)
        self._connected = True

    def _mark_disconnected(self) -> None:
        """Drop the cached connection so the next access re-initializes MT5.

        MT5's Python SDK returns ``None`` from every data call once the local
        terminal connection drops (terminal closed, restarted, or network loss).
        Without this, ``_connected`` stays ``True`` forever and the bot can never
        recover a live MT5 connection until the whole process restarts.
        """
        self._connected = False

    def _assert_demo_account(self, account: Any) -> None:
        if not self.settings.demo_only:
            return
        mt5 = self._module()
        trade_mode = _safe_int(_as_dict(account).get("trade_mode"), -1)
        real_mode = _constant(mt5, "ACCOUNT_TRADE_MODE_REAL", 2)
        if trade_mode not in {_constant(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0), _constant(mt5, "ACCOUNT_TRADE_MODE_CONTEST", 1)}:
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
            self._mark_disconnected()
            raise Mt5Error(f"MT5 symbol_info_tick failed for {symbol}: {mt5.last_error()}")
        return PriceSnapshot.from_mt5(
            instrument,
            tick,
            time_offset_seconds=self.settings.time_offset_seconds,
        )

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
            self._mark_disconnected()
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
            self._mark_disconnected()
            raise Mt5Error(f"MT5 {operation} returned no result: {self._module().last_error()}")
        payload = _as_dict(result)
        retcode = _safe_int(payload.get("retcode"), -1)
        ok_codes = {
            _constant(self._module(), "TRADE_RETCODE_DONE", 10009),
            _constant(self._module(), "TRADE_RETCODE_PLACED", 10008),
            _constant(self._module(), "TRADE_RETCODE_DONE_PARTIAL", 10010),
        }
        if retcode not in ok_codes:
            raise Mt5RejectedError(f"MT5 {operation} failed with retcode {retcode}: {payload}")
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
                    "exit_time": _mt5_time_to_datetime(
                        payload.get("time"), self.settings.time_offset_seconds
                    ),
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
            deal_time = _mt5_time_to_datetime(
                payload.get("time"), self.settings.time_offset_seconds
            )
            if deal_time and (event["exit_time"] is None or deal_time > event["exit_time"]):
                event["exit_time"] = deal_time
                event["exit_price"] = _safe_float(payload.get("price"), event["exit_price"])
            event["deals"].append(payload)
        # Opening commissions/fees are also part of realized net P&L.
        for deal in deals:
            payload = _as_dict(deal)
            if _safe_int(payload.get("entry"), -1) != _constant(mt5, "DEAL_ENTRY_IN", 0):
                continue
            trade_id = str(payload.get("position_id") or "")
            if trade_id in grouped:
                grouped[trade_id]["realized_pl"] += _safe_float(payload.get("commission")) + _safe_float(payload.get("fee"))
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
    position_id: str = "",
    result: dict[str, Any],
) -> dict[str, Any]:
    trade_id = position_id or order_id or deal_id
    now = datetime.now(timezone.utc).isoformat()
    return {
        "orderCreateTransaction": {
            "id": order_id,
            "clientExtensions": {"id": client_order_id},
        },
        "orderFillTransaction": {
            "orderID": order_id,
            "id": deal_id,
            "positionID": position_id,
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


def _mt5_time_to_datetime(value: Any, offset_seconds: int = 0) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(float(value) + offset_seconds, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _coerce_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
