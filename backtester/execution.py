"""Realistic Bybit-style futures execution simulator."""

from __future__ import annotations

import logging
import random
from dataclasses import asdict
from uuid import uuid4

import pandas as pd

from backtester.config import BacktestConfig, ExecutionConfig
from backtester.models import (
    ExitPlan,
    Fill,
    InstrumentSpec,
    Liquidity,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Position,
    PositionMode,
    Side,
    TradeRecord,
)

log = logging.getLogger(__name__)


class SimulatedExchange:
    """Candle-level simulator for Bybit USDT perpetual futures.

    The simulator avoids optimistic assumptions:

    * market orders activate on a future candle by default;
    * market and stop-market fills cross spread and slippage;
    * limit fills can be missed even when touched;
    * reduce-only orders cannot flip a position;
    * ambiguous stop/target bars are processed stop-first when configured.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.execution: ExecutionConfig = config.execution
        self.cash = config.risk.initial_equity
        self.realized_pnl = 0.0
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}
        self.fills: list[Fill] = []
        self.trades: list[TradeRecord] = []
        self.snapshots: list[PortfolioSnapshot] = []
        self.mark_prices: dict[str, float] = {}
        self.rng = random.Random(config.execution.random_seed)
        self.stats: dict[str, float] = {
            "orders_submitted": 0,
            "orders_rejected": 0,
            "orders_expired": 0,
            "missed_fills": 0,
            "partial_fills": 0,
            "fees_paid": 0.0,
            "slippage_paid": 0.0,
        }

    def submit_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: float,
        timestamp: pd.Timestamp,
        current_index: int,
        price: float | None = None,
        trigger_price: float | None = None,
        reduce_only: bool = False,
        close_on_trigger: bool = False,
        leverage: float | None = None,
        metadata: dict | None = None,
        order_link_id: str | None = None,
    ) -> Order:
        symbol = symbol.upper()
        instrument = self.instrument(symbol)
        rounded_qty = instrument.round_qty(qty)
        latency = (
            self.execution.market_latency_candles
            if order_type is OrderType.MARKET
            else self.execution.resting_order_latency_candles
        )
        expires_at = None
        if self.execution.order_timeout_candles is not None:
            expires_at = current_index + self.execution.order_timeout_candles
        order = Order(
            order_id=str(uuid4()),
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=rounded_qty,
            price=price,
            trigger_price=trigger_price,
            reduce_only=reduce_only,
            close_on_trigger=close_on_trigger,
            submitted_at=pd.Timestamp(timestamp),
            activation_index=current_index + max(0, latency),
            expires_at_index=expires_at,
            leverage=leverage or self.config.risk.leverage,
            order_link_id=order_link_id,
            metadata=metadata or {},
        )
        if rounded_qty < instrument.min_qty:
            order.status = OrderStatus.REJECTED
            self.stats["orders_rejected"] += 1
            log.info("Rejected %s %s order: qty below min", symbol, order_type.value)
            return order
        self.orders[order.order_id] = order
        self.stats["orders_submitted"] += 1
        return order

    def submit_market_entry(
        self,
        *,
        symbol: str,
        side: Side,
        qty: float,
        timestamp: pd.Timestamp,
        current_index: int,
        leverage: float,
        metadata: dict,
    ) -> Order:
        return self.submit_order(
            symbol=symbol,
            side=side.order_side,
            order_type=OrderType.MARKET,
            qty=qty,
            timestamp=timestamp,
            current_index=current_index,
            reduce_only=False,
            leverage=leverage,
            metadata=metadata,
            order_link_id=f"bt-entry-{uuid4().hex[:10]}",
        )

    def submit_protective_bracket(
        self,
        *,
        symbol: str,
        position_side: Side,
        qty: float,
        entry_price: float,
        exit_plan: ExitPlan,
        timestamp: pd.Timestamp,
        current_index: int,
        leverage: float,
        risk_amount: float,
        metadata: dict | None = None,
    ) -> list[Order]:
        """Place reduce-only stop-loss and take-profit orders."""

        close_side = position_side.close_order_side
        orders: list[Order] = []
        common_metadata = {
            "position_side": position_side.value,
            "risk_amount": risk_amount,
            "exit_plan": asdict(exit_plan),
            **(metadata or {}),
        }
        orders.append(
            self.submit_order(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.STOP_MARKET,
                qty=qty,
                trigger_price=exit_plan.stop_loss,
                timestamp=timestamp,
                current_index=current_index,
                reduce_only=True,
                close_on_trigger=True,
                leverage=leverage,
                metadata={**common_metadata, "exit_reason": "stop_loss"},
                order_link_id=f"bt-sl-{uuid4().hex[:10]}",
            )
        )

        if self.config.strategy.partial_tp_enabled:
            tp1_qty = self.instrument(symbol).round_qty(qty * self.config.strategy.tp1_qty_pct)
            tp2_qty = self.instrument(symbol).round_qty(qty - tp1_qty)
            if tp1_qty > 0:
                orders.append(
                    self.submit_order(
                        symbol=symbol,
                        side=close_side,
                        order_type=OrderType.LIMIT,
                        qty=tp1_qty,
                        price=exit_plan.take_profit,
                        timestamp=timestamp,
                        current_index=current_index,
                        reduce_only=True,
                        leverage=leverage,
                        metadata={**common_metadata, "exit_reason": "take_profit_1"},
                        order_link_id=f"bt-tp1-{uuid4().hex[:10]}",
                    )
                )
            if tp2_qty > 0:
                reward = exit_plan.reward_distance * self.config.strategy.tp2_multiplier
                raw_tp2 = entry_price + reward if position_side is Side.LONG else entry_price - reward
                tp2_price = self.instrument(symbol).round_price(raw_tp2, up=position_side is Side.SHORT)
                orders.append(
                    self.submit_order(
                        symbol=symbol,
                        side=close_side,
                        order_type=OrderType.LIMIT,
                        qty=tp2_qty,
                        price=tp2_price,
                        timestamp=timestamp,
                        current_index=current_index,
                        reduce_only=True,
                        leverage=leverage,
                        metadata={**common_metadata, "exit_reason": "take_profit_2"},
                        order_link_id=f"bt-tp2-{uuid4().hex[:10]}",
                    )
                )
        else:
            orders.append(
                self.submit_order(
                    symbol=symbol,
                    side=close_side,
                    order_type=OrderType.LIMIT,
                    qty=qty,
                    price=exit_plan.take_profit,
                    timestamp=timestamp,
                    current_index=current_index,
                    reduce_only=True,
                    leverage=leverage,
                    metadata={**common_metadata, "exit_reason": "take_profit"},
                    order_link_id=f"bt-tp-{uuid4().hex[:10]}",
                )
            )
        return orders

    def process_candle(
        self,
        *,
        symbol: str,
        timestamp: pd.Timestamp,
        candle: pd.Series,
        bar_index: int,
    ) -> list[Fill]:
        """Run active orders against one candle and return generated fills."""

        symbol = symbol.upper()
        generated: list[Fill] = []
        active = [
            order
            for order in self.orders.values()
            if order.symbol == symbol and order.is_active and order.activation_index <= bar_index
        ]
        active.sort(key=self._order_priority)

        for order in active:
            if not order.is_active:
                continue
            if order.expires_at_index is not None and bar_index > order.expires_at_index:
                order.status = OrderStatus.EXPIRED
                self.stats["orders_expired"] += 1
                continue

            fill_price, liquidity, reference_price = self._candidate_fill(order, candle)
            if fill_price is None or liquidity is None:
                continue
            if self._miss_fill(order):
                self.stats["missed_fills"] += 1
                continue

            executable_qty = self._executable_qty(order)
            if executable_qty <= 0:
                order.status = OrderStatus.CANCELED if order.reduce_only else OrderStatus.REJECTED
                continue
            fill_qty = self._maybe_partial_qty(order, executable_qty)
            fill = self._apply_fill(
                order=order,
                qty=fill_qty,
                price=fill_price,
                liquidity=liquidity,
                timestamp=pd.Timestamp(timestamp),
                reference_price=reference_price,
                bar_index=bar_index,
            )
            if fill:
                generated.append(fill)

        close = float(candle["close"])
        if close > 0:
            self.mark_prices[symbol] = close
        return generated

    def mark_to_market(self, timestamp: pd.Timestamp) -> PortfolioSnapshot:
        equity = self.equity()
        if not self.snapshots:
            peak = max(self.config.risk.initial_equity, equity)
        else:
            peak = max(max(snapshot.equity for snapshot in self.snapshots), equity)
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        snapshot = PortfolioSnapshot(
            timestamp=pd.Timestamp(timestamp),
            equity=equity,
            cash=self.cash,
            unrealized_pnl=self.unrealized_pnl(),
            realized_pnl=self.realized_pnl,
            margin_used=self.margin_used(),
            drawdown_pct=drawdown,
            open_positions=self.open_position_count(),
            gross_exposure=self.gross_exposure(),
            net_exposure=self.net_exposure(),
        )
        self.snapshots.append(snapshot)
        return snapshot

    def cancel_symbol_reduce_only(self, symbol: str) -> None:
        for order in self.orders.values():
            if order.symbol == symbol.upper() and order.reduce_only and order.is_active:
                order.status = OrderStatus.CANCELED

    def instrument(self, symbol: str) -> InstrumentSpec:
        return self.config.instrument_for(symbol)

    def equity(self) -> float:
        return self.cash + self.unrealized_pnl()

    def unrealized_pnl(self) -> float:
        total = 0.0
        for position in self.positions.values():
            mark = self.mark_prices.get(position.symbol, position.avg_price)
            total += position.unrealized_pnl(mark)
        return total

    def margin_used(self) -> float:
        return sum(position.margin for position in self.positions.values())

    def gross_exposure(self) -> float:
        total = 0.0
        for position in self.positions.values():
            mark = self.mark_prices.get(position.symbol, position.avg_price)
            total += abs(position.qty * mark)
        return total

    def net_exposure(self) -> float:
        total = 0.0
        for position in self.positions.values():
            mark = self.mark_prices.get(position.symbol, position.avg_price)
            signed = position.qty * mark if position.side is Side.LONG else -(position.qty * mark)
            total += signed
        return total

    def symbol_exposure(self, symbol: str) -> float:
        symbol = symbol.upper()
        return sum(
            abs(position.qty * self.mark_prices.get(position.symbol, position.avg_price))
            for position in self.positions.values()
            if position.symbol == symbol
        )

    def portfolio_heat(self) -> float:
        return sum(float(position.metadata.get("risk_amount") or 0.0) for position in self.positions.values())

    def open_position_count(self) -> int:
        return sum(1 for position in self.positions.values() if position.qty > 0)

    def has_position(self, symbol: str, side: Side | None = None) -> bool:
        symbol = symbol.upper()
        for position in self.positions.values():
            if position.symbol != symbol or position.qty <= 0:
                continue
            if side is None or position.side is side:
                return True
        return False

    def position_for(self, symbol: str, side: Side | None = None) -> Position | None:
        symbol = symbol.upper()
        if self.execution.position_mode is PositionMode.ONE_WAY:
            position = self.positions.get(symbol)
            if position and (side is None or position.side is side):
                return position
            return None
        if side is None:
            return next((pos for key, pos in self.positions.items() if key.startswith(f"{symbol}:")), None)
        return self.positions.get(self._position_key(symbol, side))

    def _candidate_fill(
        self,
        order: Order,
        candle: pd.Series,
    ) -> tuple[float | None, Liquidity | None, float | None]:
        open_price = float(candle["open"])
        high = float(candle["high"])
        low = float(candle["low"])

        if order.order_type is OrderType.MARKET:
            reference = open_price
            return (
                self._apply_spread_and_slippage(
                    reference,
                    order.side,
                    self.execution.market_slippage_bps,
                    cross_spread=True,
                ),
                Liquidity.TAKER,
                reference,
            )

        if order.order_type is OrderType.LIMIT:
            if order.price is None:
                return None, None, None
            touched = low <= order.price if order.side is OrderSide.BUY else high >= order.price
            if not touched:
                return None, None, None
            return (
                self._apply_spread_and_slippage(
                    order.price,
                    order.side,
                    self.execution.limit_slippage_bps,
                    cross_spread=False,
                ),
                Liquidity.MAKER,
                order.price,
            )

        if order.order_type in {OrderType.STOP_MARKET, OrderType.STOP_LIMIT}:
            if order.trigger_price is None:
                return None, None, None
            triggered = (
                low <= order.trigger_price
                if order.side is OrderSide.SELL
                else high >= order.trigger_price
            )
            if not triggered:
                return None, None, None
            if order.side is OrderSide.SELL:
                reference = min(open_price, order.trigger_price) if open_price <= order.trigger_price else order.trigger_price
            else:
                reference = max(open_price, order.trigger_price) if open_price >= order.trigger_price else order.trigger_price
            if order.order_type is OrderType.STOP_LIMIT and order.price is not None:
                limit_touched = low <= order.price if order.side is OrderSide.BUY else high >= order.price
                if not limit_touched:
                    return None, None, None
                reference = order.price
            return (
                self._apply_spread_and_slippage(
                    reference,
                    order.side,
                    self.execution.stop_slippage_bps,
                    cross_spread=True,
                ),
                Liquidity.TAKER,
                reference,
            )
        return None, None, None

    def _apply_spread_and_slippage(
        self,
        price: float,
        side: OrderSide,
        slippage_bps: float,
        *,
        cross_spread: bool,
    ) -> float:
        spread_bps = self.execution.spread_bps if cross_spread else 0.0
        adjustment_bps = slippage_bps + (spread_bps / 2.0)
        multiplier = 1.0 + adjustment_bps / 10_000.0 if side is OrderSide.BUY else 1.0 - adjustment_bps / 10_000.0
        return max(price * multiplier, 1e-12)

    def _miss_fill(self, order: Order) -> bool:
        if self.execution.missed_fill_probability > 0 and self.rng.random() < self.execution.missed_fill_probability:
            return True
        if order.order_type is OrderType.LIMIT and self.rng.random() > self.execution.limit_fill_probability:
            return True
        if order.order_type in {OrderType.STOP_MARKET, OrderType.STOP_LIMIT} and self.rng.random() > self.execution.stop_fill_probability:
            return True
        return False

    def _executable_qty(self, order: Order) -> float:
        remaining = order.remaining_qty
        if not order.reduce_only:
            return remaining
        side_to_close = Side.LONG if order.side is OrderSide.SELL else Side.SHORT
        position = self.position_for(order.symbol, side_to_close)
        if not position:
            return 0.0
        return min(remaining, position.qty)

    def _maybe_partial_qty(self, order: Order, executable_qty: float) -> float:
        if executable_qty <= 0:
            return 0.0
        if self.execution.partial_fill_probability <= 0:
            return executable_qty
        if self.rng.random() >= self.execution.partial_fill_probability:
            return executable_qty
        fraction = self.rng.uniform(
            self.execution.partial_fill_min_pct,
            self.execution.partial_fill_max_pct,
        )
        qty = max(self.instrument(order.symbol).min_qty, executable_qty * fraction)
        qty = min(executable_qty, self.instrument(order.symbol).round_qty(qty))
        if qty < executable_qty:
            self.stats["partial_fills"] += 1
        return qty

    def _apply_fill(
        self,
        *,
        order: Order,
        qty: float,
        price: float,
        liquidity: Liquidity,
        timestamp: pd.Timestamp,
        reference_price: float | None,
        bar_index: int,
    ) -> Fill | None:
        if qty <= 0:
            return None
        fee_rate = self.execution.taker_fee_rate if liquidity is Liquidity.TAKER else self.execution.maker_fee_rate
        fee = qty * price * fee_rate
        slippage_per_unit = 0.0 if reference_price is None else abs(price - reference_price)
        slippage_cost = slippage_per_unit * qty
        realized = self._update_position_from_fill(
            order=order,
            qty=qty,
            price=price,
            fee=fee,
            slippage_cost=slippage_cost,
            timestamp=timestamp,
            bar_index=bar_index,
        )
        order.record_fill(qty, price)
        if order.reduce_only and order.remaining_qty > 0 and self._executable_qty(order) <= 0:
            order.status = OrderStatus.CANCELED

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=price,
            fee=fee,
            liquidity=liquidity,
            timestamp=timestamp,
            reduce_only=order.reduce_only,
            slippage=slippage_cost,
            realized_pnl=realized,
            metadata=order.metadata.copy(),
        )
        self.fills.append(fill)
        self.stats["fees_paid"] += fee
        self.stats["slippage_paid"] += slippage_cost
        return fill

    def _update_position_from_fill(
        self,
        *,
        order: Order,
        qty: float,
        price: float,
        fee: float,
        slippage_cost: float,
        timestamp: pd.Timestamp,
        bar_index: int,
    ) -> float:
        incoming_side = order.side.position_side
        realized = 0.0

        if order.reduce_only:
            side_to_close = Side.LONG if order.side is OrderSide.SELL else Side.SHORT
            position = self.position_for(order.symbol, side_to_close)
            if not position:
                return 0.0
            return self._close_position(
                position=position,
                qty=min(qty, position.qty),
                exit_price=price,
                exit_fee=fee,
                exit_slippage=slippage_cost,
                timestamp=timestamp,
                bar_index=bar_index,
                exit_reason=str(order.metadata.get("exit_reason") or "reduce_only"),
                metadata=order.metadata,
            )

        existing = self.position_for(order.symbol)
        if existing and existing.side is not incoming_side:
            close_qty = min(qty, existing.qty)
            fee_for_close = fee * (close_qty / qty)
            slippage_for_close = slippage_cost * (close_qty / qty)
            realized += self._close_position(
                position=existing,
                qty=close_qty,
                exit_price=price,
                exit_fee=fee_for_close,
                exit_slippage=slippage_for_close,
                timestamp=timestamp,
                bar_index=bar_index,
                exit_reason="reversal",
                metadata=order.metadata,
            )
            remaining = qty - close_qty
            if remaining > 0:
                fee_for_open = fee - fee_for_close
                slippage_for_open = slippage_cost - slippage_for_close
                self._open_or_average_position(
                    symbol=order.symbol,
                    side=incoming_side,
                    qty=remaining,
                    price=price,
                    fee=fee_for_open,
                    slippage_cost=slippage_for_open,
                    leverage=order.leverage,
                    timestamp=timestamp,
                    bar_index=bar_index,
                    metadata=order.metadata,
                )
            return realized

        self._open_or_average_position(
            symbol=order.symbol,
            side=incoming_side,
            qty=qty,
            price=price,
            fee=fee,
            slippage_cost=slippage_cost,
            leverage=order.leverage,
            timestamp=timestamp,
            bar_index=bar_index,
            metadata=order.metadata,
        )
        return realized

    def _open_or_average_position(
        self,
        *,
        symbol: str,
        side: Side,
        qty: float,
        price: float,
        fee: float,
        slippage_cost: float,
        leverage: float,
        timestamp: pd.Timestamp,
        bar_index: int,
        metadata: dict,
    ) -> None:
        self.cash -= fee
        key = self._position_key(symbol, side)
        existing = self.positions.get(key)
        risk_amount = float(metadata.get("risk_amount") or 0.0)
        if existing is None:
            self.positions[key] = Position(
                symbol=symbol,
                side=side,
                qty=qty,
                avg_price=price,
                leverage=leverage,
                opened_at=timestamp,
                updated_at=timestamp,
                entry_fees=fee,
                entry_slippage=slippage_cost,
                metadata={
                    **metadata,
                    "risk_amount": risk_amount,
                    "entry_bar_index": bar_index,
                },
            )
            return

        total_qty = existing.qty + qty
        existing.avg_price = ((existing.avg_price * existing.qty) + (price * qty)) / total_qty
        existing.qty = total_qty
        existing.updated_at = timestamp
        existing.entry_fees += fee
        existing.entry_slippage += slippage_cost
        existing.metadata["risk_amount"] = float(existing.metadata.get("risk_amount") or 0.0) + risk_amount

    def _close_position(
        self,
        *,
        position: Position,
        qty: float,
        exit_price: float,
        exit_fee: float,
        exit_slippage: float,
        timestamp: pd.Timestamp,
        bar_index: int,
        exit_reason: str,
        metadata: dict,
    ) -> float:
        if qty <= 0 or position.qty <= 0:
            return 0.0
        qty = min(qty, position.qty)
        fraction = qty / position.qty
        gross = (
            (exit_price - position.avg_price) * qty
            if position.side is Side.LONG
            else (position.avg_price - exit_price) * qty
        )
        entry_fee_alloc = position.entry_fees * fraction
        entry_slip_alloc = position.entry_slippage * fraction
        risk_alloc = float(position.metadata.get("risk_amount") or 0.0) * fraction
        net = gross - entry_fee_alloc - exit_fee

        self.cash += gross - exit_fee
        self.realized_pnl += gross - exit_fee
        bars_held = max(0, bar_index - int(position.metadata.get("entry_bar_index") or bar_index))
        self.trades.append(
            TradeRecord(
                trade_id=str(uuid4()),
                symbol=position.symbol,
                side=position.side,
                entry_time=position.opened_at,
                exit_time=timestamp,
                entry_price=position.avg_price,
                exit_price=exit_price,
                qty=qty,
                gross_pnl=gross,
                fees=entry_fee_alloc + exit_fee,
                slippage=entry_slip_alloc + exit_slippage,
                net_pnl=net,
                risk_amount=risk_alloc,
                exit_reason=exit_reason,
                bars_held=bars_held,
                metadata={**position.metadata, **metadata},
            )
        )

        position.qty -= qty
        position.entry_fees -= entry_fee_alloc
        position.entry_slippage -= entry_slip_alloc
        position.updated_at = timestamp
        position.metadata["risk_amount"] = max(0.0, float(position.metadata.get("risk_amount") or 0.0) - risk_alloc)
        if position.qty <= max(1e-12, self.instrument(position.symbol).min_qty * 1e-6):
            self.positions.pop(self._position_key(position.symbol, position.side), None)
            self.cancel_symbol_reduce_only(position.symbol)
        return gross - exit_fee

    def _position_key(self, symbol: str, side: Side) -> str:
        symbol = symbol.upper()
        if self.execution.position_mode is PositionMode.ONE_WAY:
            return symbol
        return f"{symbol}:{side.value}"

    def _order_priority(self, order: Order) -> tuple[int, pd.Timestamp]:
        if self.execution.conservative_intrabar_priority and order.reduce_only and order.is_stop:
            return (0, order.submitted_at)
        if order.order_type is OrderType.MARKET:
            return (1, order.submitted_at)
        if order.reduce_only and order.is_limit:
            return (2, order.submitted_at)
        return (3, order.submitted_at)
