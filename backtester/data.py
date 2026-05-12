"""Historical data loading, validation, alignment, and resampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd

from backtester.config import DataConfig
from backtester.models import (
    timeframe_to_pandas_rule,
    timeframe_to_timedelta,
)

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class GapReport:
    symbol: str
    timeframe: str
    expected_candles: int
    actual_candles: int
    missing_candles: int
    largest_gap_candles: int
    gaps: list[tuple[pd.Timestamp, pd.Timestamp, int]] = field(default_factory=list)

    @property
    def has_gaps(self) -> bool:
        return self.missing_candles > 0


@dataclass(frozen=True)
class WalkForwardSegment:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    index: int


class DataPortal:
    """Container for synchronized multi-symbol, multi-timeframe OHLCV data."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.frames: dict[tuple[str, str], pd.DataFrame] = {}
        self.gap_reports: dict[tuple[str, str], GapReport] = {}

    @classmethod
    def from_config(cls, config: DataConfig) -> "DataPortal":
        portal = cls(config)
        portal.load()
        return portal

    def load(self) -> None:
        """Load configured data from CSV/Parquet files and derive timeframes."""

        source_timeframes = [self.config.resample_from] if self.config.resample_from else self.config.timeframes
        for symbol in self.config.symbols:
            for timeframe in source_timeframes:
                if timeframe is None:
                    continue
                normalized = str(timeframe)
                frame = self._try_load_frame(symbol, normalized)
                if frame is not None:
                    self.set_frame(symbol, normalized, frame)

        if self.config.resample_from:
            source = self.config.resample_from
            for symbol in self.config.symbols:
                source_frame = self.get_frame(symbol, source)
                for timeframe in self.config.timeframes:
                    if timeframe == source:
                        continue
                    self.set_frame(symbol, timeframe, resample_ohlcv(source_frame, timeframe))
        else:
            self._derive_missing_higher_timeframes()

        self._ensure_required_frames()

    def _try_load_frame(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        path = self._resolve_path(symbol, timeframe)
        if path is None or not path.exists():
            return None
        log.info("Loading %s %s from %s", symbol, timeframe, path)
        if path.suffix.lower() == ".csv":
            raw = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            try:
                raw = pd.read_parquet(path)
            except ImportError as exc:
                raise ImportError(
                    "Reading Parquet requires pyarrow or fastparquet. "
                    "Install one of them or use CSV input."
                ) from exc
        else:
            raise ValueError(f"Unsupported data file extension: {path}")
        return normalize_ohlcv(raw, self.config)

    def _resolve_path(self, symbol: str, timeframe: str) -> Path | None:
        data_path = Path(self.config.data_path)
        if data_path.is_file():
            return data_path
        filename = self.config.file_pattern.format(symbol=symbol.upper(), timeframe=timeframe)
        direct = data_path / filename
        if direct.exists():
            return direct
        parquet = direct.with_suffix(".parquet")
        if parquet.exists():
            return parquet
        csv = direct.with_suffix(".csv")
        if csv.exists():
            return csv
        return direct

    def _derive_missing_higher_timeframes(self) -> None:
        for symbol in self.config.symbols:
            available = [timeframe for (sym, timeframe) in self.frames if sym == symbol]
            if not available:
                continue
            smallest = min(available, key=lambda tf: timeframe_to_timedelta(tf))
            source = self.frames[(symbol, smallest)]
            for timeframe in self.config.timeframes:
                key = (symbol, timeframe)
                if key not in self.frames and timeframe_to_timedelta(timeframe) >= timeframe_to_timedelta(smallest):
                    log.info("Deriving %s %s from %s", symbol, timeframe, smallest)
                    self.set_frame(symbol, timeframe, resample_ohlcv(source, timeframe))

    def _ensure_required_frames(self) -> None:
        missing = [
            f"{symbol}:{timeframe}"
            for symbol in self.config.symbols
            for timeframe in self.config.timeframes
            if (symbol, timeframe) not in self.frames
        ]
        if missing:
            raise FileNotFoundError(
                "Missing required historical data frames: "
                + ", ".join(missing)
                + f" under {self.config.data_path}"
            )

    def set_frame(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
        symbol = symbol.upper()
        timeframe = str(timeframe)
        validated = validate_ohlcv(frame, timeframe) if self.config.validate_candles else frame.copy()
        clipped = clip_time_range(validated, self.config.start, self.config.end)
        if self.config.fill_missing:
            clipped = fill_missing_candles(clipped, timeframe)
        self.frames[(symbol, timeframe)] = clipped
        self.gap_reports[(symbol, timeframe)] = detect_gaps(clipped, symbol, timeframe)

    def get_frame(self, symbol: str, timeframe: str) -> pd.DataFrame:
        try:
            return self.frames[(symbol.upper(), str(timeframe))]
        except KeyError as exc:
            raise KeyError(f"No frame loaded for {symbol.upper()} {timeframe}") from exc

    def replace_frame(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
        """Replace a frame after indicator enrichment while keeping reports."""

        self.frames[(symbol.upper(), str(timeframe))] = frame.sort_index()

    def symbols(self) -> list[str]:
        return list(self.config.symbols)

    def timeframes(self) -> list[str]:
        return list(self.config.timeframes)

    def synchronized_index(self, timeframe: str | None = None) -> pd.DatetimeIndex:
        timeframe = timeframe or self.config.base_timeframe
        indexes = [self.get_frame(symbol, timeframe).index for symbol in self.config.symbols]
        if not indexes:
            return pd.DatetimeIndex([])
        if self.config.alignment == "union":
            aligned = indexes[0]
            for index in indexes[1:]:
                aligned = aligned.union(index)
            return aligned.sort_values()
        aligned = indexes[0]
        for index in indexes[1:]:
            aligned = aligned.intersection(index)
        return aligned.sort_values()

    def get_candle(self, symbol: str, timeframe: str, timestamp: pd.Timestamp) -> pd.Series | None:
        frame = self.get_frame(symbol, timeframe)
        timestamp = pd.Timestamp(timestamp)
        if timestamp in frame.index:
            return frame.loc[timestamp]
        return None

    def history(
        self,
        symbol: str,
        timeframe: str,
        decision_time: pd.Timestamp,
        *,
        lookback: int | None = None,
        closed: bool = True,
    ) -> pd.DataFrame:
        """Return historical candles visible at a decision timestamp.

        Data indexes are candle open times.  With `closed=True`, a higher
        timeframe bar is available only when open_time + timeframe <=
        decision_time.
        """

        frame = self.get_frame(symbol, timeframe)
        decision_time = pd.Timestamp(decision_time)
        if closed:
            delta = pd.Timedelta(timeframe_to_timedelta(timeframe))
            mask = frame.index + delta <= decision_time
            history = frame.loc[mask]
        else:
            history = frame.loc[frame.index <= decision_time]
        if lookback is not None and lookback > 0:
            history = history.tail(lookback)
        return history.copy()

    def apply_to_frames(self, transform: Callable[[pd.DataFrame], pd.DataFrame], timeframes: list[str] | None = None) -> None:
        """Apply an indicator/enrichment function to selected frames."""

        selected = set(timeframes or self.config.timeframes)
        for key, frame in list(self.frames.items()):
            symbol, timeframe = key
            if timeframe in selected:
                log.info("Enriching %s %s", symbol, timeframe)
                self.frames[key] = transform(frame.copy()).sort_index()

    def iter_symbol_candles(
        self,
        timeframe: str | None = None,
    ) -> Iterator[tuple[pd.Timestamp, dict[str, pd.Series]]]:
        timeframe = timeframe or self.config.base_timeframe
        for timestamp in self.synchronized_index(timeframe):
            candles = {
                symbol: candle
                for symbol in self.config.symbols
                if (candle := self.get_candle(symbol, timeframe, timestamp)) is not None
            }
            yield timestamp, candles

    def walk_forward_segments(self, train_bars: int, test_bars: int) -> list[WalkForwardSegment]:
        index = self.synchronized_index(self.config.base_timeframe)
        segments: list[WalkForwardSegment] = []
        start = 0
        segment_id = 0
        while start + train_bars + test_bars <= len(index):
            train_slice = index[start : start + train_bars]
            test_slice = index[start + train_bars : start + train_bars + test_bars]
            segments.append(
                WalkForwardSegment(
                    train_start=train_slice[0],
                    train_end=train_slice[-1],
                    test_start=test_slice[0],
                    test_end=test_slice[-1],
                    index=segment_id,
                )
            )
            start += test_bars
            segment_id += 1
        return segments


def normalize_ohlcv(raw: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Normalize input columns, timezone, types, sort order, and index."""

    frame = raw.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    timestamp_column = config.timestamp_column.lower()
    if timestamp_column not in frame.columns:
        for candidate in ("time", "start", "open_time", "datetime", "date"):
            if candidate in frame.columns:
                timestamp_column = candidate
                break
        else:
            raise ValueError(f"Timestamp column '{config.timestamp_column}' not found")

    missing_columns = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing OHLCV columns: {missing_columns}")

    timestamp_raw = frame[timestamp_column]
    if pd.api.types.is_numeric_dtype(timestamp_raw):
        # Bybit exports milliseconds; smaller values are usually seconds.
        unit = "ms" if float(timestamp_raw.dropna().abs().median()) > 10_000_000_000 else "s"
        timestamps = pd.to_datetime(timestamp_raw, unit=unit, utc=True, errors="coerce")
    else:
        timestamps = pd.to_datetime(timestamp_raw, utc=True, errors="coerce")

    frame = frame.assign(timestamp=timestamps)
    for column in OHLCV_COLUMNS + ["turnover"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["timestamp", *OHLCV_COLUMNS])
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
    frame = frame.set_index("timestamp")
    if config.timezone.upper() != "UTC":
        frame.index = frame.index.tz_convert(config.timezone)
    return frame[[column for column in frame.columns if column in {*OHLCV_COLUMNS, "turnover"}]]


def validate_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Validate candle consistency without trying to repair market data."""

    required = set(OHLCV_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Frame missing required columns: {sorted(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("OHLCV frame must use a DatetimeIndex")
    if frame.index.has_duplicates:
        raise ValueError("OHLCV frame contains duplicate timestamps")
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    if frame[OHLCV_COLUMNS].isna().any().any():
        raise ValueError("OHLCV frame contains NaN values")
    bad_high = frame["high"] < frame[["open", "close", "low"]].max(axis=1)
    bad_low = frame["low"] > frame[["open", "close", "high"]].min(axis=1)
    if bool((bad_high | bad_low).any()):
        first_bad = frame.index[bad_high | bad_low][0]
        raise ValueError(f"Invalid OHLC relationship at {first_bad} for {timeframe}")
    if bool((frame[["open", "high", "low", "close"]] <= 0).any().any()):
        raise ValueError("OHLC prices must be positive")
    if bool((frame["volume"] < 0).any()):
        raise ValueError("Volume cannot be negative")
    return frame


def detect_gaps(frame: pd.DataFrame, symbol: str, timeframe: str) -> GapReport:
    if frame.empty:
        return GapReport(symbol, timeframe, 0, 0, 0, 0, [])
    delta = pd.Timedelta(timeframe_to_timedelta(timeframe))
    diffs = frame.index.to_series().diff().dropna()
    gaps: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    missing = 0
    largest = 0
    for end_time, diff in diffs.items():
        if diff > delta:
            gap_candles = int(round(diff / delta)) - 1
            start_time = end_time - diff
            gaps.append((start_time, end_time, gap_candles))
            missing += gap_candles
            largest = max(largest, gap_candles)
    expected = len(frame) + missing
    report = GapReport(symbol, timeframe, expected, len(frame), missing, largest, gaps)
    if report.has_gaps:
        log.warning(
            "%s %s gap report: %s missing candles, largest gap %s candles",
            symbol,
            timeframe,
            report.missing_candles,
            report.largest_gap_candles,
        )
    return report


def fill_missing_candles(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Forward-fill missing OHLC candles with zero volume.

    This is useful for alignment but should be enabled only when the user
    understands the implications for execution realism.
    """

    if frame.empty:
        return frame
    rule = timeframe_to_pandas_rule(timeframe)
    full_index = pd.date_range(frame.index[0], frame.index[-1], freq=rule, tz=frame.index.tz)
    filled = frame.reindex(full_index)
    previous_close = filled["close"].ffill()
    for column in ["open", "high", "low", "close"]:
        filled[column] = filled[column].fillna(previous_close)
    filled["volume"] = filled["volume"].fillna(0.0)
    if "turnover" in filled:
        filled["turnover"] = filled["turnover"].fillna(0.0)
    filled.index.name = "timestamp"
    return filled


def clip_time_range(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    clipped = frame
    if start:
        clipped = clipped.loc[clipped.index >= _coerce_timestamp_for_index(start, clipped.index)]
    if end:
        clipped = clipped.loc[clipped.index <= _coerce_timestamp_for_index(end, clipped.index)]
    return clipped


def _coerce_timestamp_for_index(value: str, index: pd.DatetimeIndex) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if index.tz is not None:
        timestamp = timestamp.tz_convert(index.tz)
    return timestamp


def resample_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = timeframe_to_pandas_rule(timeframe)
    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "turnover" in frame.columns:
        aggregation["turnover"] = "sum"
    result = frame.resample(rule, label="left", closed="left").agg(aggregation)
    result = result.dropna(subset=OHLCV_COLUMNS)
    return validate_ohlcv(result, timeframe)
