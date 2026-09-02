# FX Strategy Retune Rationale

This phase preserves the original bot's useful structure: MA7/MA14/MA28 trend
stacking, ADX/DI confirmation, higher-timeframe agreement, ATR-based exits,
partial take-profit, trailing-stop intent, daily loss limits, and portfolio
heat controls. The retune changes the assumptions that were crypto-specific.

## Sessions

The default entry timeframe moves to 15 minutes and new entries are limited to
London, New York, and their overlap. Major FX pairs usually offer the best
combination of liquidity, tighter spreads, and directional follow-through in
those windows. Asian-session trading is left configurable, but disabled by
default for the initial EUR/USD, GBP/USD, USD/JPY, AUD/USD basket because range
behavior and lower liquidity can make a momentum/ADX strategy overtrade.

## Market Close, Weekend, And Rollover

The bot now treats the market as closed from Friday 5pm New York through Sunday
5pm New York, blocks fresh entries in the last hour before the Friday close, and
avoids the 5pm New York rollover window. This prevents stale signals from being
queued into the weekend gap and avoids the daily spread/financing reset period.

## Volatility And Stops

Crypto percent-volatility bands were replaced with pip-aware checks. Defaults
use ATR stops at 1.8x ATR, a 4-pip minimum stop, and a 40-pip maximum stop for
the initial majors. ATR must also sit inside a 2-28 pip band before the setup is
accepted. That keeps the strategy out of dead micro-ranges and skips conditions
where a 15-minute trend signal is probably chasing an already-expanded move.

## Signals And Volume

The trend logic remains MA stack plus ADX/DI plus HTF agreement, but the ADX
floor is reduced to 18 on the entry timeframe and 20 on HTF. FX trends often
register lower ADX than small-cap crypto perpetuals. Broker volume is not true
centralized FX volume, so volume confirmation is disabled by default and can be
re-enabled as a tick-volume proxy only after observation.

## Position Sizing

Sizing now works in base-currency units and converts to MT5 lots only at the
broker boundary. Risk is:

```text
units * stop_distance_price * quote_currency_to_account_currency
```

For EUR/USD in a USD account, one pip per unit is 0.0001 USD. For USD/JPY in a
USD account, one pip per unit is 0.01 JPY converted back to USD at the current
price. Cross pairs use direct or reverse account-currency conversion symbols
from MT5 when needed. This is the critical change that prevents JPY pairs and
cross pairs from being mis-sized.

## Margin And Leverage

The code uses MT5 symbol contract size, volume min/max/step, and account
leverage to estimate margin instead of setting exchange leverage. Position size
is capped by risk budget, pair exposure, gross exposure, currency exposure,
available margin, the instrument's max units, and a local
`FX_MAX_UNITS_PER_TRADE` cap. The final order volume is rounded down to the
broker's lot step.

## Swap And Financing

MT5 open-position snapshots journal the current `swap` value reported by the
terminal. Closed-trade sync reads MT5 history deals and carries realized P&L,
commission/fee, and swap into the trade journal once the broker reports the
closing deal.

## Portfolio Risk

The original single-symbol risk controls are extended into portfolio heat,
gross exposure, per-pair exposure, and currency exposure caps. This matters
because EUR/USD and GBP/USD can both become large USD-short or USD-long bets
even though they are separate instruments.

## News Risk

High-impact economic releases can widen spreads and produce slippage that demo
fills may understate. The implementation includes a configurable high-impact
news blackout model loaded from `FX_NEWS_EVENTS_JSON` or
`FX_NEWS_EVENTS_FILE`. It does not download a live calendar yet; operators
should load or schedule events such as NFP, CPI, FOMC, central-bank rate
decisions, and major GDP/inflation releases before relying on unattended
forward-test results.

## Forward-Test Controls

The runnable path is MT5 demo-first and defaults to `MT5_DEMO_ONLY=true`, which
refuses real MT5 trade mode at connection time. Orders use deterministic client
IDs plus a local reservation table before submission, so retry/reconnect paths
do not place a duplicate position for the same signal leg. Daily loss and
max-drawdown halt state is persisted in SQLite, which keeps the kill switch
intact across process restarts.
