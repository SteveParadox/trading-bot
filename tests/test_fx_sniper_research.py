import json

import pandas as pd
import pytest

from fxbot.sniper_research import quote_outcome, validate_quotes, metrics, load_candidates, study


def candidate(at="2026-01-06T14:00:00Z", side="LONG"):
    return {"instrument": "EUR_USD", "decision_time": at, "would_allow": True,
            "snapshot": {"observed_at": at, "side": side, "bid": 1.1, "ask": 1.1001,
                         "parameters": {"slippage_pips_per_side": 0., "commission_pips_round_trip": 0.},
                         "stages": {"cost": True, "location": True}, "extension_atr": .3},
            "baseline_risk": {"exit_plan": {"stop_loss": 1.0991 if side == "LONG" else 1.101}}}


def quotes(bids, asks, start="2026-01-06T14:00:30Z"):
    return validate_quotes(pd.DataFrame({"timestamp": pd.date_range(start, periods=len(bids), freq="30s"),
                                         "instrument": "EUR_USD", "bid": bids, "ask": asks}))


def test_long_quote_path_uses_bid_and_caps_target_improvement():
    q = quotes([1.1000, 1.1006, 1.102], [1.1001, 1.1007, 1.1021])
    o = quote_outcome(candidate(), q, target_r=.5, horizon_seconds=300)
    assert o.reason == "target" and o.net_r == pytest.approx(.5)
    assert o.holding_seconds == 60
    assert o.mae_r == pytest.approx(.1)


def test_short_quote_path_uses_ask_and_gap_stop():
    q = quotes([1.1008, 1.1014], [1.1009, 1.1015])
    o = quote_outcome(candidate(side="SHORT"), q, target_r=.5, horizon_seconds=300)
    assert o.reason == "stop" and o.net_r == pytest.approx(-1.5)


def test_missing_coverage_is_not_forced_into_a_win_or_loss():
    q = quotes([1.102], [1.1021], start="2026-01-06T14:05:00Z")
    assert quote_outcome(candidate(), q, target_r=.5, horizon_seconds=300) is None


def test_cost_stress_reduces_same_path_net_result():
    c = candidate()
    c["snapshot"]["parameters"] = {"slippage_pips_per_side": .1, "commission_pips_round_trip": .2}
    q = quotes([1.1003, 1.1003], [1.1004, 1.1004])
    one = quote_outcome(c, q, target_r=1, horizon_seconds=60)
    two = quote_outcome(c, q, target_r=1, horizon_seconds=60, cost_multiplier=2)
    assert two.net_r < one.net_r


def test_purge_outcomes_that_cross_split_boundary():
    q = quotes([1.1003, 1.1003], [1.1004, 1.1004])
    report = study([candidate()], q, train_end="2026-01-06T14:00:40Z", validation_end="2026-01-07T00:00:00Z",
                    horizon_seconds=60)
    assert report["purged_boundary_candidates"] == 1
    assert all(e["groups"]["baseline_candidates"]["trades"] == 0 for e in report["experiments"])
    assert report["portfolio_metrics"]["max_drawdown"] is None


def test_dedup_keeps_earliest_observation(tmp_path):
    c = candidate()
    later = json.loads(json.dumps(c))
    later["snapshot"]["observed_at"] = "2026-01-06T14:01:00Z"
    path = tmp_path / "journal.jsonl"
    path.write_text("\n".join(json.dumps({"type": "event", "payload": {"event_type": "sniper_candidate", "payload": item}})
                              for item in [later, c]))
    assert load_candidates(path) == [c]


def test_strict_quotes_and_empty_metrics():
    q = quotes([1.1, 1.1], [1.1001, 1.1001])
    with pytest.raises(ValueError):
        validate_quotes(q.iloc[::-1])
    q.loc[0, "ask"] = 1.099
    with pytest.raises(ValueError):
        validate_quotes(q)
    assert metrics([])["expectancy_r"] is None
