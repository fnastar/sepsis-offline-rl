"""Guard against incomplete exports and misleading checkpoint/estimator labels."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.export_results import validate
from scripts.plot_results import estimator_verdicts

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data():
    return json.loads((ROOT / "results/thesis_results.json").read_text())


def test_saved_results_are_complete_and_match_thesis(data):
    validate(data)
    assert data["protocol"]["test_trajectories"] == 3817
    assert data["protocol"]["test_decisions"] == 37874
    expected = {("d3qn", "best"): (13.308, 57.146),
                ("d3qn", "final"): (11.698, 86.192),
                ("cql", "best"): (12.468, .003),
                ("cql", "final"): (12.537, 1.321),
                ("iql", "best"): (12.874, .066),
                ("iql", "final"): (12.470, .171)}
    for row in data["policy_summary"]:
        value, unsupported = expected[row["method"], row["checkpoint"]]
        assert row["fqe_mean"] == pytest.approx(value, abs=.0005)
        assert row["pct_unsupported"] == pytest.approx(unsupported, abs=.0005)


@pytest.mark.parametrize("rows", ["fqe_runs", "ope_runs"])
@pytest.mark.parametrize("damage", ["missing", "duplicate", "mixed_split", "wrong_checkpoint"])
def test_invalid_export_cannot_be_plotted(data, rows, damage):
    altered = deepcopy(data)
    if damage == "missing":
        altered[rows].pop()
    elif damage == "duplicate":
        altered[rows][-1] = deepcopy(altered[rows][0])
    elif damage == "mixed_split":
        altered[rows][0]["split"] = "val"
    else:
        altered[rows][0]["checkpoint"] = "final"
    with pytest.raises(ValueError):
        validate(altered)


def test_mismatched_mean_is_rejected(data):
    data["policy_summary"][0]["fqe_mean"] += 1
    with pytest.raises(ValueError, match="summary"):
        validate(data)


def test_reliability_counts_match_thesis(data):
    verdicts = estimator_verdicts(data)
    expected = {"WIS": {"defined": 24, "undefined": 6},
                "PDWIS": {"defined": 2, "withheld": 22, "undefined": 6},
                "DR": {"defined": 18, "outside": 12},
                "FQE": {"defined": 30}}
    for estimator, counts in expected.items():
        assert Counter(v for (e, *_), v in verdicts.items() if e == estimator) == counts


def test_zero_is_not_an_undefined_estimate(data):
    row = data["ope_runs"][0]
    row["wis"] = row["pdwis"] = row["dr"] = 0
    row["n_full_trajectory_matches"] = 0
    key = row["method"], row["checkpoint"], row["seed"]
    verdicts = estimator_verdicts(data)
    for estimator in ("WIS", "PDWIS", "DR"):
        assert verdicts[(estimator, *key)] == "defined"


def test_export_contains_no_patient_keys_or_arrays(data):
    forbidden = {"subject_id", "stay_id", "hadm_id", "v0", "states", "actions",
                 "rewards", "n_supported_by_step", "n_active_by_step"}
    def check(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(data)
    # Strict JSON parsers must not encounter non-standard NaN tokens.
    json.dumps(data, allow_nan=False)
