"""
Export selected experiment-level statistics from six summary files.

Reads aggregates only. Patient arrays, trajectories and checkpoints are
never loaded or copied.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, stdev

METHODS = ("d3qn", "cql", "iql")
CHECKPOINTS = ("best", "final")
SEEDS = (42, 43, 44, 45, 46)
SUPPORT_NAMES = {"d3qn": "D3QN", "cql": "CQL (alpha=2.0)", "iql": "IQL"}
SOURCE_FILES = (
    "final_results_test.csv", "final_results_test.json",
    "data_support/support_metrics.json",
    "data_support/support_metrics_final.json",
    "data_support/support_metrics_withhold.json", "ope_diagnostics.csv",
)
SUPPORT_FIELDS = ("mean_local_support", "pct_unsupported", "pct_zero_support")
OPE_FIELDS = ("wis", "pdwis", "dr", "ess", "n_trajectories",
              "n_full_trajectory_matches")
PAIRED_FIELDS = ("method", "reference", "checkpoint", "split", "n_seeds",
                 "mean_diff", "sd_diff", "p", "p_holm", "holm_family_size")


def number(value):
    """Represent undefined values as JSON null, never as zero or NaN."""
    if value is None or value == "":
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(data):
    expected = {(m, c, s) for m in METHODS for c in CHECKPOINTS for s in SEEDS}
    for name in ("fqe_runs", "ope_runs"):
        rows = data[name]
        keys = [(r["method"], r["checkpoint"], r["seed"]) for r in rows]
        require(len(keys) == len(set(keys)) and set(keys) == expected,
                f"{name}: expected exactly 30 unique method/checkpoint/seed rows")
        require(all(r["split"] == "test" for r in rows), f"{name}: mixed splits")
    require(len(data["policy_summary"]) == 6, "Expected six policy summaries")
    require({(r["method"], r["checkpoint"]) for r in data["policy_summary"]}
            == {(m, c) for m in METHODS for c in CHECKPOINTS},
            "Summary checkpoint mismatch")
    for r in data["policy_summary"]:
        vals = [v["fqe"] for v in data["fqe_runs"]
                if (v["method"], v["checkpoint"]) == (r["method"], r["checkpoint"])]
        require(all(v is not None for v in vals), "Missing FQE estimate")
        require(r["n_seeds"] == 5, "Incomplete seed coverage")
        require(math.isclose(mean(vals), r["fqe_mean"], abs_tol=1e-10)
                and math.isclose(stdev(vals), r["fqe_sd"], abs_tol=1e-10),
                "FQE summary does not match seed results")
        require(0 <= r["mean_local_support"] <= 1, "Invalid support fraction")
        require(0 <= r["pct_unsupported"] <= 100, "Invalid unsupported percentage")
    require(all(r["n_trajectories"] == data["protocol"]["test_trajectories"]
                for r in data["ope_runs"]), "OPE population mismatch")
    require(all(r["dr_control_variate"] == "fqe" for r in data["ope_runs"]),
            "Greedy DR must use the fitted evaluation critic")
    pairs = [(r["method"], r["checkpoint"], r["reference"]) for r in data["paired_tests"]]
    require(len(pairs) == 12 and set(pairs) ==
            {(m, c, ref) for m in METHODS for c in CHECKPOINTS
             for ref in ("behaviour_est", "withhold")}, "Incomplete paired comparisons")
    require(all(r["split"] == "test" and r["n_seeds"] == 5
                and r["holm_family_size"] == 6 for r in data["paired_tests"]),
            "Paired comparison protocol mismatch")


def export(run_dir, source_commit, source_directory=None):
    raw = {p: (run_dir / p).read_bytes() for p in SOURCE_FILES}
    doc = lambda p: json.loads(raw[p])
    csv_rows = lambda p: list(csv.DictReader(io.StringIO(raw[p].decode())))
    final = doc("final_results_test.json")
    require(final["complete"] and final["split"] == "test", "Incomplete results")
    supports = {"best": doc("data_support/support_metrics.json"),
                "final": doc("data_support/support_metrics_final.json")}
    control = doc("data_support/support_metrics_withhold.json")
    first = supports["best"]
    for sup in [*supports.values(), control]:
        require(sup["split"] == "test" and sup["n_eval_points"] == first["n_eval_points"]
                and sup["knn"] == first["knn"]
                and sup["unsupported_threshold"] == first["unsupported_threshold"],
                "Support populations or protocols differ")
    for sup in supports.values():
        require(sup["seeds"] == list(SEEDS), "Support seed order differs")
        require(all(sup["policies"][n]["n_seeds"] == 5 for n in SUPPORT_NAMES.values()),
                "Incomplete support seeds")
        for name in SUPPORT_NAMES.values():
            require(len(sup["per_seed"][name]) == 5, "Missing per-seed support summaries")
            for metric in SUPPORT_FIELDS:
                values = [r[metric] for r in sup["per_seed"][name]]
                require(math.isclose(mean(values), sup["policies"][name][metric], abs_tol=1e-10)
                        and math.isclose(stdev(values), sup["policies"][name][metric + "_sd"], abs_tol=1e-10),
                        "Support summary differs from seed results")
    require(supports["best"]["policies"]["Clinician (observed)"]
            == supports["final"]["policies"]["Clinician (observed)"],
            "Observed clinician baseline changed across checkpoints")

    fqe = []
    references = {}
    for r in csv_rows("final_results_test.csv"):
        require(r["split"] == "test", "Unexpected FQE split")
        if r["method"] in METHODS:
            fqe.append({"method": r["method"], "checkpoint": r["checkpoint"],
                        "seed": int(float(r["seed"])), "split": "test", "fqe": number(r["fqe"])})
        else:
            name = r["method"]
            require(name in ("behaviour_est", "bc", "withhold"), "Unknown reference")
            value = number(r["fqe"])
            require(name not in references or references[name] == value,
                    "Reference differs across checkpoint labels")
            references[name] = value
    require(set(references) == {"behaviour_est", "bc", "withhold"}, "Missing reference")

    summary = []
    for m in METHODS:
        for c in CHECKPOINTS:
            rows = [r for r in final["table"] if r["method"] == m and r["checkpoint"] == c]
            require(len(rows) == 1, "Missing or duplicated summary row")
            r = rows[0]
            s = supports[c]["policies"][SUPPORT_NAMES[m]]
            summary.append({"method": m, "checkpoint": c, "n_seeds": r["count"],
                            "fqe_mean": r["mean"], "fqe_sd": r["std"],
                            **{k: s[k] for k in SUPPORT_FIELDS},
                            **{k + "_sd": s[k + "_sd"] for k in SUPPORT_FIELDS}})
    ope = []
    for r in csv_rows("ope_diagnostics.csv"):
        if r["split"] != "test" or r["policy"] != "greedy (exact argmax)":
            continue
        ope.append({"method": r["method"], "seed": int(r["seed"]),
                    "checkpoint": r["checkpoint"], "split": "test",
                    **{k: number(r[k]) for k in OPE_FIELDS},
                    "dr_control_variate": r["dr_control_variate"],
                    "unsupported_step_count": len(json.loads(r["unsupported_steps"]))})
    require(bool(ope), "No greedy test diagnostics")
    data = {
        "schema_version": 1,
        "provenance": {"source_commit": source_commit,
                       "source_directory": source_directory or run_dir.name,
                       "source_files": [{"path": p, "sha256": hashlib.sha256(raw[p]).hexdigest()}
                                        for p in SOURCE_FILES],
                       "description": "Experiment-level aggregates only; no patient-level records."},
        "protocol": {"split": "test", "policy": "greedy (exact argmax)",
                     "test_trajectories": int(ope[0]["n_trajectories"]),
                     "test_decisions": first["n_eval_points"], "knn": first["knn"],
                     "unsupported_threshold": first["unsupported_threshold"],
                     "return_bound": 15,
                     "checkpoint_definitions": final["checkpoint_definition_by_method"]},
        "references": references,
        "support_references": {
            "observed_clinician": {k: first["policies"]["Clinician (observed)"][k]
                                   for k in SUPPORT_FIELDS},
            "withhold": {k: control["policies"]["Withhold treatment"][k] for k in SUPPORT_FIELDS}},
        "fqe_runs": fqe, "policy_summary": summary, "ope_runs": ope,
        "paired_tests": [{k: r[k] for k in PAIRED_FIELDS} for r in final["paired_tests"]],
    }
    validate(data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True, help="Revision that produced these inputs")
    parser.add_argument("--source-directory", help="Portable source label; defaults to the run directory name")
    parser.add_argument("--out", type=Path, default=Path("results/thesis_results.json"))
    args = parser.parse_args()
    data = export(args.run_dir, args.source_commit, args.source_directory)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(f"Exported 30 policy fits and six summaries to {args.out}")


if __name__ == "__main__":
    main()
