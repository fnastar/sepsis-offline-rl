"""
Rebuild research figures and the results table from saved aggregates.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from scripts.export_results import CHECKPOINTS, METHODS, SEEDS, validate

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"d3qn": "#2876B8", "cql": "#C66522", "iql": "#23846D", "withhold": "#858585"}
LABELS = {"d3qn": "D3QN", "cql": "CQL", "iql": "IQL", "withhold": "Withhold\ncontrol"}


def estimator_verdicts(data):
    fqe = {(r["method"], r["checkpoint"], r["seed"]): r["fqe"] for r in data["fqe_runs"]}
    verdicts = {}
    for r in data["ope_runs"]:
        key = r["method"], r["checkpoint"], r["seed"]
        verdicts[("WIS", *key)] = "undefined" if r["wis"] is None else "defined"
        verdicts[("PDWIS", *key)] = (
            "defined" if r["pdwis"] is not None else
            "undefined" if r["n_full_trajectory_matches"] == 0 else "withheld")
        verdicts[("DR", *key)] = (
            "undefined" if r["dr"] is None else
            "outside" if abs(r["dr"]) > data["protocol"]["return_bound"] else "defined")
        verdicts[("FQE", *key)] = "undefined" if fqe[key] is None else "defined"
    return verdicts


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E2E6EA", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#B7C0C8")
    ax.tick_params(length=0, pad=7)


def save(fig, directory, stem):
    fig.savefig(directory / f"{stem}.png", dpi=180, facecolor="white")
    fig.savefig(directory / f"{stem}.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def value_support(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ref = data["references"]["behaviour_est"]
    keys = ["d3qn", "iql", "cql", "withhold"]
    x = np.arange(4)
    for row, cp in enumerate(CHECKPOINTS):
        rows = {r["method"]: r for r in data["policy_summary"] if r["checkpoint"] == cp}
        vals = [rows[k]["fqe_mean"] - ref if k != "withhold" else data["references"][k] - ref for k in keys]
        sds = [rows[k]["fqe_sd"] if k != "withhold" else 0 for k in keys]
        sup = [rows[k]["mean_local_support"] if k != "withhold"
               else data["support_references"][k]["mean_local_support"] for k in keys]
        ss = [rows[k]["mean_local_support_sd"] if k != "withhold" else 0 for k in keys]
        for j, (values, errors) in enumerate(((vals, sds), (sup, ss))):
            ax = axes[row, j]
            style(ax)
            bars = ax.bar(x, values, yerr=errors, capsize=4,
                          color=[COLORS[k] for k in keys], width=0.57,
                          error_kw={"elinewidth": 1.2, "ecolor": "#26333D"})
            bars[-1].set_hatch("///")
            bars[-1].set_edgecolor("white")
            ax.set_xticks(x, [LABELS[k] for k in keys])
            for pos, val, err in zip(x, values, errors):
                label = f"{val:+.2f}" if j == 0 else f"{val:.3f}"
                gap = 0.06 if j == 0 else 0.008
                ax.text(pos, val + (err + gap) * (1 if val >= 0 else -1), label,
                        ha="center", va="bottom" if val >= 0 else "top", fontsize=10)
            checkpoint = "Early" if cp == "best" else "Final"
            ax.set_title(f"{checkpoint} checkpoint · " + ("estimated value" if j == 0 else "local data support"),
                         loc="left", fontsize=12, pad=15, weight="bold")
            if j == 0:
                ax.axhline(0, color="#657482", linewidth=1)
                ax.set_ylim(-1.9, 1.25)
                ax.set_ylabel("FQE difference vs estimated behaviour")
            else:
                clin = data["support_references"]["observed_clinician"]["mean_local_support"]
                ax.axhline(clin, color="#657482", linestyle="--", linewidth=1.2)
                ax.set_ylim(0, .30)
                ax.set_ylabel("Mean fraction of 50 neighbours")
    fig.suptitle("A higher estimated value can come with weaker data support", fontsize=16,
                 weight="bold", x=.06, ha="left", y=.98)
    fig.text(.06, .93, "Test split · five training seeds per method · bars show means; whiskers show ±1 seed SD",
             fontsize=10, color="#52616D")
    fig.text(.06, .055,
             "Dashed line: support for observed clinician actions. Hatched bar: always-withhold control (one fit; no seed SD).\n"
             "Early/final: epoch-1/40 Q-network for D3QN/CQL; actor from the epoch-1/40 critic for IQL.\n"
             "FQE is an estimated discounted return, not a survival benefit. Source: results/thesis_results.json.",
             fontsize=9, color="#52616D", linespacing=1.5)
    fig.subplots_adjust(left=.09, right=.98, bottom=.19, top=.86, hspace=.65, wspace=.30)
    save(fig, out, "value_support")


def reliability(data, out):
    v = estimator_verdicts(data)
    estimators = ["WIS", "PDWIS", "DR", "FQE"]
    fig, ax = plt.subplots(figsize=(11, 5.3))
    fills = {"defined": "#536C7C", "outside": "#B84336", "withheld": "white"}
    totals = {}
    for i, est in enumerate(estimators):
        totals[est] = Counter(cat for (e, *_), cat in v.items() if e == est)
        for j, (m, cp) in enumerate((m, c) for m in METHODS for c in CHECKPOINTS):
            for k, seed in enumerate(SEEDS):
                x, y = j * 1.6 + k * .22, 3 - i
                cat = v[(est, m, cp, seed)]
                if cat == "undefined":
                    ax.plot(x, y, marker="x", color="#9BA4AB", markersize=7, markeredgewidth=1.5)
                else:
                    ax.add_patch(Rectangle((x-.09, y-.12), .18, .24,
                                           facecolor=fills[cat], edgecolor="#536C7C",
                                           linewidth=.8, hatch="////" if cat == "outside" else None))
            if est == "WIS":
                ess = median(r["ess"] for r in data["ope_runs"] if r["method"] == m and r["checkpoint"] == cp)
                ax.text(j * 1.6 + .44, y-.24, f"ESS {ess:.1f}", ha="center", va="top", fontsize=9, color="#52616D")
        count = totals[est]
        defined = count["defined"] + count["outside"]
        detail = f"{defined}/30 defined"
        if count["outside"]:
            detail += f"\n{count['outside']} outside ±15"
        if count["withheld"]:
            detail += f"\n{count['withheld']} withheld"
        ax.text(9.5, 3-i, detail, va="center", fontsize=10)
    for j, (m, cp) in enumerate((m, c) for m in METHODS for c in CHECKPOINTS):
        ax.text(j*1.6+.44, 3.6, "Early" if cp == "best" else "Final", ha="center", fontsize=10)
    for j, m in enumerate(METHODS):
        ax.text(j*3.2+1.24, 4.0, LABELS[m], ha="center", fontsize=12, weight="bold")
    ax.set_yticks([3, 2, 1, 0], estimators, fontsize=12)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_xticks([])
    ax.spines[:].set_visible(False)
    ax.set_xlim(-.3, 11.9)
    ax.set_ylim(-.5, 4.3)
    legend = [Patch(facecolor=fills["defined"], label="Defined (not proof of reliability)"),
              Patch(facecolor="white", edgecolor="#536C7C", label="Withheld: unsupported step"),
              Patch(facecolor=fills["outside"], hatch="////", label="Defined, outside return bound"),
              plt.Line2D([], [], linestyle="", marker="x", color="#9BA4AB", label="Undefined: no matched trajectory")]
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(.065, .10), ncol=2,
               frameon=False, fontsize=10)
    fig.suptitle("Estimator diagnostics across all 30 policy fits", x=.06, ha="left",
                 y=.98, fontsize=16, weight="bold")
    n = data["protocol"]["test_trajectories"]
    fig.text(.06, .91, f"One mark per training seed · greedy policies · test split of {n:,} trajectories", fontsize=10, color="#52616D")
    fig.text(.06, .045, f"WIS effective sample sizes (ESS) are medians across five seeds, out of {n:,} trajectories.\n"
             "FQE is defined for every fit but still depends on model assumptions. Source: results/thesis_results.json.",
             fontsize=9, color="#52616D", linespacing=1.5)
    fig.subplots_adjust(left=.10, right=.98, top=.84, bottom=.30)
    save(fig, out, "estimator_reliability")
    return totals


def table(data, path):
    ref = data["references"]["behaviour_est"]
    protocol = data["protocol"]
    knn = protocol["knn"]
    lines = ["# Saved thesis results", "",
             "Generated by `python -m scripts.plot_results` from `results/thesis_results.json`.", "",
             "Test split; five training seeds per learned policy. SD is between-seed variation, not a patient confidence interval.", "",
             "| Method | Checkpoint | FQE mean ± SD | Difference vs estimated behaviour | Unsupported actions | Mean local support | Zero local support |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in data["policy_summary"]:
        cp = "Early" if r["checkpoint"] == "best" else "Final"
        lines.append(f"| {LABELS[r['method']]} | {cp} | {r['fqe_mean']:.3f} ± {r['fqe_sd']:.3f} | "
                     f"{r['fqe_mean']-ref:+.3f} | {r['pct_unsupported']:.3f}% | "
                     f"{r['mean_local_support']:.3f} | {r['pct_zero_support']:.2f}% |")
    lines += ["", f"Reference FQE values: estimated behaviour **{ref:.3f}**; "
              f"behaviour cloning **{data['references']['bc']:.3f}**; "
              f"always-withhold control **{data['references']['withhold']:.3f}**. "
              "Each reference has one fit, shared across checkpoint columns; no seed SD is available.", "",
              "The behaviour reference is a fitted model of logged actions. It is not the observed clinician return.", "",
              f"Unsupported actions: estimated behaviour probability below {protocol['unsupported_threshold']}, "
              f"over {protocol['test_decisions']:,} test decisions. "
              f"Local support: fraction of {knn['k']} nearest training decisions choosing the recommendation, "
              f"averaged over {knn['n_query']:,} sampled test decisions against {knn['n_ref']:,} training decisions. "
              "Zero local support: no neighbour chose the recommendation. "
              f"Observed clinician actions have mean local support {data['support_references']['observed_clinician']['mean_local_support']:.3f}; "
              f"the withhold control has {data['support_references']['withhold']['mean_local_support']:.3f}.", "",
              "Early/final mean epoch 1 (5,000 updates)/epoch 40 for D3QN and CQL. "
              "For IQL they identify the critic used to train the actor; both actors are selected by validation loss.", "",
              "## Estimator diagnostics", "",
              "| Estimator | Defined within return bound | Defined outside ±15 | Withheld | Undefined |",
              "|---|---:|---:|---:|---:|"]
    v = estimator_verdicts(data)
    for est in ("WIS", "PDWIS", "DR", "FQE"):
        c = Counter(cat for (e, *_), cat in v.items() if e == est)
        lines.append(f"| {est} | {c['defined']} | {c['outside']} | {c['withheld']} | {c['undefined']} |")
    lines += ["", "Defined estimates are not necessarily reliable. WIS has very low effective sample sizes; "
              "FQE remains sensitive to extrapolation and evaluator fitting. These diagnostics do not measure clinical efficacy.", "",
              "## Saved paired comparisons", "",
              "Seed-level tests use five seeds and Holm adjustment across three methods × two checkpoints, "
              "separately for each reference. Values below are the saved research statistics, not new patient-level calculations.", "",
              "| Method | Checkpoint | Reference | Mean difference | Raw p | Holm p |",
              "|---|---|---|---:|---:|---:|"]
    for r in data["paired_tests"]:
        lines.append(f"| {LABELS[r['method']]} | {'Early' if r['checkpoint']=='best' else 'Final'} | "
                     f"{r['reference']} | {r['mean_diff']:+.4f} | {r['p']:.4f} | {r['p_holm']:.4f} |")
    lines += ["", "See [provenance and interpretation](results-provenance.md) for source files, checks and limitations.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "results/thesis_results.json")
    p.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    p.add_argument("--table", type=Path, default=ROOT / "docs/results.md")
    args = p.parse_args()
    data = json.loads(args.input.read_text())
    validate(data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "sepsis-thesis"})
    value_support(data, args.out_dir)
    counts = reliability(data, args.out_dir)
    table(data, args.table)
    print({e: dict(c) for e, c in counts.items()})
    print(f"Wrote two PNG/SVG figures to {args.out_dir} and table to {args.table}")


if __name__ == "__main__":
    main()
