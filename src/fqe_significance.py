"""
fqe_significance.py: paired test of FQE(policy) against FQE(behaviour reference).

Tests the paired per-stay difference
V̂_policy(s_0) − V̂_behaviour(s_0), both fitted by FQE and evaluated on the same
initial states.

WHAT THE COMPARISON IS. The second arm is an **estimated behaviour policy**, an
MLP classifier fitted to the logged actions, not the logged clinician. The
observed discounted return is a separate quantity. On this cohort the behaviour
reference's FQE value sits well *above* the empirical return, where on the
earlier cohort it sat far below, so the direction of the absolute-scale
discrepancy is not stable across datasets and no absolute claim is available in
either direction.

Only a **shared** component of evaluator error cancels in the difference. A
policy-dependent component does not, and that is the unresolved concern rather
than a solved one: the two arms are separately fitted FQE regressions over
policies with different action distributions and different data support. The
sign of this test is the sign of an estimator-dependent ranking. It is not
evidence about the sign of a true difference in clinical value.

WHAT IS CONDITIONED ON. A single fixed behaviour-reference fit, and the held-out
patients common to both arms. Five training seeds quantify observed
training-seed variability and nothing else. Validation-split results are
diagnostic, not a second confirmatory endpoint.

Three uncertainty treatments are reported because they target three different
estimands, and being explicit about which is which is the point:

  (A) patient-only bootstrap — resamples stays, conditioning on the fitted
      networks. Answers "for THESE trained policies, is the mean patient
      difference > 0?". Narrow; understates method-level uncertainty. NOT a
      headline: it omits the between-seed variance and corresponds to an
      averaged-net policy that is not deployable. Report only as an
      explicitly-labelled "conditional on the fitted nets" ablation, if at all.
  (B) hierarchical bootstrap — resamples training seeds AND stays. Targets the
      method-level estimand, but a nonparametric percentile bootstrap over only
      ~5 outer units cannot inflate for the uncertainty in the between-seed SD,
      so it UNDERCOVERS. Cross-check only.
  (C) seed-level t — treats each seed's mean difference as one observation
      (n = number of seeds, df = n−1). Same method-level estimand as (B) with
      the small-sample correction (B) lacks. **PRIMARY interval.**

Lead with the per-seed differences and the interval, not the p-value. At n=5 a
p-value carries little information and invites a threshold that was never
pre-specified.

The `two_component_se` reported alongside (C) is an approximate crossed-
components check, sqrt(patient_se² + seed_se²). It assumes the two sources are
independent and that each is estimated adequately at this sample size. It is a
diagnostic for whether (C) omits a large patient component, not proof that (C)
is a complete uncertainty interval.

MULTIPLICITY. Three methods are tested under two checkpoint rules against each
reference, so each reference family carries six test-split comparisons. Raw
p-values are preserved and Holm-adjusted p-values are reported within each
declared family. Neither the checkpoint rule nor the reference family may be
chosen after seeing which p-value is smallest, and no binary "significant"
label is emitted anywhere.

Usage (90-day namespace, primary outcome):
  SEPSIS_PROCESSED_DIR=data/processed_90d \
  SEPSIS_OUTPUT_BASE=outputs/runs_90d python -m src.fqe_significance
  # optional: restrict to one policy family, e.g. cql
  SEPSIS_PROCESSED_DIR=... python -m src.fqe_significance cql
"""
import json
import os
import sys

import numpy as np
from scipy import stats

from config.paths import OUTPUT_BASE
from src.fqe_v0 import load_v0, load_clinician_v0, pair, verify_against_stored

N_BOOT = 20000
BOOT_SEED = 42

# Policy families to test. Each entry: (display name, run-dir prefix, net kind,
# seed range). All use the D3QN forward path for FQE (QR uses q_values()).
FAMILIES = {
    "CQL":  ("cql_a2.0_seed", "d3qn", range(42, 47)),
    "D3QN": ("d3qn_seed",     "d3qn", range(42, 47)),
    "IQL":  ("iql_seed",      "iql",  range(42, 47)),
    "QR":   ("qr_d3qn_seed",   "qr",   range(42, 47)),
}


def _stack_differences(prefix, kind, seeds, split, clin_ids, clin_v0):
    """(seeds × stays) matrix of paired per-stay differences policy − clinician.

    Every seed is aligned to the clinician's stay ordering via `pair`, so column
    j is the same stay across all rows and across the clinician arm.
    """
    rows, used = [], []
    for s in seeds:
        run = f"{prefix}{s}"
        run_dir = os.path.join(OUTPUT_BASE, run)
        if not os.path.isdir(run_dir):
            continue
        # A run with no persisted v0 and no fitted FQE net cannot be recovered.
        # Skip it (with a note) rather than crash.
        # NOTE (2026-07-24): this handler used to silently swallow the entire QR
        # family. QR's FQE nets *are* saved — under `fqe_mean_{split}` /
        # `fqe_q25_{split}`, not `fqe_{split}` — and `fqe_v0` looked in the wrong
        # place. Fixed via `fqe_v0.fqe_subdir()`. If a whole family disappears
        # from the results again, suspect a naming convention before believing
        # the artefacts are missing.
        try:
            verify_against_stored(run, split, kind)   # refuses silent v0 drift
            ids, v0 = load_v0(run, split, kind)
        except FileNotFoundError as e:
            print(f"  (skip {run}/{split}: {os.path.basename(str(e).split()[-1])} "
                  f"— no recoverable v0)")
            continue
        _, pv, cv = pair(ids, v0, clin_ids, clin_v0)
        rows.append(pv - cv)
        used.append(s)
    if not rows:
        return None, []
    return np.vstack(rows), used


def _patient_only(D, rng):
    """(A) Resample stays; condition on the fitted nets via the seed-averaged
    per-stay difference."""
    dbar = D.mean(axis=0)
    n = len(dbar)
    idx = rng.integers(0, n, (N_BOOT, n))
    boot = dbar[idx].mean(axis=1)
    return {
        "mean": float(dbar.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "p_le_0": float((boot <= 0).mean()),
        "se": float(dbar.std(ddof=1) / np.sqrt(n)),
    }


def _hierarchical(D, rng):
    """(B) Resample seeds (rows) and stays (cols) jointly."""
    n_seed, n = D.shape
    boot = np.empty(N_BOOT)
    for k in range(N_BOOT):
        si = rng.integers(0, n_seed, n_seed)
        st = rng.integers(0, n, n)
        boot[k] = D[si][:, st].mean()
    return {
        "mean": float(D.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "p_le_0": float((boot <= 0).mean()),
    }


def _seed_t(D):
    """(C) Seed as the unit; t-test of the per-seed mean differences vs 0."""
    sm = D.mean(axis=1)
    n = len(sm)
    mean, sd = float(sm.mean()), float(sm.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    if n >= 2:
        t, p = stats.ttest_1samp(sm, 0.0)
        t, p = float(t), float(p)
    else:
        t, p = float("nan"), float("nan")
    return {
        "per_seed_means": [float(x) for x in sm],
        "mean": mean, "sd": sd, "se": se, "df": n - 1,
        "ci95": [mean - tcrit * se, mean + tcrit * se],
        "t": t, "p_two_sided": p, "all_positive": bool((sm > 0).all()),
    }


def _completeness(D):
    """Two-component SE vs the seed-only SE.

    Guards the choice of (C) as primary against the objection that it omits
    patient-sampling variance: if the patient component is negligible relative
    to the seed component, the seed-only interval is effectively complete.
    """
    dbar = D.mean(axis=0)
    patient_se = float(dbar.std(ddof=1) / np.sqrt(len(dbar)))
    seed_se = float(D.mean(axis=1).std(ddof=1) / np.sqrt(D.shape[0]))
    two_comp = float(np.hypot(patient_se, seed_se))
    return {
        "patient_se": patient_se, "seed_se": seed_se,
        "two_component_se": two_comp,
        "inflation_over_seed_se": two_comp / seed_se if seed_se else float("nan"),
    }


def test_family(name, prefix, kind, seeds, split, clin_ids, clin_v0):
    D, used = _stack_differences(prefix, kind, seeds, split, clin_ids, clin_v0)
    if D is None:
        return None
    rng = np.random.default_rng(BOOT_SEED)
    sm = D.mean(axis=1)
    return {
        "policy": name, "split": split, "n_seeds": len(used), "seeds": used,
        "n_stays": int(D.shape[1]),
        "primary": "seed_level_t",
        "worst_seed_floor": float(sm.min()),
        "all_seeds_positive": bool((sm > 0).all()),
        "completeness_check": _completeness(D),
        "patient_only_bootstrap": _patient_only(D, rng),
        "hierarchical_bootstrap": _hierarchical(D, rng),
        "seed_level_t": _seed_t(D),
    }


def _print(res):
    a, b, c = (res["patient_only_bootstrap"], res["hierarchical_bootstrap"],
               res["seed_level_t"])
    comp = res["completeness_check"]
    print(f"\n{res['policy']}  [{res['split']}]  "
          f"n={res['n_stays']:,} stays × {res['n_seeds']} seeds")
    print(f"  per-seed diffs: {[round(x, 3) for x in c['per_seed_means']]}  "
          f"(all positive: {c['all_positive']}, worst seed {res['worst_seed_floor']:+.3f})")
    print(f"  (C) seed t ({c['df']}df) [PRIMARY]: {c['mean']:+.4f}  "
          f"95% CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  "
          f"t={c['t']:.2f}  p={c['p_two_sided']:.2e}")
    print(f"  (B) hierarchical [cross-check, undercovers at n=5]: {b['mean']:+.4f}  "
          f"95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]")
    print(f"  (A) patient-only [ablation only, do NOT lead]: {a['mean']:+.4f}  "
          f"95% CI [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}]")
    infl = 100 * (comp["inflation_over_seed_se"] - 1)
    # An approximate crossed-components check, not a completeness proof. It
    # assumes the patient and seed components are independent and that each is
    # estimated adequately at this sample size, neither of which is established
    # at five seeds. It says whether the seed-only interval omits a large
    # patient component, and nothing more.
    tag = ("patient component small relative to the seed component" if infl < 15
           else "patient component NOT small; the seed-only interval omits it")
    print(f"  crossed-components check (diagnostic, assumes independence): "
          f"patient_SE={comp['patient_se']:.4f} "
          f"seed_SE={comp['seed_se']:.4f} "
          f"quadrature_SE={comp['two_component_se']:.4f} "
          f"(+{infl:.0f}% over seed-only, {tag})")


def main():
    only = sys.argv[1].upper() if len(sys.argv) > 1 else None
    fams = {only: FAMILIES[only]} if only in FAMILIES else FAMILIES

    out = {"note": ("Paired per-stay FQE difference, policy minus the estimated "
                    "behaviour policy. The sign is the sign of an "
                    "estimator-dependent ranking, not of a difference in "
                    "clinical value. Three uncertainty treatments: (C) "
                    "seed-level t is primary, (B) hierarchical corroborates, "
                    "(A) conditions on the fitted nets and never leads."),
           "reference": ("estimated behaviour policy, an MLP classifier fitted "
                         "to the logged actions; not the logged clinician"),
           "p_values": ("raw, with Holm adjustment within the test-split "
                        "family of this reference. No significance threshold "
                        "is applied anywhere, and neither the checkpoint rule "
                        "nor the family may be chosen after seeing them."),
           "n_boot": N_BOOT, "results": {}}

    for split in ("val", "test"):
        clin_ids, clin_v0 = load_clinician_v0(split)
        for name, (prefix, kind, seeds) in fams.items():
            res = test_family(name, prefix, kind, seeds, split, clin_ids, clin_v0)
            if res is None:
                print(f"  (skip {name}/{split}: no runs found)")
                continue
            _print(res)
            out["results"].setdefault(name, {})[split] = res

    # Holm within the test-split family: the methods compared against this
    # reference under this checkpoint rule. Ordering is fixed by the raw
    # p-values, which are preserved untouched beside the adjusted ones.
    fam_p = [(n, r["test"]["seed_level_t"]["p_two_sided"])
             for n, r in out["results"].items()
             if "test" in r and "seed_level_t" in r["test"]
             and np.isfinite(r["test"]["seed_level_t"].get("p_two_sided", np.nan))]
    m = len(fam_p)
    prev = 0.0
    for rank, (n, praw) in enumerate(sorted(fam_p, key=lambda x: x[1])):
        adj = min(1.0, max(prev, (m - rank) * praw))
        prev = adj
        # INTERIM. This process sees one checkpoint rule, so this adjusts
        # three of the six comparisons in the declared family.
        # `src/final_multiplicity.py` runs after both invocations and writes
        # the joint six-value adjustment over the top.
        out["results"][n]["test"]["seed_level_t"]["p_holm_interim"] = float(adj)
        out["results"][n]["test"]["seed_level_t"]["holm_family_size_interim"] = m
    if m:
        print(f"\nINTERIM Holm over {m} of the six comparisons in the "
              f"declared family. The joint adjustment across both checkpoint "
              f"rules is written by src.final_multiplicity:")
        for n, praw in sorted(fam_p, key=lambda x: x[1]):
            a = out["results"][n]["test"]["seed_level_t"]["p_holm_interim"]
            print(f"  {n:<6} raw {praw:.4g}   Holm {a:.4g}")
        print("  No threshold is applied. At n=5 the interval carries more "
              "than the p-value.")

    out_dir = os.path.join(OUTPUT_BASE, "significance")
    os.makedirs(out_dir, exist_ok=True)
    # The filename carries the checkpoint, so running both rules does not
    # leave one silently overwriting the other.
    from src.fqe_v0 import FQE_SUFFIX
    tag = FQE_SUFFIX or "_bestckpt"
    out["checkpoint"] = "final epoch" if FQE_SUFFIX else "lowest validation loss"
    path = os.path.join(out_dir, f"fqe_significance{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
