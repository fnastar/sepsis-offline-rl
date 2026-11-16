"""
final_multiplicity.py: one Holm adjustment over the family that was declared.

The significance module runs once per checkpoint rule, so each process sees
three of the six comparisons made against a given reference: three methods
under two rules. Adjusting inside it would treat half a family as a whole one.
This reads both outputs, adjusts the six jointly, and writes the result back to
both so neither file can be read as final on its own.

`final_results.py` emits the same six comparisons against the behaviour
reference and six more against always-withhold. Those are adjusted here too,
within each reference family separately, so the headline table and the
significance artefacts cannot disagree about multiplicity.

Raw p-values are preserved everywhere. No threshold is applied, no binary
significance label is emitted, and neither the checkpoint rule nor the
reference family may be chosen after seeing which p-value is smallest. At five
seeds the interval carries more than the p-value does.

    python -m src.final_multiplicity
"""
import json
import os

import numpy as np

from config.paths import OUTPUT_BASE

SIG_DIR = os.path.join(OUTPUT_BASE, "significance")
CKPT_FILES = {"best": "fqe_significance_bestckpt.json",
              "final": "fqe_significance_finalckpt.json"}

# The core methods of this campaign. QR-D3QN was not run and must not inflate a
# family: adjusting six comparisons as though there were seven would be
# conservative in the wrong place and wrong in the reported family size.
CORE = ("CQL", "D3QN", "IQL")
CORE_LOWER = ("cql", "d3qn", "iql")
EXPECTED = len(CORE) * len(CKPT_FILES)   # three methods, two checkpoint rules


def holm(pairs: list) -> dict:
    """Holm-Bonferroni over (key, raw p). Returns key -> adjusted p."""
    finite = [(k, p) for k, p in pairs if p is not None and np.isfinite(p)]
    m, prev, out = len(finite), 0.0, {}
    for rank, (k, p) in enumerate(sorted(finite, key=lambda x: x[1])):
        prev = min(1.0, max(prev, (m - rank) * p))
        out[k] = prev
    return out


def _significance() -> dict:
    loaded = {}
    for ckpt, fn in CKPT_FILES.items():
        p = os.path.join(SIG_DIR, fn)
        if os.path.exists(p):
            loaded[ckpt] = (p, json.load(open(p)))
    if len(loaded) < 2:
        raise AssertionError(
            f"the declared family spans both checkpoint rules and only "
            f"{sorted(loaded)} is present in {SIG_DIR}. Adjusting one rule "
            f"would report half a family as a whole one.")

    pairs, missing = [], []
    for ckpt, (_, d) in loaded.items():
        res = d.get("results", {})
        for method in CORE:
            t = res.get(method, {}).get("test", {}).get("seed_level_t", {})
            p_raw = t.get("p_two_sided")
            if p_raw is None or not np.isfinite(p_raw):
                missing.append(f"{method}|{ckpt}")
            else:
                pairs.append((f"{method}|{ckpt}", p_raw))
    adj = holm(pairs)
    if len(adj) != EXPECTED:
        raise AssertionError(
            f"the behaviour-reference family needs {EXPECTED} finite "
            f"comparisons, three methods under two checkpoint rules, and has "
            f"{len(adj)}. Missing or undefined: {missing}. A Holm correction "
            f"over a partial family understates the adjustment.")

    for ckpt, (path, d) in loaded.items():
        for method, by_split in d.get("results", {}).items():
            t = by_split.get("test", {}).get("seed_level_t", {})
            key = f"{method}|{ckpt}"
            if key in adj:
                t["p_holm"] = float(adj[key])
                t["holm_family"] = ("three methods x two checkpoint rules "
                                    "against the behaviour reference, "
                                    "test split")
                t["holm_family_size"] = len(adj)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
    return adj


def _results_table() -> dict:
    p = os.path.join(OUTPUT_BASE, "final_results_test.json")
    if not os.path.exists(p):
        print("results table: not found, nothing to adjust")
        return {}
    d = json.load(open(p))
    tests = [t for t in d.get("paired_tests", [])
             if t.get("method") in CORE_LOWER]
    out, short = {}, {}
    for ref in sorted({t["reference"] for t in tests}):
        fam = [t for t in tests if t["reference"] == ref]
        adj = holm([(f"{t['method']}|{t['checkpoint']}", t.get("p"))
                    for t in fam])
        for t in fam:
            k = f"{t['method']}|{t['checkpoint']}"
            if k in adj:
                t["p_holm"] = round(float(adj[k]), 6)
                t["holm_family_size"] = len(adj)
                t["holm_family"] = f"three methods x two checkpoints vs {ref}"
        out[ref] = adj
        if len(adj) != EXPECTED:
            short[ref] = (len(adj), EXPECTED)
    d["holm_family_sizes"] = {ref: len(a) for ref, a in out.items()}
    with open(p, "w") as f:
        json.dump(d, f, indent=2)
    if short or not out:
        raise AssertionError(
            f"each results-table reference family needs {EXPECTED} "
            f"comparisons and has {short or 'none at all'}. The headline "
            f"table would then disagree with the significance artefacts about "
            f"multiplicity.")
    return out


def run() -> dict:
    # Both are attempted before either can stop the step, so a partial state is
    # fully described rather than reported one failure at a time.
    errors = []
    try:
        sig = _significance()
    except AssertionError as e:
        sig, _ = {}, errors.append(str(e))
    try:
        res = _results_table()
    except AssertionError as e:
        res, _ = {}, errors.append(str(e))
    if sig:
        print(f"\nJoint Holm over {len(sig)} comparisons against the behaviour "
              f"reference, test split:")
        for k, v in sorted(sig.items(), key=lambda x: x[1]):
            print(f"  {k:<16} Holm {v:.4g}")
    for ref, adj in res.items():
        print(f"\nResults table, family vs {ref} ({len(adj)} comparisons):")
        for k, v in sorted(adj.items(), key=lambda x: x[1]):
            print(f"  {k:<16} Holm {v:.4g}")
    print("\nRaw p-values are preserved. No threshold is applied, and at five "
          "seeds the interval carries more than the p-value.")

    summary = {"expected_family_size": EXPECTED, "core_methods": list(CORE),
               "behaviour_family_size": len(sig),
               "results_table_family_sizes": {k: len(v) for k, v in res.items()},
               "complete": not errors, "errors": errors}
    from src.utils import ensure_dir
    ensure_dir(OUTPUT_BASE)
    sp = os.path.join(OUTPUT_BASE, "multiplicity_summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {sp}")
    if errors:
        raise AssertionError(" | ".join(errors))
    return {"significance": sig, "results_table": res, "summary": summary}


if __name__ == "__main__":
    run()
