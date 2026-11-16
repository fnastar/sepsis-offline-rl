"""
Tests for dataset signoff, experiment completion and result collection.
"""
import os

import pytest


def _require(monkeypatch, tmp_path, env_ok, write_file):
    import src.final_campaign as fc
    monkeypatch.setattr(fc, "STATUS_DIR", str(tmp_path))
    if env_ok:
        monkeypatch.setenv("SEPSIS_SEEDS_OK", "1")
    else:
        monkeypatch.delenv("SEPSIS_SEEDS_OK", raising=False)
    if write_file:
        (tmp_path / "SIGNOFF").write_text("ok\n")
    return fc


def test_gate_blocks_without_the_file(monkeypatch, tmp_path):
    """
    The environment variable alone must not authorise the campaign.
    """
    fc = _require(monkeypatch, tmp_path, env_ok=True, write_file=False)
    with pytest.raises(SystemExit):
        fc._require_signoff("seeds")


def test_gate_blocks_without_the_variable(monkeypatch, tmp_path):
    """A stale sign-off file left behind must not release a fresh campaign."""
    fc = _require(monkeypatch, tmp_path, env_ok=False, write_file=True)
    with pytest.raises(SystemExit):
        fc._require_signoff("seeds")


def test_gate_opens_with_both(monkeypatch, tmp_path):
    fc = _require(monkeypatch, tmp_path, env_ok=True, write_file=True)
    fc._require_signoff("seeds")


def test_pilot_is_not_gated():
    """
    The pilot exists to exercise the pipeline, so it runs on any dataset.

    Only the stages that cost hours and produce reported numbers are gated.
    """
    import inspect
    import src.final_campaign as fc
    assert "_require_signoff" not in inspect.getsource(fc.stage_pilot)
    for stage in (fc.stage_iql, fc.stage_seeds):
        assert "_require_signoff" in inspect.getsource(stage)


# --- the pilot summary, against the current FQE key schema -------------------


def _fqe_shaped_results():
    """Exactly the shape `fqe_for` returns: keyed "{checkpoint}_{split}"."""
    return {
        "best_val":   {"fqe_value": 12.1, "empirical_clinician_return": 6.94},
        "best_test":  {"fqe_value": 12.2, "empirical_clinician_return": 6.88},
        "final_val":  {"fqe_value": 11.9, "empirical_clinician_return": 6.94},
        "final_test": {"fqe_value": 12.0, "empirical_clinician_return": 6.88},
    }


def test_empirical_return_reads_the_current_key_schema():
    """
    Read the split from checkpoint-qualified FQE result keys.
    """
    from src.final_campaign import empirical_return
    r = _fqe_shaped_results()
    assert empirical_return(r, "test") == 6.88
    assert empirical_return(r, "val") == 6.94


def test_empirical_return_is_none_when_absent():
    """A missing value must not raise inside a summary the stage depends on."""
    from src.final_campaign import empirical_return
    assert empirical_return({"best_test": {"fqe_value": 1.0}}, "test") is None
    assert empirical_return({}, "test") is None


def test_pilot_summary_does_not_index_fqe_by_a_bare_split():
    """Guards the specific expression, since the shape is easy to reintroduce."""
    import inspect
    import src.final_campaign as fc
    src = inspect.getsource(fc.stage_pilot)
    for split in ("val", "test"):
        assert f'["{split}"]' not in src


# --- run completion ---------------------------------------------------------


def _run_dir(tmp_path, final=True, rows=40):
    (tmp_path / "d3qn_best.pt").write_bytes(b"x")
    if final:
        (tmp_path / "d3qn_final.pt").write_bytes(b"x")
    if rows is not None:
        lines = ["epoch,train_loss"] + [f"{i},1.0" for i in range(rows)]
        (tmp_path / "training_history.csv").write_text("\n".join(lines) + "\n")
    return str(tmp_path)


def test_a_run_stopped_after_epoch_one_is_not_complete(tmp_path):
    """
    `d3qn_best.pt` appears after the first epoch.

    Treating its presence as completion would let a crashed run resume as done
    and then be scored as one of the five seeds.
    """
    from src.final_campaign import is_complete
    assert not is_complete(_run_dir(tmp_path, final=False, rows=1))


def test_a_run_with_a_short_history_is_not_complete(tmp_path):
    from src.final_campaign import is_complete
    assert not is_complete(_run_dir(tmp_path, final=True, rows=7))


def test_a_finished_run_is_complete(tmp_path):
    from src.final_campaign import is_complete
    assert is_complete(_run_dir(tmp_path, final=True, rows=40))


def test_an_unreadable_history_is_not_complete(tmp_path):
    from src.final_campaign import is_complete
    d = _run_dir(tmp_path, final=True, rows=None)
    (tmp_path / "training_history.csv").write_bytes(b"\x00\x01\x02")
    assert not is_complete(d)


def test_a_diverged_run_is_not_complete(tmp_path):
    """
    `train.py` writes DIVERGED and stops, which can happen on the last epoch.

    Such a run has a full-length history and a final checkpoint, so length
    alone would accept it.
    """
    from src.final_campaign import is_complete
    d = _run_dir(tmp_path, final=True, rows=40)
    (tmp_path / "DIVERGED").write_text("epoch 40\n")
    assert not is_complete(d)


def test_support_actor_filename_selects_the_critic(monkeypatch):
    """
    The final support run must load the actor extracted against the final
    critic, not the early one. Passing the path without the critic argument
    made the two support runs silently identical.
    """
    import importlib
    import src.data_support as ds
    monkeypatch.setenv("SEPSIS_SUPPORT_CKPT", "final")
    importlib.reload(ds)
    assert ds.CHECKPOINTS["IQL"][0].endswith("actor_criticfinal.pt")
    assert ds.CHECKPOINTS["CQL (alpha=2.0)"][0].endswith("d3qn_final.pt")
    monkeypatch.setenv("SEPSIS_SUPPORT_CKPT", "best")
    importlib.reload(ds)
    assert ds.CHECKPOINTS["IQL"][0].endswith("actor_best.pt")
    assert ds.CHECKPOINTS["CQL (alpha=2.0)"][0].endswith("d3qn_best.pt")


# --- the doubly-robust control variate must match the policy ----------------


def test_only_greedy_gets_the_fitted_fqe_control_variate():
    """
    The saved FQE net was fitted under the greedy policy, so it is Q of that
    policy. Handing it to the softmax row would put the wrong policy's value
    function in the control variate, which is the half of the estimator that
    is supposed to be model-correct.
    """
    import inspect
    import src.final_ope as fo
    src = inspect.getsource(fo.run)
    assert '_fqe_net_for(' in src
    assert 'if temp == "greedy" else None' in src


def test_evaluate_policy_labels_which_model_dr_used():
    """A DR value is not interpretable without knowing its control variate."""
    import inspect
    import src.evaluate as ev
    src = inspect.getsource(ev.evaluate_policy)
    assert '"dr_control_variate"' in src
    assert "q_hat_net or q_net" in src


def test_evaluate_policy_reports_both_effective_sample_sizes():
    """
    WIS clips weights at the 99th percentile before averaging, so an ESS from
    unclipped weights describes a different estimator from the WIS value it is
    printed beside.
    """
    import inspect
    import src.evaluate as ev
    src = inspect.getsource(ev.evaluate_policy)
    assert "ess_clipped" in src and "clip_weights(w_norm)" in src


def test_expected_fqe_row_count_matches_what_fqe_for_emits():
    """
    The cardinality assertion has to track `fqe_for`, not a remembered number.

    CQL and D3QN are scored under both checkpoint rules on both splits, four
    rows each. IQL's policy is its actor, so the Q-net rule does not apply and
    it yields two, with the final-critic variant scored separately in
    stage_iqlfinal. Five seeds gives 20 + 20 + 10.
    """
    import inspect
    import src.final_campaign as fc
    src = inspect.getsource(fc.fqe_for)
    # IQL skips the non-best checkpoint, which is what makes it two not four
    assert 'if ckpt != "best":' in src and "continue" in src
    assert len(fc.SEEDS) == 5
    expected = len(fc.SEEDS) * (2 * 2 + 2 * 2 + 2)
    assert expected == 50
    assert f"expected = len(SEEDS) * (2 * 2 + 2 * 2 + 2)" in \
        inspect.getsource(fc.stage_fqe)


# --- FQE fit reuse ----------------------------------------------------------


def test_fqe_fit_is_reused_across_splits():
    """
    The FQE fit uses training transitions only, so it does not depend on the
    split; the split selects the initial states it is scored on. Fitting twice
    produced byte-identical networks and doubled the dominant stage.
    """
    import inspect
    import src.final_campaign as fc
    for fn in (fc.fqe_for, fc.run_references, fc.stage_iqlfinal):
        src = inspect.getsource(fn)
        assert "reuse_from=fitted" in src, fn.__name__
        assert "fitted = fitted or d" in src, fn.__name__


def test_a_reused_fit_is_verified_against_its_source():
    """A reused network must be the one it claims to be, not merely present."""
    import inspect
    import src.final_fqe as ff
    src = inspect.getsource(ff.run_fqe)
    assert "_sha256(net_path) == reused_hash" in src
    assert '"reused_fit_from"' in src


def test_fqe_result_records_the_loss_minimum():
    """
    FQE training loss reaches its minimum at iteration 8 and rises by about a
    third by iteration 20, so the final loss alone does not describe the fit.
    Every result carries the minimum and where it occurred.
    """
    import inspect
    import src.final_fqe as ff
    src = inspect.getsource(ff.run_fqe)
    assert '"min_train_loss"' in src and '"argmin_train_loss"' in src


# --- subgroup artefact paths ------------------------------------------------


def test_subgroup_maps_each_family_to_its_own_checkpoint_directory():
    """
    IQL's final variant is the actor re-extracted against the epoch-40 critic,
    so it lives under `_criticfinal`, not `_finalckpt`. Mapping it to the plain
    directory made the final subgroup table repeat the early one.
    """
    from src.final_subgroup import fqe_dir
    assert fqe_dir("cql", "test", "best") == "fqe_test"
    assert fqe_dir("cql", "test", "final") == "fqe_test_finalckpt"
    assert fqe_dir("d3qn", "val", "final") == "fqe_val_finalckpt"
    assert fqe_dir("iql", "test", "best") == "fqe_test"
    assert fqe_dir("iql", "test", "final") == "fqe_test_criticfinal"
    # the two rules must never resolve to the same place for any family
    for kind in ("cql", "d3qn", "iql"):
        assert fqe_dir(kind, "test", "best") != fqe_dir(kind, "test", "final")


def test_subgroup_refuses_an_incomplete_seed_set():
    """
    It records what it expected and found, writes the partial table, and only
    then raises, so the evidence survives the failure.
    """
    import inspect
    import src.final_subgroup as fs
    src = inspect.getsource(fs.run)
    assert "is missing seeds" in src
    for k in ("seeds_expected", "seeds_missing", "complete"):
        assert f'"{k}"' in src, k
    # the raise has to come after the file is written
    assert src.index("json.dump(out") < src.index("Refusing to report")


def test_subgroup_selects_one_actual_initial_row():
    """
    Pandas `first()` takes the first non-null value of each column
    independently, so it can pair `sofa_observed = 0` from the initial row with
    a `sofa_total` recorded hours later. The initial state must be one row.
    """
    import inspect
    import src.final_subgroup as fs
    src = inspect.getsource(fs.run)
    assert ".head(1)" in src and ".first()" not in src


def test_subgroup_reports_the_missing_sofa_group():
    """
    A third of test stays have no SOFA at their initial decision point and were
    being dropped from every summary. They are a missingness group, not a
    fourth severity band.
    """
    import inspect
    import src.final_subgroup as fs
    src = inspect.getsource(fs.run)
    assert '"sofa_missing_at_initial_state"' in src
    assert '"mortality_90d_pct"' in src


def test_actor_checkpoint_alone_is_not_a_complete_extraction(tmp_path):
    """
    `train_actor` saves its checkpoint inside the epoch loop and writes the
    hyperparameters after it, so a crash leaves a checkpoint that `load_actor`
    cannot use. The hyperparameters file is the completion marker.
    """
    from src.final_campaign import actor_is_complete
    (tmp_path / "actor_best.pt").write_bytes(b"x")
    assert not actor_is_complete(str(tmp_path))
    (tmp_path / "actor_hyperparams.json").write_text("{}")
    assert not actor_is_complete(str(tmp_path))
    (tmp_path / "actor_history.csv").write_text("epoch\n0\n")
    assert actor_is_complete(str(tmp_path))


def test_final_critic_actor_has_its_own_completion_marker(tmp_path):
    from src.final_campaign import actor_is_complete
    for f in ("actor_best.pt", "actor_hyperparams.json", "actor_history.csv"):
        (tmp_path / f).write_text("x")
    assert actor_is_complete(str(tmp_path), "best")
    assert not actor_is_complete(str(tmp_path), "final")


def test_fqe_result_json_is_written_last():
    """
    Every caller skips a directory that already has `fqe_result.json`, so that
    file has to be the completion marker. Written before `v0.npz`, a crash
    between them would leave a fit that reads as done but whose per-stay values
    the paired test cannot find, dropping the policy from the comparison in
    silence.
    """
    import inspect
    import src.final_fqe as ff
    src = inspect.getsource(ff.run_fqe)
    assert src.index("v0.npz") < src.rindex("fqe_result.json")


def test_seed42_shortcut_requires_an_actor_for_iql():
    """
    Seed 42 is trained by the pilot and the IQL gate, so `stage_seeds` reuses
    it rather than retraining. For IQL the critic alone is not enough: without
    the extracted actor the run is a complete instance of a method the thesis
    does not evaluate.
    """
    import inspect
    import src.final_campaign as fc
    src = inspect.getsource(fc.stage_seeds)
    assert 'kind != "iql" or actor_is_complete(run)' in src


def test_action_profile_skips_incomplete_runs():
    """
    A checkpoint appears after the first epoch, so the profile module has to
    use the same completion test as everything else. Profiling a half-trained
    policy puts a number in a table that describes nothing, and it happened:
    a seed still training was picked up mid-run.
    """
    import inspect
    import src.final_action_profile as fa
    src = inspect.getsource(fa.run)
    assert "is_complete(run_dir)" in src
    assert "actor_is_complete(" in src


# --- exact deterministic importance weights ---------------------------------


def test_a_mismatching_action_carries_exactly_zero_weight():
    """
    A deterministic policy assigns zero to actions it would not take, so a
    logged trajectory that disagrees must carry zero weight. Flooring the
    target at 1e-12 gave every mismatching trajectory a tiny positive weight,
    and because WIS self-normalises the scale then cancels, returning a
    plausible number computed from the least mismatching trajectories.
    """
    import numpy as np
    from src.evaluate import trajectory_log_weights, n_supported

    class T:
        n = 2
        actions = [np.array([0, 1]), np.array([0, 0])]

    pi_e = np.zeros((4, 25)); pi_e[:, 0] = 1.0          # deterministic: always 0
    pi_b = np.full((4, 25), 1.0 / 25)
    log_w, step = trajectory_log_weights(T(), pi_e, pi_b)
    assert not np.isfinite(log_w[0])                    # took action 1 at t=1
    assert np.isfinite(log_w[1])                        # matched throughout
    assert n_supported(log_w) == 1


def test_wis_is_undefined_without_support():
    """Zero support must not be normalised into a number."""
    import numpy as np
    from src.evaluate import wis_estimator
    log_w = np.array([-np.inf, -np.inf])
    assert np.isnan(wis_estimator(log_w, np.array([1.0, 2.0])))


def test_softmax_policies_keep_every_trajectory():
    """The fix must not touch the stochastic sensitivity, where nothing is zero."""
    import numpy as np
    from src.evaluate import trajectory_log_weights, n_supported

    class T:
        n = 2
        actions = [np.array([0, 1]), np.array([3, 4])]

    pi_e = np.full((4, 25), 1.0 / 25)
    pi_b = np.full((4, 25), 1.0 / 25)
    log_w, _ = trajectory_log_weights(T(), pi_e, pi_b)
    assert n_supported(log_w) == 2
    assert np.allclose(log_w, 0.0)


def test_clipping_uses_positive_weights_only():
    """
    With one supported trajectory in thousands, a percentile over all weights
    is zero and would clip the only valid weight away, turning a thin estimate
    into no estimate.
    """
    import numpy as np
    from src.evaluate import clip_weights
    w = np.zeros(1000); w[0] = 5.0
    assert clip_weights(w)[0] == 5.0
    assert clip_weights(np.zeros(10)).sum() == 0.0


def test_bootstrap_reports_its_defined_fraction():
    """
    A resample can contain no supported trajectory. Those replicates are
    undefined rather than zero, and an interval resting on a handful of defined
    ones has to say so.
    """
    import numpy as np
    from src.evaluate import bootstrap_ci
    # defined only when index 0 is drawn
    fn = lambda idx: 1.0 if (idx == 0).any() else float("nan")
    mean, lo, hi, frac = bootstrap_ci(fn, 5, n_boot=100, seed=1)
    assert 0.0 < frac <= 1.0 and mean == 1.0
    mean, lo, hi, frac = bootstrap_ci(lambda idx: float("nan"), 5, n_boot=20)
    assert frac == 0.0 and np.isnan(mean)


def test_pdwis_and_dr_are_undefined_when_a_live_step_has_no_support():
    """
    A per-decision estimator sums over every step, and a step with no support
    contributes zero through denominator clipping, silently truncating the
    horizon. Checking only whether any trajectory matches end to end misses
    this: a policy can have supported prefixes and no supported tail.

    On the real data CQL-early has 252 trajectories still running at decision
    index 16 with zero support, and the estimators returned 19.262 and 35.437
    before this.
    """
    import inspect
    import src.evaluate as ev
    src = inspect.getsource(ev.evaluate_policy)
    assert "unsupported_steps" in src
    assert "n_match == 0 or unsupported_steps" in src


def test_ope_expected_grid_is_derived_from_the_grid_it_iterates():
    """
    The completeness constant referred to a name the module does not define,
    so the step would have written its CSV and then raised NameError. It is
    now computed from the run list actually built.
    """
    import inspect
    import src.final_ope as fo
    src = inspect.getsource(fo.run)
    assert "expected = len(runs) * 2 * len(splits) * len(policies)" in src
    assert "len(METHODS)" not in src


def test_ope_rows_carry_the_support_detail():
    """The support vectors are what make an estimate interpretable."""
    import inspect
    import src.final_ope as fo
    src = inspect.getsource(fo.run)
    for k in ("n_supported_by_step", "n_active_by_step", "unsupported_steps",
              "wis_ci_defined_fraction"):
        assert f'"{k}"' in src, k


# --- multiplicity across both checkpoint rules -------------------------------


def test_holm_spans_both_checkpoint_files(tmp_path, monkeypatch):
    """
    The declared family is three methods under two checkpoint rules against one
    reference, six comparisons. The significance module runs once per rule and
    sees three, so adjusting inside it treats half a family as a whole one.
    """
    import json
    import importlib
    import src.final_multiplicity as fm
    sig = tmp_path / "significance"
    sig.mkdir()
    raw = {"best": {"CQL": 0.001, "IQL": 0.010, "D3QN": 0.200},
           "final": {"CQL": 0.004, "IQL": 0.050, "D3QN": 0.900}}
    for ckpt, fn in fm.CKPT_FILES.items():
        d = {"results": {m: {"test": {"seed_level_t": {"p_two_sided": p}}}
                         for m, p in raw[ckpt].items()}}
        (sig / fn).write_text(json.dumps(d))
    monkeypatch.setattr(fm, "SIG_DIR", str(sig))
    monkeypatch.setattr(fm, "OUTPUT_BASE", str(tmp_path))
    adj = fm._significance()

    assert len(adj) == 6, "the family spans both checkpoint rules"
    # smallest raw p gets multiplied by the full family size
    assert abs(adj["CQL|best"] - 6 * 0.001) < 1e-12
    # and the adjustment is written back into both files
    for fn in fm.CKPT_FILES.values():
        d = json.loads((sig / fn).read_text())
        for m in d["results"]:
            t = d["results"][m]["test"]["seed_level_t"]
            assert t["holm_family_size"] == 6


def test_holm_is_monotone_and_bounded():
    from src.final_multiplicity import holm
    adj = holm([("a", 0.001), ("b", 0.01), ("c", 0.2), ("d", 0.9)])
    vals = [adj[k] for k in ("a", "b", "c", "d")]
    assert vals == sorted(vals) and max(vals) <= 1.0


def test_holm_ignores_undefined_p_values():
    """A comparison with too few seeds has no p-value and must not inflate m."""
    import numpy as np
    from src.final_multiplicity import holm
    adj = holm([("a", 0.01), ("b", float("nan")), ("c", None)])
    assert set(adj) == {"a"} and abs(adj["a"] - 0.01) < 1e-12


def test_significance_module_labels_its_adjustment_interim():
    """The per-invocation value must not be mistaken for the joint one."""
    import inspect
    import src.fqe_significance as fs
    src = inspect.getsource(fs.main)
    assert "p_holm_interim" in src and '"p_holm"]' not in src


def test_multiplicity_refuses_a_single_checkpoint_file(tmp_path, monkeypatch):
    """
    Returning quietly when one rule is missing let the orchestrator mark the
    step OK while the declared adjustment was never produced.
    """
    import json
    import pytest as _pytest
    import src.final_multiplicity as fm
    sig = tmp_path / "significance"; sig.mkdir()
    (sig / fm.CKPT_FILES["best"]).write_text(json.dumps(
        {"results": {m: {"test": {"seed_level_t": {"p_two_sided": 0.01}}}
                     for m in fm.CORE}}))
    monkeypatch.setattr(fm, "SIG_DIR", str(sig))
    with _pytest.raises(AssertionError, match="both checkpoint rules"):
        fm._significance()


def test_multiplicity_refuses_a_short_family(tmp_path, monkeypatch):
    """Five finite comparisons out of six is not the declared family."""
    import json
    import pytest as _pytest
    import src.final_multiplicity as fm
    sig = tmp_path / "significance"; sig.mkdir()
    full = {m: {"test": {"seed_level_t": {"p_two_sided": 0.01}}} for m in fm.CORE}
    short = dict(full); short["IQL"] = {"test": {"seed_level_t": {}}}
    (sig / fm.CKPT_FILES["best"]).write_text(json.dumps({"results": full}))
    (sig / fm.CKPT_FILES["final"]).write_text(json.dumps({"results": short}))
    monkeypatch.setattr(fm, "SIG_DIR", str(sig))
    with _pytest.raises(AssertionError, match="IQL\\|final"):
        fm._significance()


def test_multiplicity_excludes_qr_from_the_family():
    """
    QR-D3QN was not run in this campaign. Counting it would report a family
    size that never existed and adjust in the wrong place.
    """
    from src.final_multiplicity import CORE, CORE_LOWER, EXPECTED
    assert "QR" not in CORE and "qr" not in CORE_LOWER
    assert EXPECTED == 6


def test_ope_skips_incomplete_runs():
    """
    The same rule as everywhere else. A checkpoint exists after the first
    epoch, so selecting on its presence would evaluate a network nobody
    finished training and report it as one of the fifteen.
    """
    import inspect
    import src.final_ope as fo
    src = inspect.getsource(fo.run)
    assert "if not is_complete(d):" in src
