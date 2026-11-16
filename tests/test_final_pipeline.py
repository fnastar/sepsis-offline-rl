"""
Invariants for the final Sepsis-3 MDP pipeline.

Every test here corresponds to a defect that would not crash. The pipeline runs
for hours and its output feeds training, so a wrong number is only visible much
later as a slightly-off FQE value — by which point the run is already paid for.
These are the checks that make such defects fail immediately instead.
"""
import numpy as np
import pandas as pd
import pytest

from src.final_features import _allocate_across_bins, _asof_state

ONSET = pd.Series({1: pd.Timestamp("2020-01-01 00:00")})


def _ev(rows, index=None):
    return pd.DataFrame(rows, index=index)


# ─────────────────────────────────────────────────────────────────────────────
# Infusion allocation
# ─────────────────────────────────────────────────────────────────────────────

def test_allocation_conserves_volume_across_bins():
    """A 10 h / 1000 mL infusion must still be 1000 mL after being split."""
    ev = _ev({"stay_id": [1],
              "starttime": [pd.Timestamp("2020-01-01 01:00")],
              "endtime": [pd.Timestamp("2020-01-01 11:00")],
              "amount": [1000.0], "rate": [100.0]})
    a = _allocate_across_bins(ev, ONSET)
    vol = a["amount"] * a["frac"]
    assert a["time_bin"].tolist() == [0, 1, 2]
    assert a["overlap_h"].tolist() == [3.0, 4.0, 3.0]
    assert vol.sum() == pytest.approx(1000.0)
    assert vol.tolist() == pytest.approx([300.0, 400.0, 300.0])


def test_allocation_survives_duplicate_index_labels():
    """
    The chunked loaders concatenate frames, so index labels repeat. Label-based
    expansion would silently return a cross product.
    """
    ev = _ev({"stay_id": [1, 1],
              "starttime": [pd.Timestamp("2020-01-01 01:00"),
                            pd.Timestamp("2020-01-01 05:00")],
              "endtime": [pd.Timestamp("2020-01-01 11:00"),
                          pd.Timestamp("2020-01-01 09:00")],
              "amount": [1000.0, 400.0], "rate": [100.0, 50.0]}, index=[0, 0])
    a = _allocate_across_bins(ev, ONSET)
    assert (a["amount"] * a["frac"]).sum() == pytest.approx(1400.0)


def test_zero_duration_bolus_lands_wholly_in_one_interval():
    ev = _ev({"stay_id": [1],
              "starttime": [pd.Timestamp("2020-01-01 02:00")],
              "endtime": [pd.Timestamp("2020-01-01 02:00")],
              "amount": [250.0], "rate": [np.nan]}, index=[7])
    a = _allocate_across_bins(ev, ONSET)
    assert a["time_bin"].tolist() == [0]
    assert (a["amount"] * a["frac"]).sum() == pytest.approx(250.0)


def test_event_ending_on_a_boundary_does_not_touch_the_next_interval():
    """An infusion stopping at exactly +4 h belongs to bin 0 alone."""
    ev = _ev({"stay_id": [1],
              "starttime": [pd.Timestamp("2020-01-01 00:00")],
              "endtime": [pd.Timestamp("2020-01-01 04:00")],
              "amount": [500.0], "rate": [125.0]})
    a = _allocate_across_bins(ev, ONSET)
    assert a["time_bin"].tolist() == [0]


def test_pre_onset_events_get_negative_bins():
    """The window opens 24 h before onset, so bins -6..-1 are real."""
    ev = _ev({"stay_id": [1],
              "starttime": [pd.Timestamp("2019-12-31 22:00")],
              "endtime": [pd.Timestamp("2020-01-01 02:00")],
              "amount": [400.0], "rate": [100.0]})
    a = _allocate_across_bins(ev, ONSET)
    assert a["time_bin"].tolist() == [-1, 0]
    assert (a["amount"] * a["frac"]).sum() == pytest.approx(400.0)


# ─────────────────────────────────────────────────────────────────────────────
# Past-only state
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def boundaries():
    return pd.DataFrame({
        "stay_id": [1, 1, 1], "time_bin": [0, 1, 2],
        "boundary": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 04:00",
                                    "2020-01-01 08:00"])})


@pytest.fixture
def events():
    return pd.DataFrame({
        "stay_id": [1, 1, 1, 1],
        "charttime": pd.to_datetime(["2019-12-31 23:00", "2020-01-01 04:00",
                                     "2020-01-01 05:30", "2020-01-01 09:00"]),
        "feature": ["mbp"] * 4,
        "valuenum": [70.0, 999.0, 55.0, 42.0]})


def test_state_excludes_values_timestamped_at_the_boundary(boundaries, events):
    """
    The action is taken at the boundary, so a measurement stamped at exactly
    that instant is not information the decision could have used. 999 is the
    value at 04:00; it must not appear in bin 1.
    """
    r = _asof_state(boundaries, events, "charttime", strictly_before=True)
    assert r["mbp"].tolist() == [70.0, 70.0, 55.0]


def test_state_never_sees_the_future(boundaries, events):
    r = _asof_state(boundaries, events, "charttime", strictly_before=True)
    assert 42.0 not in r["mbp"].values   # 09:00, after every boundary
    assert 999.0 not in r["mbp"].values  # 04:00, at a boundary


def test_sofa_path_allows_boundary_exact_matches(boundaries, events):
    """
    SOFA rows are keyed on the *end* of their scoring window, so a row ending
    exactly at the boundary is already available at it — unlike a raw
    observation stamped at that instant.
    """
    r = _asof_state(boundaries, events, "charttime", strictly_before=False)
    assert r["mbp"].tolist() == [70.0, 999.0, 55.0]


def test_carry_forward_holds_the_last_value(boundaries, events):
    """Bin 0 has only the 23:00 value before it and must carry it forward."""
    r = _asof_state(boundaries, events, "charttime", strictly_before=True)
    assert r.loc[r["time_bin"] == 0, "mbp"].iloc[0] == 70.0


def test_boundary_with_no_prior_observation_is_missing():
    b = pd.DataFrame({"stay_id": [1], "time_bin": [0],
                      "boundary": pd.to_datetime(["2020-01-01 00:00"])})
    e = pd.DataFrame({"stay_id": [1],
                      "charttime": pd.to_datetime(["2020-01-01 06:00"]),
                      "feature": ["mbp"], "valuenum": [88.0]})
    r = _asof_state(b, e, "charttime", strictly_before=True)
    assert pd.isna(r["mbp"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# Vasopressor intensity on a shared timeline
# ─────────────────────────────────────────────────────────────────────────────

def _intensity(rows, mode="intensity"):
    """
    Norepinephrine-equivalent dose for one interval, given (drug, start, end, rate).

    ``mode`` is passed explicitly rather than left to ``SEPSIS_VASO_AGG`` so
    each test states which estimand it is about. The reported specification is
    ``max``; ``intensity`` is kept for the sensitivity audit and is still
    tested, since the audit compares against it.
    """
    from src.final_features import _intensity_on_shared_timeline
    ev = pd.DataFrame([{"stay_id": 1, "drug": d,
                        "starttime": pd.Timestamp(s), "endtime": pd.Timestamp(e),
                        "amount": 0, "rate": 0, "ne_rate": r}
                       for d, s, e, r in rows])
    a = _allocate_across_bins(ev, ONSET)
    a = a[a["overlap_h"] > 0]
    return float(_intensity_on_shared_timeline(a, mode=mode)["rate"].iloc[0])


def test_concurrent_drugs_are_additive():
    """Norepinephrine equivalents add at a given instant."""
    assert _intensity([
        ("norepinephrine", "2020-01-01 00:00", "2020-01-01 04:00", 0.2),
        ("phenylephrine",  "2020-01-01 00:00", "2020-01-01 04:00", 0.2),
    ]) == pytest.approx(0.4)


def test_disjoint_drugs_do_not_sum():
    """
    One pressor stopping as another starts is not a doubled dose.

    Summing each drug's own duration-weighted mean would report 0.4 here, a
    peak the patient never reached. This affects 4.85% of treated intervals.
    """
    assert _intensity([
        ("norepinephrine", "2020-01-01 00:00", "2020-01-01 02:00", 0.2),
        ("phenylephrine",  "2020-01-01 02:00", "2020-01-01 04:00", 0.2),
    ]) == pytest.approx(0.2)


def test_partial_overlap_is_time_weighted():
    """Two hours at 0.4 then two at 0.2 averages to 0.3 over the covered time."""
    assert _intensity([
        ("norepinephrine", "2020-01-01 00:00", "2020-01-01 04:00", 0.2),
        ("phenylephrine",  "2020-01-01 00:00", "2020-01-01 02:00", 0.2),
    ]) == pytest.approx(0.3)


def test_intensity_is_conditional_on_running():
    """
    A drug running for one of four hours is coded at its actual rate.

    This is the estimand: the intensity of support while supported, not the
    dose averaged over an interval the patient spent mostly off it. Averaging
    over the full interval would code this 0.15 and place a top-level dose in
    the lowest non-zero bin.
    """
    assert _intensity([
        ("norepinephrine", "2020-01-01 02:00", "2020-01-01 04:00", 0.3),
    ]) == pytest.approx(0.3)


def test_same_drug_sequential_rows_are_time_weighted():
    """Titration recorded as consecutive rows averages over the interval."""
    assert _intensity([
        ("norepinephrine", "2020-01-01 00:00", "2020-01-01 02:00", 0.2),
        ("norepinephrine", "2020-01-01 02:00", "2020-01-01 04:00", 0.3),
    ]) == pytest.approx(0.25)


def test_same_drug_overlapping_rows_do_not_sum():
    """
    Overlapping orders for one drug are a rate update, not two infusions.

    96.7% of the 816 overlapping same-drug groups carry differing rates, and
    nobody runs two norepinephrine infusions at different rates at once.
    Summing them would report 0.35 for a patient never above 0.3.
    """
    assert _intensity([
        ("norepinephrine", "2020-01-01 00:00", "2020-01-01 03:00", 0.2),
        ("norepinephrine", "2020-01-01 02:00", "2020-01-01 04:00", 0.3),
    ]) == pytest.approx(0.25)


def test_duplicate_orders_do_not_multiply():
    """
    MIMIC records byte-identical duplicate rows; three of them are one infusion.

    This is the case that inflated one real cell 19-fold before the fix.
    """
    rows = [("norepinephrine", "2020-01-01 00:00", "2020-01-01 04:00", 0.5)] * 3
    assert _intensity(rows) == pytest.approx(0.5)


# --- max-in-window, the reported specification -----------------------------
#
# Same cases as above under the estimand the campaign actually uses: the
# highest instantaneous total rate reached anywhere in the interval.


def test_max_concurrent_drugs_are_additive():
    """Equivalents still add at a given instant under max."""
    assert _intensity([
        ("norepinephrine", "2020-01-01 00:00", "2020-01-01 04:00", 0.2),
        ("phenylephrine",  "2020-01-01 00:00", "2020-01-01 04:00", 0.2),
    ], mode="max") == pytest.approx(0.4)


def test_max_disjoint_drugs_do_not_sum():
    """A peak that never occurred is not the maximum."""
    assert _intensity([
        ("norepinephrine", "2020-01-01 00:00", "2020-01-01 02:00", 0.2),
        ("phenylephrine",  "2020-01-01 02:00", "2020-01-01 04:00", 0.2),
    ], mode="max") == pytest.approx(0.2)


def test_max_takes_the_peak_not_the_average():
    """Titration to 0.3 is coded 0.3, which is the point of the estimand."""
    assert _intensity([
        ("norepinephrine", "2020-01-01 00:00", "2020-01-01 02:00", 0.2),
        ("norepinephrine", "2020-01-01 02:00", "2020-01-01 04:00", 0.3),
    ], mode="max") == pytest.approx(0.3)


def test_max_same_drug_overlapping_rows_do_not_sum():
    """Overlapping orders for one drug are a rate update, so the peak is 0.3."""
    assert _intensity([
        ("norepinephrine", "2020-01-01 00:00", "2020-01-01 03:00", 0.2),
        ("norepinephrine", "2020-01-01 02:00", "2020-01-01 04:00", 0.3),
    ], mode="max") == pytest.approx(0.3)


def test_max_is_idempotent_under_duplicate_orders():
    """
    Duplicated rows cannot move the action. This is the property that made
    max the final choice: the 19-fold cell is bounded by construction.
    """
    rows = [("norepinephrine", "2020-01-01 00:00", "2020-01-01 04:00", 0.5)] * 3
    assert _intensity(rows, mode="max") == pytest.approx(0.5)


def test_max_brief_infusion_keeps_its_rate():
    """A ten-minute run at 0.3 is coded 0.3, not diluted over the interval."""
    assert _intensity([
        ("norepinephrine", "2020-01-01 03:50", "2020-01-01 04:00", 0.3),
    ], mode="max") == pytest.approx(0.3)


def test_max_default_is_the_module_default():
    """
    The default must be max, so no ambient environment or stale cache can
    silently produce the conditional mean in a campaign build.
    """
    from src.final_features import VASO_AGG
    import os
    if "SEPSIS_VASO_AGG" not in os.environ:
        assert VASO_AGG == "max"


# --- source-record ceiling --------------------------------------------------


def _ne(rows):
    """Norepinephrine-equivalent rate for raw (drug, rate, rateuom, weight) rows."""
    from src.final_features import _norepi_equivalent_rate
    ev = pd.DataFrame([{"drug": d, "rate": r, "rateuom": u, "patientweight": w}
                       for d, r, u, w in rows])
    return _norepi_equivalent_rate(ev).tolist()


def test_impossible_rate_is_clipped_to_the_ceiling():
    """
    A whole bag over one recorded minute is a documentation artefact.

    32 mg of norepinephrine for an 89 kg patient in one minute is stored as
    359 mcg/kg/min. Nobody administers that, and left alone it sets the scale
    of every ordinary dose in the state.
    """
    from src.final_features import NE_RATE_CEILING
    assert _ne([("norepinephrine", 359.55, "mcg/kg/min", 89.0)]) == \
        pytest.approx([NE_RATE_CEILING])


def test_ordinary_doses_are_untouched_by_the_ceiling():
    """Clipping must not move any dose a clinician actually orders."""
    assert _ne([("norepinephrine", 0.05, "mcg/kg/min", 80.0),
                ("norepinephrine", 0.30, "mcg/kg/min", 80.0),
                ("norepinephrine", 2.00, "mcg/kg/min", 80.0)]) == \
        pytest.approx([0.05, 0.30, 2.00])


def test_negative_rate_is_not_a_dose():
    """A negative amount is a reversal entry, not an infusion."""
    assert _ne([("phenylephrine", -5.724638, "mcg/kg/min", 82.8)]) == \
        pytest.approx([0.0])


def test_clipping_cannot_change_an_action_level():
    """
    Every clipped value stays in the top bin, so no action label moves.

    This is what makes the ceiling a data-quality fix rather than a change to
    the approved action space.
    """
    from src.final_features import NE_RATE_CEILING
    top_cutpoint = 0.2608928386859285   # train-split third quartile, v2 bins
    assert NE_RATE_CEILING > top_cutpoint


# --- source plausibility ----------------------------------------------------


def _plausible(rows):
    """rows are (feature, value); returns the values that survive filtering."""
    from src.final_features import apply_plausible_ranges
    df = pd.DataFrame([{"feature": f, "valuenum": v} for f, v in rows])
    return apply_plausible_ranges(df, "feature", "valuenum")["valuenum"].tolist()


def test_impossible_pressures_are_dropped():
    """
    A mean pressure of 1,234,570 is a wrong entry, not a high pressure.

    Left in, it sets the standardisation scale for every real blood pressure:
    the fitted SD was 2,998 against 15 for the inner 99.8% of the column.
    """
    assert _plausible([("mbp", 75.0), ("mbp", 1_234_570.0), ("mbp", 299.0),
                       ("mbp", 300.0)]) == [75.0, 299.0]


def test_physiological_values_survive():
    """The filter must not touch anything a patient can actually have."""
    rows = [("heart_rate", 85.0), ("sbp", 115.0), ("dbp", 60.0),
            ("mbp", 75.0), ("resp_rate", 19.0), ("temperature", 36.9),
            ("spo2", 97.0), ("glucose", 126.0), ("fio2", 40.0)]
    assert _plausible(rows) == [v for _, v in rows]


def test_missing_is_not_implausible():
    """An absent reading is absent, and must not be confused with a bad one."""
    import numpy as np
    out = _plausible([("mbp", np.nan), ("mbp", 75.0)])
    assert len(out) == 2 and out[1] == 75.0


def test_fio2_below_room_air_is_dropped():
    """
    Room air is 21%, so nothing below it is a fraction of inspired oxygen.

    The bound is the derived bg table's own: it publishes FiO2 only in
    [21, 100].
    """
    assert _plausible([("fio2", 0.0), ("fio2", 4.0), ("fio2", 21.0),
                       ("fio2", 100.0), ("fio2", 1000.0)]) == [21.0, 100.0]


def test_temperature_bounds_are_the_derived_layer_bounds():
    """15 to 43.1 is what the derived vitalsign table publishes; 10 to 50 admits it."""
    assert _plausible([("temperature", -17.78), ("temperature", 36.9),
                       ("temperature", 536.39)]) == [36.9]


def test_base_excess_beyond_survivable_is_dropped():
    """-413 is not a base excess. The derived layer publishes no bound here."""
    assert _plausible([("base_excess", -413.0), ("base_excess", -22.0),
                       ("base_excess", 0.0)]) == [-22.0, 0.0]


def test_gu_irrigant_is_subtracted_not_added():
    """
    Bladder irrigant instilled is not urine the patient made.

    Summed as output it put 48,900 mL in a four-hour window; the official
    concept negates it and the corrected maximum across the cohort is 9,800.
    """
    from src.final_features import GU_IRRIGANT_IN_ITEMID
    assert GU_IRRIGANT_IN_ITEMID == 227488
    from config.clinical import URINE_OUTPUT_ITEMIDS
    assert GU_IRRIGANT_IN_ITEMID in URINE_OUTPUT_ITEMIDS, (
        "the negation only fires if the item is actually collected")


def test_plausibility_filter_requires_a_unique_index():
    """
    The filter assigns by label, so a repeated label would drop every row
    sharing it. The chunked loaders concatenate with ignore_index, and this
    asserts that rather than trusting it, because a duplicate-label cross
    product has already cost this pipeline one rebuild.
    """
    import pandas as pd
    import pytest as _pytest
    from src.final_features import apply_plausible_ranges
    df = pd.DataFrame({"feature": ["mbp", "mbp"], "valuenum": [75.0, 1e6]},
                      index=[0, 0])
    with _pytest.raises(AssertionError):
        apply_plausible_ranges(df, "feature", "valuenum")
