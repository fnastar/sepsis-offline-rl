"""Unit tests for the final first-eligible-episode cohort rule."""
from __future__ import annotations

import unittest

import pandas as pd

from src.final_cohort import select_final_cohort


class FinalCohortTests(unittest.TestCase):
    def setUp(self) -> None:
        self.official = pd.DataFrame(
            [
                # Earlier stay for patient 1 has <2 complete bins; keep 102.
                (1, 101, "2100-01-01 04:00", "2100-01-01 05:00", 3, True),
                (1, 102, "2100-02-01 04:00", "2100-02-01 05:00", 4, True),
                # Earlier stay for patient 2 has <2 bins before discharge.
                (2, 201, "2100-03-03 03:00", "2100-03-03 03:30", 5, True),
                (2, 202, "2100-04-01 02:00", "2100-04-01 03:00", 4, True),
                # Both patient-3 stays qualify; keep the earlier onset.
                (3, 301, "2100-05-01 01:00", "2100-05-01 02:00", 6, True),
                (3, 302, "2100-06-01 01:00", "2100-06-01 02:00", 6, True),
                # A <24h ICU stay is valid when it has >=2 complete bins.
                (4, 401, "2100-07-01 00:00", "2100-07-01 01:00", 3, True),
                # Late-onset sepsis in a >30d stay is also valid.
                (5, 501, "2100-09-01 00:00", "2100-09-01 01:00", 5, True),
            ],
            columns=[
                "subject_id", "stay_id", "suspected_infection_time",
                "sofa_time", "sofa_score", "sepsis3",
            ],
        )

        self.icustays = pd.DataFrame(
            [
                (1, 1001, 101, "2100-01-01 00:00", "2100-01-01 06:00"),
                (1, 1002, 102, "2100-02-01 00:00", "2100-02-03 00:00"),
                (2, 2001, 201, "2100-03-03 00:00", "2100-03-03 04:00"),
                (2, 2002, 202, "2100-04-01 00:00", "2100-04-03 00:00"),
                (3, 3001, 301, "2100-05-01 00:00", "2100-05-03 00:00"),
                (3, 3002, 302, "2100-06-01 00:00", "2100-06-03 00:00"),
                (4, 4001, 401, "2100-07-01 00:00", "2100-07-01 12:00"),
                (5, 5001, 501, "2100-08-01 00:00", "2100-09-15 00:00"),
            ],
            columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        )
        self.icustays["first_careunit"] = "MICU"
        self.icustays["last_careunit"] = "MICU"

        self.patients = pd.DataFrame(
            [
                # Day 90 is included in 90-day mortality.
                (1, "F", 70, "2100-05-02"),
                # Patient 2 dies on day 91 and is labelled a survivor at day 90.
                (2, "M", 60, "2100-07-01"),
                (3, "F", 50, None),
                (4, "M", 40, None),
                (5, "F", 65, None),
            ],
            columns=["subject_id", "gender", "anchor_age", "dod"],
        )
        self.admissions = pd.DataFrame(
            [(hadm_id, "2099-12-01", "2101-01-01", None, 0, "UNKNOWN")
             for hadm_id in self.icustays["hadm_id"]],
            columns=[
                "hadm_id", "admittime", "dischtime", "deathtime",
                "hospital_expire_flag", "race",
            ],
        )

    def test_first_episode_is_selected_after_eligibility(self) -> None:
        final, audit = select_final_cohort(
            self.official, self.icustays, self.patients, self.admissions
        )

        self.assertEqual(final["stay_id"].tolist(), [102, 202, 301, 401, 501])
        self.assertEqual(final["subject_id"].nunique(), len(final))
        self.assertFalse(final["subject_id"].duplicated().any())
        self.assertEqual(audit["trajectory_eligible_stays"], 6)
        self.assertEqual(audit["excluded_repeat_episode_stays"], 1)
        self.assertEqual(audit["final_first_episode_stays"], 5)

    def test_ninety_day_mortality_is_day_inclusive(self) -> None:
        final, _ = select_final_cohort(
            self.official, self.icustays, self.patients, self.admissions
        )
        labels = final.set_index("subject_id")["icu_mortality"].to_dict()

        self.assertEqual(labels, {1: 1, 2: 0, 3: 0, 4: 0, 5: 0})


if __name__ == "__main__":
    unittest.main()
