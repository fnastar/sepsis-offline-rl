-- Official MIMIC-IV v3.1 Sepsis-3 source used by the final cohort pipeline.
-- Run read-only in BigQuery and save the complete result as CSV.
SELECT
  subject_id,
  stay_id,
  suspected_infection_time,
  sofa_time,
  sofa_score,
  sepsis3
FROM `physionet-data.mimiciv_3_1_derived.sepsis3`
WHERE sepsis3 = TRUE
ORDER BY subject_id, suspected_infection_time, stay_id;
