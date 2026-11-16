-- Independent BigQuery audit for the final first-episode cohort.
-- Expected MIMIC-IV v3.1 result (2026-08-19):
-- 32899 official/adult, 32690 trajectory-eligible, 25442 first episodes,
-- and 6370 deaths by day 90.
WITH base AS (
  SELECT
    s.subject_id,
    s.stay_id,
    s.suspected_infection_time,
    i.intime,
    i.outtime,
    GREATEST(
      DATETIME_DIFF(i.intime, s.suspected_infection_time, SECOND) / 3600.0,
      -24
    ) AS episode_start_hours,
    LEAST(
      DATETIME_DIFF(i.outtime, s.suspected_infection_time, SECOND) / 3600.0,
      48
    ) AS episode_end_hours,
    p.anchor_age,
    p.dod
  FROM `physionet-data.mimiciv_3_1_derived.sepsis3` AS s
  INNER JOIN `physionet-data.mimiciv_3_1_icu.icustays` AS i
    USING (subject_id, stay_id)
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` AS p
    USING (subject_id)
  WHERE s.sepsis3 = TRUE
),
with_intervals AS (
  SELECT
    *,
    GREATEST(
      FLOOR(episode_end_hours / 4) - CEIL(episode_start_hours / 4),
      0
    ) AS n_full_4h_intervals
  FROM base
  WHERE anchor_age >= 18
),
eligible AS (
  SELECT *
  FROM with_intervals
  WHERE n_full_4h_intervals >= 2
),
first_episode AS (
  SELECT *
  FROM eligible
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY subject_id
    ORDER BY suspected_infection_time, stay_id
  ) = 1
)
SELECT
  (SELECT COUNT(*) FROM base) AS official,
  (SELECT COUNTIF(anchor_age >= 18) FROM base) AS adult,
  (SELECT COUNT(*) FROM eligible) AS trajectory_eligible,
  (SELECT COUNT(*) FROM first_episode) AS first_episode,
  (SELECT COUNTIF(
    dod IS NOT NULL
    AND DATE_DIFF(dod, DATE(intime), DAY) BETWEEN 0 AND 90
  ) FROM first_episode) AS deaths_90d;
