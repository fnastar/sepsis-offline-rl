-- Official MIMIC-IV v3.1 hourly SOFA for the final Sepsis-3 cohort.
--
-- Replaces the defective local SOFA (respiratory structurally capped at 2
-- because on_ventilator was never populated; renal creatinine-only because
-- urine_output_24h never reached sofa_renal). The derived table scores all
-- six components with ventilation-aware respiration (pao2fio2ratio_vent)
-- and rolling 24 h urine output (uo_24hr).
--
-- Rows are restricted to a generous margin around each stay's earliest
-- suspected-infection onset, so the local pipeline can select, for every
-- 4-hour decision boundary, the latest row with endtime <= boundary.
-- Margin is -25 h to +49 h against an episode window of -24 h to +48 h.
WITH onset AS (
  SELECT stay_id, MIN(suspected_infection_time) AS onset_time
  FROM `physionet-data.mimiciv_3_1_derived.sepsis3`
  WHERE sepsis3
  GROUP BY stay_id
)
SELECT
  s.stay_id,
  s.endtime,
  s.sofa_24hours
FROM `physionet-data.mimiciv_3_1_derived.sofa` s
JOIN onset o USING (stay_id)
WHERE s.endtime BETWEEN TIMESTAMP_SUB(o.onset_time, INTERVAL 25 HOUR)
                    AND TIMESTAMP_ADD(o.onset_time, INTERVAL 49 HOUR)
  AND s.sofa_24hours IS NOT NULL
ORDER BY s.stay_id, s.endtime;
