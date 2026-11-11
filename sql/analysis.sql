-- Labels, scores and frozen thresholds are joined by explicit keys.
CREATE VIEW scored_items AS
SELECT i.run_id, s.judge, s.variant, i.item_id, i.split, i.y,
       i.jailbreak, s.probability, v.frozen_threshold,
       CASE WHEN m.item_id IS NULL THEN 0 ELSE 1 END AS in_sensitivity_slice
FROM items AS i
JOIN scores AS s USING (run_id, item_id)
JOIN variants AS v USING (run_id, judge, variant)
LEFT JOIN memberships AS m
  ON m.run_id = i.run_id AND m.item_id = i.item_id
 AND m.partition_name = 'test_without_train_text_overlap';

CREATE VIEW cohort_scores AS
SELECT *, 'all' AS cohort FROM scored_items
UNION ALL SELECT *, 'ordinary' FROM scored_items
  WHERE split = 'test' AND jailbreak = 0
UNION ALL SELECT *, 'jailbreak' FROM scored_items
  WHERE split = 'test' AND jailbreak = 1
UNION ALL SELECT *, 'test_without_train_text_overlap' FROM scored_items
  WHERE split = 'test' AND in_sensitivity_slice = 1
UNION ALL SELECT *, 'jailbreak_without_train_text_overlap' FROM scored_items
  WHERE split = 'test' AND in_sensitivity_slice = 1 AND jailbreak = 1;

CREATE VIEW confusion_counts AS
SELECT run_id, judge, variant, split, cohort, COUNT(*) AS n,
       SUM(y) AS positives, SUM(1 - y) AS negatives,
       SUM(CASE WHEN probability >= frozen_threshold AND y = 1 THEN 1 ELSE 0 END)
         AS true_positives,
       SUM(CASE WHEN probability < frozen_threshold AND y = 0 THEN 1 ELSE 0 END)
         AS true_negatives,
       SUM(CASE WHEN probability >= frozen_threshold AND y = 0 THEN 1 ELSE 0 END)
         AS false_positives,
       SUM(CASE WHEN probability < frozen_threshold AND y = 1 THEN 1 ELSE 0 END)
         AS false_negatives,
       AVG((probability - y) * (probability - y)) AS brier,
       MIN(frozen_threshold) AS frozen_threshold
FROM cohort_scores
GROUP BY run_id, judge, variant, split, cohort;

-- Empty bins contribute zero; only occupied bins need rows.
CREATE VIEW bin_statistics AS
SELECT c.run_id, c.judge, c.variant, c.split, c.cohort,
       b.bin_index, b.lower_edge, b.upper_edge, COUNT(*) AS n,
       SUM(c.probability) AS probability_sum, SUM(c.y) AS label_sum,
       AVG(c.probability) AS mean_probability, AVG(c.y) AS positive_rate
FROM cohort_scores AS c
JOIN calibration_bins AS b ON c.run_id = b.run_id
 AND c.probability <= b.upper_edge
 AND (b.bin_index = 0 OR c.probability > b.lower_edge)
GROUP BY c.run_id, c.judge, c.variant, c.split, c.cohort, b.bin_index;

CREATE VIEW calibration_error AS
SELECT run_id, judge, variant, split, cohort,
       SUM(ABS(probability_sum - label_sum)) / SUM(n) AS ece
FROM bin_statistics
GROUP BY run_id, judge, variant, split, cohort;

-- Group exact-score ties, count negatives strictly below each tied block,
-- then give each positive half credit for tied negatives (Mann-Whitney AUROC).
-- No Python metric function or SQLite user-defined metric is used.
CREATE VIEW ranking_quality AS
WITH tied_scores AS (
  SELECT run_id, judge, variant, split, cohort, probability,
         SUM(y) AS positives_at_score, SUM(1 - y) AS negatives_at_score
  FROM cohort_scores
  GROUP BY run_id, judge, variant, split, cohort, probability
), preceding_negatives AS (
  SELECT *, COALESCE(SUM(negatives_at_score) OVER (
    PARTITION BY run_id, judge, variant, split, cohort ORDER BY probability
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS negatives_below
  FROM tied_scores
)
SELECT run_id, judge, variant, split, cohort,
       SUM(positives_at_score * (negatives_below + 0.5 * negatives_at_score))
       / NULLIF(1.0 * SUM(positives_at_score) * SUM(negatives_at_score), 0) AS auroc
FROM preceding_negatives
GROUP BY run_id, judge, variant, split, cohort;

CREATE VIEW evaluation_metrics AS
SELECT c.*, 1.0 * c.positives / c.n AS pos_rate, e.ece, r.auroc,
       1.0 * c.false_positives / NULLIF(c.negatives, 0) AS fpr_at_cal_thr,
       1.0 * c.false_negatives / NULLIF(c.positives, 0) AS fnr_at_cal_thr
FROM confusion_counts AS c
JOIN calibration_error AS e USING (run_id, judge, variant, split, cohort)
JOIN ranking_quality AS r USING (run_id, judge, variant, split, cohort);

-- Reconstruct excluded ROWS from published membership; this cannot recover
-- the distinct normalized-text-hash overlap counts absent from the archive.
CREATE VIEW sensitivity_membership_counts AS
SELECT m.run_id, COUNT(*) AS test_rows,
       SUM(CASE WHEN clean.item_id IS NULL THEN 1 ELSE 0 END) AS excluded_test_rows,
       SUM(CASE WHEN clean.item_id IS NOT NULL THEN 1 ELSE 0 END) AS retained_test_rows
FROM memberships AS m
LEFT JOIN memberships AS clean
  ON m.run_id = clean.run_id AND m.item_id = clean.item_id
 AND clean.partition_name = 'test_without_train_text_overlap'
WHERE m.partition_name = 'test'
GROUP BY m.run_id;
