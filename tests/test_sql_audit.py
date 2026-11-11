"""Cross-implementation and adversarial data-contract tests for SQL metrics."""
import json
import math
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sql_eval_audit import ROOT, ingest, initialize, metric_rows, sha256, verify_against_report

RUN = ROOT / "results/reproduction-cpu-20261111"


def synthetic_database(scores, labels, *, bins=15, threshold=0.5, jailbreak=None):
    """A deliberately independent fixture, without the evidence importer."""
    connection = sqlite3.connect(":memory:")
    initialize(connection)
    connection.execute("INSERT INTO runs VALUES ('example', 'unused', 'synthetic', 'v1', ?, 0.05)", (bins,))
    step = 1.0 / bins
    connection.executemany("INSERT INTO calibration_bins VALUES ('example', ?, ?, ?)",
        ((i, i * step, (i + 1) * step if i + 1 < bins else 1.0) for i in range(bins)))
    connection.execute("INSERT INTO variants VALUES ('example', 'judge', 'raw', ?)", (threshold,))
    flags = jailbreak if jailbreak is not None else [0] * len(scores)
    for i, (score, label, flag) in enumerate(zip(scores, labels, flags)):
        connection.execute("INSERT INTO items VALUES ('example', ?, 'test', ?, ?)", (str(i), label, flag))
        connection.execute("INSERT INTO scores VALUES ('example', ?, 'judge', 'raw', ?)", (str(i), score))
    return connection


class SQLMetricTests(unittest.TestCase):
    def test_all_archived_variants_match_independent_python_report(self):
        with sqlite3.connect(":memory:") as connection:
            initialize(connection)
            run_id, report = ingest(connection, RUN)
            rows = metric_rows(connection, run_id)
            self.assertEqual(verify_against_report(rows, report), 174)
            self.assertEqual(len(rows), 18)
            platt = next(r for r in rows if r["variant"] == "platt" and r["split"] == "test" and r["cohort"] == "all")
            self.assertAlmostEqual(platt["auroc"], 0.9089149690872216, places=14)
            self.assertAlmostEqual(platt["ece"], 0.009935944367364941, places=14)
            jailbreak = next(r for r in rows if r["variant"] == "platt" and r["cohort"] == "jailbreak")
            self.assertEqual((jailbreak["n"], jailbreak["false_negatives"]), (91, 8))
            self.assertAlmostEqual(jailbreak["ece"], 0.32544079679081644, places=14)
            self.assertIsNone(jailbreak["auroc"])
            overlap = dict(connection.execute("SELECT * FROM sensitivity_membership_counts").fetchone())
            self.assertEqual((overlap["test_rows"], overlap["excluded_test_rows"], overlap["retained_test_rows"]), (5083, 250, 4833))

    def test_auc_gives_half_credit_to_ties_with_unequal_class_sizes(self):
        # Each positive wins against 0.1 and ties with 0.5: (1 + .5) / 3.
        connection = synthetic_database([0.1, 0.5, 0.9, 0.5, 0.5], [0, 0, 0, 1, 1])
        row = next(r for r in metric_rows(connection, "example") if r["cohort"] == "all")
        self.assertAlmostEqual(row["auroc"], 0.5)
        self.assertEqual((row["true_positives"], row["false_positives"], row["true_negatives"]), (2, 2, 1))
        connection.close()

    def test_auc_and_fpr_undefined_for_all_positive_subgroup(self):
        connection = synthetic_database([0.0, 1.0, 0.5], [1, 1, 1], jailbreak=[1, 1, 1])
        row = next(r for r in metric_rows(connection, "example") if r["cohort"] == "jailbreak")
        self.assertIsNone(row["auroc"])
        self.assertIsNone(row["fpr_at_cal_thr"])
        self.assertAlmostEqual(row["ece"], 0.5)
        self.assertEqual(row["false_negatives"], 1)
        connection.close()

    def test_probability_one_can_be_rejected_by_frozen_threshold_above_one(self):
        connection = synthetic_database([1.0, 1.0], [0, 1], threshold=math.nextafter(1.0, math.inf))
        row = next(r for r in metric_rows(connection, "example") if r["cohort"] == "all")
        self.assertEqual((row["false_positives"], row["false_negatives"]), (0, 1))
        self.assertAlmostEqual(row["auroc"], 0.5)
        connection.close()

    def test_inclusive_upper_edges_and_next_float_enter_distinct_bins(self):
        step = 1.0 / 15
        edge = 5 * step
        scores = [0.0, edge, math.nextafter(edge, math.inf), 1.0]
        connection = synthetic_database(scores, [0, 1, 0, 1])
        bins = {r["bin_index"]: r["n"] for r in connection.execute(
            "SELECT bin_index, n FROM bin_statistics WHERE cohort = 'all'")}
        self.assertEqual(bins, {0: 1, 4: 1, 5: 1, 14: 1})
        # The two adjacent scores have opposite labels and MUST NOT cancel.
        row = next(r for r in metric_rows(connection, "example") if r["cohort"] == "all")
        expected = ((1 - edge) + math.nextafter(edge, math.inf)) / 4
        self.assertAlmostEqual(row["ece"], expected)
        connection.close()


class SQLIngestionTests(unittest.TestCase):
    def copied_run(self, tmp):
        target = Path(tmp) / "run"
        shutil.copytree(RUN, target)
        return target

    def test_modified_prediction_artifact_fails_before_ingestion_commits(self):
        with tempfile.TemporaryDirectory() as tmp, sqlite3.connect(":memory:") as connection:
            target = self.copied_run(tmp)
            score_file = target / "tfidf_lr_predictions.jsonl"
            score_file.write_bytes(score_file.read_bytes() + b"\n")
            initialize(connection)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                ingest(connection, target)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 0)

    def test_duplicate_scores_rejected_even_with_updated_manifest_hash(self):
        with tempfile.TemporaryDirectory() as tmp, sqlite3.connect(":memory:") as connection:
            target = self.copied_run(tmp)
            score_file = target / "tfidf_lr_predictions.jsonl"
            lines = score_file.read_bytes().splitlines()
            payload = b"\n".join([lines[0], lines[0], *lines[2:]]) + b"\n"
            score_file.write_bytes(payload)
            report = json.loads((target / "report.json").read_text())
            report["judges"]["tfidf_lr"]["predictions"]["sha256"] = sha256(payload)
            (target / "report.json").write_text(json.dumps(report))
            initialize(connection)
            with self.assertRaises(sqlite3.IntegrityError):
                ingest(connection, target)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 0)

    def test_existing_database_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "evidence.sqlite"
            output.write_bytes(b"keep this file")
            result = subprocess.run([sys.executable, str(ROOT / "sql_eval_audit.py"), str(RUN), "--database", str(output)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(output.read_bytes(), b"keep this file")


if __name__ == "__main__":
    unittest.main()
