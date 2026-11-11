"""Load archived safety-score evidence into SQLite and recompute metrics in SQL.

Python handles validated ingestion, hashes and JSON output. The SQL views perform
cohort joins, confusion counts, Brier scores, ECE and tie-aware AUROC. Only saved
scores/labels/membership are loaded; no raw prompts or model inference are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARTITIONS = {"fit", "calibration", "test", "test_without_train_text_overlap"}
METRIC_FIELDS = ("n", "positives", "negatives", "pos_rate", "ece", "brier",
                 "false_positives", "false_negatives", "fpr_at_cal_thr",
                 "fnr_at_cal_thr", "auroc")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_artifact(run_dir: Path, name: str, expected_hash: str) -> bytes:
    # Manifest paths must be plain local filenames, never parent/absolute paths.
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("artifact filename must be a local basename")
    data = (run_dir / name).read_bytes()
    if sha256(data) != expected_hash:
        raise ValueError(f"{name}: artifact SHA-256 mismatch")
    return data


def initialize(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript((ROOT / "sql/schema.sql").read_text())
    connection.executescript((ROOT / "sql/analysis.sql").read_text())
    connection.row_factory = sqlite3.Row


def ingest(connection: sqlite3.Connection, run_dir: Path) -> tuple[str, dict]:
    """Validate archived structure and ingest a single run in one transaction."""
    report_bytes = (run_dir / "report.json").read_bytes()
    report = json.loads(report_bytes)
    if report.get("schema_version") != 1:
        raise ValueError("only evidence schema_version 1 is supported")
    bins = report["configuration"]["ece_bins"]
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("ece_bins must be a positive integer")
    target_fpr = report["configuration"]["target_fpr"]
    if isinstance(target_fpr, bool) or not isinstance(target_fpr, (int, float)) or not math.isfinite(target_fpr) or not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must be finite and between zero and one")
    membership_bytes = checked_artifact(
        run_dir, "split_membership.json", report["split"]["membership_file_sha256"])
    membership = json.loads(membership_bytes)
    if set(membership) != PARTITIONS:
        raise ValueError("unexpected membership partitions")
    for partition, ids in membership.items():
        if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("membership IDs must be nonempty strings")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{partition}: duplicate membership ID")
        if len(ids) != report["split"]["counts"][partition]:
            raise ValueError(f"{partition}: membership count mismatch")
        digest = sha256(json.dumps(sorted(ids), separators=(",", ":")).encode())
        if digest != report["split"]["id_set_sha256"][partition]:
            raise ValueError(f"{partition}: membership ID-set hash mismatch")
    sets = {name: set(ids) for name, ids in membership.items()}
    if sets["fit"] & sets["calibration"] or (sets["fit"] | sets["calibration"]) & sets["test"]:
        raise ValueError("fit/calibration/test IDs must be disjoint")
    if not sets["test_without_train_text_overlap"] <= sets["test"]:
        raise ValueError("sensitivity membership must be a test subset")

    run_id = sha256(report_bytes)
    seen_items = {}
    with connection:
        connection.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, run_id, report["dataset"]["id"], report["dataset"]["revision"], bins, target_fpr))
        connection.execute("INSERT INTO artifacts VALUES (?, ?, ?)",
            (run_id, "report.json", run_id))
        connection.execute("INSERT INTO artifacts VALUES (?, ?, ?)",
            (run_id, "split_membership.json", sha256(membership_bytes)))
        connection.executemany("INSERT INTO memberships VALUES (?, ?, ?)",
            ((run_id, item_id, partition) for partition, ids in membership.items() for item_id in ids))
        # Match np.linspace's historical arithmetic without depending on NumPy.
        step = 1.0 / bins
        connection.executemany("INSERT INTO calibration_bins VALUES (?, ?, ?, ?)",
            ((run_id, i, i * step, (i + 1) * step if i + 1 < bins else 1.0)
             for i in range(bins)))
        if not report["judges"]:
            raise ValueError("at least one judge is required")
        for judge, manifest in report["judges"].items():
            variants = manifest["variants"]
            if not variants:
                raise ValueError("at least one score variant is required")
            for variant, metrics in variants.items():
                threshold = metrics["threshold"]
                if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
                    raise ValueError("frozen threshold must be finite")
                connection.execute("INSERT INTO variants VALUES (?, ?, ?, ?)",
                    (run_id, judge, variant, threshold))
            spec = manifest["predictions"]
            payload = checked_artifact(run_dir, spec["file"], spec["sha256"])
            connection.execute("INSERT OR IGNORE INTO artifacts VALUES (?, ?, ?)",
                (run_id, spec["file"], sha256(payload)))
            rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
            if len(rows) != spec["rows"]:
                raise ValueError("prediction row count mismatch")
            split_ids = {"calibration": [], "test": []}
            for row in rows:
                if set(row) != {"id", "split", "y", "jailbreak", "probabilities"}:
                    raise ValueError("unexpected prediction fields; raw text is not accepted")
                item_id, split = row["id"], row["split"]
                if split not in split_ids or item_id not in sets[split]:
                    raise ValueError("prediction ID/split disagrees with membership")
                if type(row["y"]) is not int or row["y"] not in (0, 1) or type(row["jailbreak"]) is not int or row["jailbreak"] not in (0, 1):
                    raise ValueError("labels and jailbreak flags must be binary integers")
                split_ids[split].append(item_id)
                item = (split, row["y"], row["jailbreak"])
                if item_id in seen_items and seen_items[item_id] != item:
                    raise ValueError("judges disagree on item labels or split")
                if item_id not in seen_items:
                    connection.execute("INSERT INTO items VALUES (?, ?, ?, ?, ?)",
                        (run_id, item_id, *item))
                    seen_items[item_id] = item
                if set(row["probabilities"]) != set(variants):
                    raise ValueError("score variants disagree with manifest")
                for variant, probability in row["probabilities"].items():
                    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
                        raise ValueError("scores must be finite probabilities")
                    connection.execute("INSERT INTO scores VALUES (?, ?, ?, ?, ?)",
                        (run_id, item_id, judge, variant, probability))
            for split, ids in split_ids.items():
                if ids != membership[split]:
                    raise ValueError(f"{split}: prediction order/IDs differ from membership")
    return run_id, report


def metric_rows(connection: sqlite3.Connection, run_id: str) -> list[dict]:
    return [dict(row) for row in connection.execute(
        "SELECT * FROM evaluation_metrics WHERE run_id = ? ORDER BY judge, variant, split, cohort",
        (run_id,))]


def verify_against_report(rows: list[dict], report: dict) -> int:
    """Compare SQL results to the independently produced Python report."""
    checked = 0
    for row in rows:
        source = report["judges"][row["judge"]]["variants"][row["variant"]]
        if row["split"] == "calibration":
            expected = {"negatives": source["calibration_negative_count"],
                        "false_positives": source["calibration_false_positives"],
                        "fpr_at_cal_thr": source["calibration_empirical_fpr"]}
        else:
            expected = {field: source[row["cohort"]][field] for field in METRIC_FIELDS}
        for field, value in expected.items():
            actual = row[field]
            matches = actual is None if value is None else (
                actual is not None and math.isclose(actual, value, rel_tol=1e-10, abs_tol=1e-12))
            if not matches:
                raise ValueError(f"SQL/report mismatch: {row['judge']}.{row['variant']}.{row['split']}.{row['cohort']}.{field}")
            checked += 1
    expected_rows = sum(6 for judge in report["judges"].values() for _ in judge["variants"])
    if len(rows) != expected_rows:
        raise ValueError("SQL metric cohort coverage is incomplete")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--database", type=Path, help="create a new SQLite database; existing files are never overwritten")
    args = parser.parse_args()
    created_path = None
    try:
        if args.database:
            # Exclusive creation avoids accidentally replacing another artifact.
            with args.database.open("xb"):
                pass
            created_path = args.database
        connection = sqlite3.connect(str(args.database) if args.database else ":memory:")
        try:
            initialize(connection)
            run_id, report = ingest(connection, args.run_dir)
            rows = metric_rows(connection, run_id)
            verified = verify_against_report(rows, report)
            overlap = dict(connection.execute(
                "SELECT * FROM sensitivity_membership_counts WHERE run_id = ?", (run_id,)).fetchone())
            print(json.dumps({"run_id": run_id, "database": str(args.database) if args.database else ":memory:",
                "sql_metric_fields_verified": verified, "metrics": rows,
                "sensitivity_membership": overlap,
                "not_recomputed": ["confidence intervals", "AUPRC", "normalized-text distinct overlap counts"],
                "raw_text_required": False}, indent=2, allow_nan=False))
        finally:
            connection.close()
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        if created_path is not None:
            created_path.unlink(missing_ok=True)
        parser.exit(2, f"SQL evidence audit: {exc}\n")


if __name__ == "__main__":
    main()
