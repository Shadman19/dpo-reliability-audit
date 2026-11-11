PRAGMA foreign_keys = ON;

CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    report_sha256 TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    dataset_revision TEXT NOT NULL,
    ece_bins INTEGER NOT NULL CHECK (ece_bins > 0),
    target_fpr REAL NOT NULL CHECK (target_fpr BETWEEN 0 AND 1)
);

CREATE TABLE artifacts (
    run_id TEXT NOT NULL REFERENCES runs,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    PRIMARY KEY (run_id, filename)
);

-- A sensitivity slice is a membership relation, not a new independent split.
CREATE TABLE memberships (
    run_id TEXT NOT NULL REFERENCES runs,
    item_id TEXT NOT NULL,
    partition_name TEXT NOT NULL CHECK (partition_name IN
      ('fit', 'calibration', 'test', 'test_without_train_text_overlap')),
    PRIMARY KEY (run_id, partition_name, item_id)
);

CREATE TABLE items (
    run_id TEXT NOT NULL REFERENCES runs,
    item_id TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('calibration', 'test')),
    y INTEGER NOT NULL CHECK (y IN (0, 1)),
    jailbreak INTEGER NOT NULL CHECK (jailbreak IN (0, 1)),
    PRIMARY KEY (run_id, item_id)
);

CREATE TABLE variants (
    run_id TEXT NOT NULL REFERENCES runs,
    judge TEXT NOT NULL,
    variant TEXT NOT NULL,
    frozen_threshold REAL NOT NULL,
    PRIMARY KEY (run_id, judge, variant)
);

CREATE TABLE scores (
    run_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    judge TEXT NOT NULL,
    variant TEXT NOT NULL,
    probability REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
    PRIMARY KEY (run_id, judge, variant, item_id),
    FOREIGN KEY (run_id, item_id) REFERENCES items,
    FOREIGN KEY (run_id, judge, variant) REFERENCES variants
);

-- Explicit edges preserve the study's [0, 1/B], (1/B, 2/B], ... convention.
CREATE TABLE calibration_bins (
    run_id TEXT NOT NULL REFERENCES runs,
    bin_index INTEGER NOT NULL,
    lower_edge REAL NOT NULL,
    upper_edge REAL NOT NULL,
    PRIMARY KEY (run_id, bin_index),
    CHECK (lower_edge >= 0 AND upper_edge <= 1 AND lower_edge < upper_edge)
);

CREATE INDEX membership_item ON memberships (run_id, item_id, partition_name);
CREATE INDEX score_order ON scores (run_id, judge, variant, probability);
