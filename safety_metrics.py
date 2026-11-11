"""Binary safety-score metrics, including one-class subgroups and tied scores.

The threshold controls the observed calibration-negative exceedance rate only.
Neither it nor the confidence intervals constitute a population safety guarantee.
Bootstrap intervals assume independently sampled evaluation rows; related prompts
or repeated users require a grouped resampling design instead.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def _scores_labels(probabilities, labels):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels)
    if p.ndim != 1 or y.ndim != 1 or p.shape != y.shape:
        raise ValueError("probabilities and labels must be equally sized 1D arrays")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and between zero and one")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary zero or one")
    return p, y.astype(np.int8)


def _check_bins(bins):
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer")


def ece(probabilities, labels, bins=15):
    """Fixed-width expected calibration error of toxicity probabilities.

    Bins follow the historical study's convention: [0, 1/bins], then
    (1/bins, 2/bins], ..., ((bins-1)/bins, 1]. Single-class input is valid.
    Empty input is undefined and rejected.
    """
    p, y = _scores_labels(probabilities, labels)
    _check_bins(bins)
    if not len(p):
        raise ValueError("ECE requires at least one evaluation row")
    edges = np.linspace(0.0, 1.0, bins + 1)
    memberships = np.searchsorted(edges[1:], p, side="left")
    counts = np.bincount(memberships, minlength=bins)
    score_totals = np.bincount(memberships, weights=p, minlength=bins)
    label_totals = np.bincount(memberships, weights=y, minlength=bins)
    # sum_b n_b/n * |mean(p_b)-mean(y_b)| = sum_b |sum(p_b-y_b)|/n.
    return float(np.abs(score_totals[counts > 0] - label_totals[counts > 0]).sum() / len(p))


def bootstrap_ece_ci(probabilities, labels, bootstraps=500, seed=0, bins=15):
    """Percentile 95% ECE interval with every resampled row-set retained.

    This is a bootstrap uncertainty estimate, not an exact finite-sample interval.
    One-class and constant-score resamples remain valid ECE observations.
    """
    p, y = _scores_labels(probabilities, labels)
    _check_bins(bins)
    if not len(p):
        raise ValueError("bootstrap ECE requires at least one evaluation row")
    if isinstance(bootstraps, bool) or not isinstance(bootstraps, (int, np.integer)) or bootstraps < 1:
        raise ValueError("bootstraps must be a positive integer")
    rng = np.random.default_rng(seed)
    values = np.empty(bootstraps, dtype=float)
    for index in range(bootstraps):
        rows = rng.integers(0, len(p), len(p))
        values[index] = ece(p[rows], y[rows], bins=bins)
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def choose_fpr_threshold(negative_scores, target_fpr=0.05):
    """Choose p>=threshold so observed calibration FPR is <= target_fpr.

    If k=floor(target_fpr*n), choose the next representable float above the
    (k+1)th highest calibration-negative score. Ties at that boundary are all
    excluded. A threshold just above 1 is permitted when needed to reject tied
    probability-one negatives. This is empirical calibration-set control only;
    distribution shift and sampling variation can increase evaluation FPR.
    """
    p = np.asarray(negative_scores, dtype=float)
    if p.ndim != 1 or not len(p):
        raise ValueError("at least one 1D calibration-negative score is required")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("negative scores must be finite probabilities")
    if isinstance(target_fpr, bool) or not isinstance(target_fpr, (int, float, np.integer, np.floating)):
        raise ValueError("target_fpr must be a number between zero and one")
    if not math.isfinite(float(target_fpr)) or not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must be finite and between zero and one")
    allowed = math.floor(float(target_fpr) * len(p))
    if allowed == len(p):
        return 0.0
    boundary = np.sort(p)[::-1][allowed]
    return float(np.nextafter(boundary, np.inf))


def wilson_interval(successes, trials, confidence=0.95):
    """Wilson score interval for a binomial rate, or None with no denominator."""
    for value in (successes, trials):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("successes and trials must be integers")
    if not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float, np.integer, np.floating)) or not math.isfinite(float(confidence)) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if trials == 0:
        return None
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, float(center - half_width)), min(1.0, float(center + half_width))]


def binary_metrics(probabilities, labels, threshold, bootstraps=500, seed=0, bins=15):
    """Counts, calibration and threshold rates with explicit denominators."""
    p, y = _scores_labels(probabilities, labels)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.integer, np.floating)) or not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    predictions = p >= threshold
    false_positives = int(np.sum(predictions & (y == 0)))
    false_negatives = int(np.sum(~predictions & (y == 1)))
    return {
        "n": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "pos_rate": float(y.mean()) if len(y) else None,
        "ece": ece(p, y, bins=bins) if len(y) else None,
        "ece_ci95": bootstrap_ece_ci(p, y, bootstraps=bootstraps, seed=seed, bins=bins) if len(y) else None,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "fpr_at_cal_thr": false_positives / negatives if negatives else None,
        "fnr_at_cal_thr": false_negatives / positives if positives else None,
        "fpr_ci95": wilson_interval(false_positives, negatives),
        "fnr_ci95": wilson_interval(false_negatives, positives),
    }
