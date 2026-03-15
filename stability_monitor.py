"""
Stability Monitor — Customer Segmentation Pipeline
Implements the full stability tracking framework from the documentation:
  1. Adjusted Rand Index (ARI) for label consistency
  2. Centroid shift (Euclidean distance) for KMeans drift
  3. Silhouette score tracking over time
  4. Cluster profile drift (mean RFM per cluster)
  5. Drift alert flags with configurable thresholds
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import os
import json

from sklearn.metrics import adjusted_rand_score, silhouette_score

logger = logging.getLogger(__name__)

# ─── Configurable thresholds ────────────────────────────────────────────────
THRESHOLDS = {
    "ari_warning": 0.80,        # ARI below this triggers a WARNING
    "ari_critical": 0.60,       # ARI below this triggers a CRITICAL alert
    "centroid_shift_warning": 0.5,   # Normalised Euclidean distance
    "centroid_shift_critical": 1.0,
    "silhouette_drop_warning": 0.05, # Absolute drop from previous run
    "silhouette_drop_critical": 0.10,
    "profile_drift_warning": 0.15,   # Relative change in cluster mean RFM
    "profile_drift_critical": 0.30,
}


# ─── 1. Adjusted Rand Index ──────────────────────────────────────────────────

def compute_ari(labels_previous: np.ndarray, labels_new: np.ndarray) -> float:
    """
    Compute ARI between two sets of cluster labels.
    ARI = 1 → perfect stability.
    ARI = 0 → random, no similarity.
    ARI < 0 → active anti-correlation (significant instability).
    """
    ari = adjusted_rand_score(labels_previous, labels_new)
    logger.info(f"Adjusted Rand Index: {ari:.4f}")
    return round(float(ari), 4)


def interpret_ari(ari: float) -> dict:
    """Map ARI to status and recommendation."""
    if ari >= THRESHOLDS["ari_warning"]:
        status = "STABLE"
        recommendation = "Cluster structure is consistent. No action needed."
    elif ari >= THRESHOLDS["ari_critical"]:
        status = "WARNING"
        recommendation = (
            "Moderate cluster drift detected. Review RFM distribution changes "
            "and consider re-evaluating optimal k."
        )
    else:
        status = "CRITICAL"
        recommendation = (
            "Significant cluster instability. Re-run elbow analysis, recalibrate "
            "the number of clusters, and investigate data distribution shifts."
        )
    return {"ari": ari, "status": status, "recommendation": recommendation}


# ─── 2. Centroid Shift ───────────────────────────────────────────────────────

def compute_centroid_shift(old_centroids: np.ndarray,
                           new_centroids: np.ndarray) -> dict:
    """
    Compute per-centroid and overall Euclidean shift between two KMeans runs.
    Centroids must be matched by cluster index.

    Returns dict with per-cluster distances and overall mean shift.
    """
    if old_centroids.shape != new_centroids.shape:
        raise ValueError(
            f"Centroid shape mismatch: {old_centroids.shape} vs {new_centroids.shape}"
        )
    per_cluster = np.linalg.norm(new_centroids - old_centroids, axis=1)
    overall = float(np.mean(per_cluster))
    result = {
        "centroid_shift_mean": round(overall, 4),
        "centroid_shift_per_cluster": {
            f"cluster_{i}": round(float(d), 4) for i, d in enumerate(per_cluster)
        },
        "centroid_shift_max": round(float(np.max(per_cluster)), 4),
    }
    logger.info(f"Centroid shift — mean: {overall:.4f}, max: {np.max(per_cluster):.4f}")
    return result


def interpret_centroid_shift(shift_mean: float) -> dict:
    """Map mean centroid shift to status."""
    if shift_mean < THRESHOLDS["centroid_shift_warning"]:
        status = "STABLE"
        recommendation = "Centroids are stable. No recalibration needed."
    elif shift_mean < THRESHOLDS["centroid_shift_critical"]:
        status = "WARNING"
        recommendation = "Moderate centroid drift. Monitor next period before re-training."
    else:
        status = "CRITICAL"
        recommendation = "High centroid drift. Trigger re-training with updated data."
    return {"centroid_shift_mean": shift_mean, "status": status, "recommendation": recommendation}


# ─── 3. Silhouette Score Tracking ────────────────────────────────────────────

def evaluate_silhouette(X_scaled: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score for the current clustering."""
    score = silhouette_score(X_scaled, labels)
    logger.info(f"Silhouette score: {score:.4f}")
    return round(float(score), 4)


def compare_silhouette(previous_score: float, current_score: float) -> dict:
    """Compare silhouette score to previous period."""
    drop = previous_score - current_score  # positive = degraded
    if drop <= THRESHOLDS["silhouette_drop_warning"]:
        status = "STABLE"
        recommendation = "Cluster quality maintained."
    elif drop <= THRESHOLDS["silhouette_drop_critical"]:
        status = "WARNING"
        recommendation = "Cluster quality slightly degraded. Consider re-evaluating k."
    else:
        status = "CRITICAL"
        recommendation = (
            "Significant cluster quality drop. "
            "Re-run elbow analysis and recalibrate model."
        )
    return {
        "silhouette_previous": previous_score,
        "silhouette_current": current_score,
        "silhouette_drop": round(drop, 4),
        "status": status,
        "recommendation": recommendation,
    }


# ─── 4. Cluster Profile Drift ────────────────────────────────────────────────

def compute_profile_drift(profile_old: pd.DataFrame,
                          profile_new: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative change in cluster mean RFM between two periods.

    Returns DataFrame with columns: Recency_drift, Frequency_drift, Monetary_drift
    for each cluster (as % relative change).
    """
    # Align cluster order
    common_clusters = profile_old.index.intersection(profile_new.index)
    old = profile_old.loc[common_clusters]
    new = profile_new.loc[common_clusters]

    drift = ((new - old) / (old.abs() + 1e-9)).round(4)
    drift.columns = [f"{c}_pct_change" for c in drift.columns]
    logger.info(f"Profile drift computed across {len(common_clusters)} clusters.")
    return drift


def interpret_profile_drift(drift_df: pd.DataFrame) -> dict:
    """Flag clusters where any metric drifts beyond thresholds."""
    flags = {}
    for cluster in drift_df.index:
        max_drift = drift_df.loc[cluster].abs().max()
        if max_drift < THRESHOLDS["profile_drift_warning"]:
            status = "STABLE"
        elif max_drift < THRESHOLDS["profile_drift_critical"]:
            status = "WARNING"
        else:
            status = "CRITICAL"
        flags[str(cluster)] = {"max_drift": round(float(max_drift), 4), "status": status}
    return flags


# ─── 5. Aggregate Stability Report ───────────────────────────────────────────

def build_stability_report(
    ari: float,
    centroid_shift: dict,
    silhouette_comparison: dict,
    profile_flags: dict,
    run_period: str,
) -> dict:
    """
    Assemble all stability metrics into a single report dict
    suitable for MLflow logging.
    """
    # Determine overall status (worst of all signals)
    statuses = [
        interpret_ari(ari)["status"],
        interpret_centroid_shift(centroid_shift["centroid_shift_mean"])["status"],
        silhouette_comparison["status"],
        *[v["status"] for v in profile_flags.values()],
    ]
    severity_rank = {"STABLE": 0, "WARNING": 1, "CRITICAL": 2}
    overall = max(statuses, key=lambda s: severity_rank[s])

    report = {
        "run_period": run_period,
        "overall_status": overall,
        "ari": ari,
        "ari_status": interpret_ari(ari)["status"],
        "centroid_shift_mean": centroid_shift["centroid_shift_mean"],
        "centroid_shift_max": centroid_shift["centroid_shift_max"],
        "centroid_shift_status": interpret_centroid_shift(
            centroid_shift["centroid_shift_mean"]
        )["status"],
        "silhouette_previous": silhouette_comparison["silhouette_previous"],
        "silhouette_current": silhouette_comparison["silhouette_current"],
        "silhouette_drop": silhouette_comparison["silhouette_drop"],
        "silhouette_status": silhouette_comparison["status"],
        "cluster_profile_flags": profile_flags,
    }
    logger.info(
        f"Stability report — overall: {overall}, ARI: {ari:.4f}, "
        f"centroid shift: {centroid_shift['centroid_shift_mean']:.4f}"
    )
    return report


# ─── 6. Stability Plot ───────────────────────────────────────────────────────

def plot_stability_history(history: list, output_dir: str) -> str:
    """
    Plot ARI and Silhouette trends over runs.

    Args:
        history: List of dicts with keys 'run_period', 'ari', 'silhouette_current'.
    """
    os.makedirs(output_dir, exist_ok=True)
    if len(history) < 2:
        logger.info("Not enough history to plot stability trends.")
        return None

    periods = [h["run_period"] for h in history]
    aris = [h["ari"] for h in history]
    silhouettes = [h.get("silhouette_current", None) for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ARI panel
    axes[0].plot(periods, aris, marker="o", color="#2E75B6", linewidth=2)
    axes[0].axhline(THRESHOLDS["ari_warning"], color="orange", linestyle="--",
                    label=f"Warning ({THRESHOLDS['ari_warning']})")
    axes[0].axhline(THRESHOLDS["ari_critical"], color="red", linestyle="--",
                    label=f"Critical ({THRESHOLDS['ari_critical']})")
    axes[0].set_ylabel("Adjusted Rand Index")
    axes[0].set_title("Cluster Stability Over Time")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Silhouette panel
    valid_sil = [(p, s) for p, s in zip(periods, silhouettes) if s is not None]
    if valid_sil:
        p_sil, s_sil = zip(*valid_sil)
        axes[1].plot(p_sil, s_sil, marker="s", color="#70AD47", linewidth=2)
        axes[1].axhline(0.5, color="orange", linestyle="--", label="Good threshold (0.5)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_xlabel("Run Period")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, "stability_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Stability history plot saved: {path}")
    return path


def save_stability_report(report: dict, output_dir: str, run_period: str) -> str:
    """Persist stability report as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    fname = f"stability_report_{run_period.replace(' ', '_')}.json"
    path = os.path.join(output_dir, fname)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Stability report saved: {path}")
    return path
