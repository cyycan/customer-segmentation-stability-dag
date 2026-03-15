"""
MLflow Logger — Customer Segmentation Pipeline
Centralises all MLflow logging: params, metrics, artifacts, model registration.
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "Customer_Segmentation_RFM"
STABILITY_EXPERIMENT_NAME = "Customer_Segmentation_Stability"
MODEL_NAME = "CustomerSegmentationKMeans"


def setup_mlflow(tracking_uri: str = "http://127.0.0.1:5001"):
    """Configure MLflow tracking server."""
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")


def get_or_create_experiment(name: str) -> str:
    """Return existing experiment ID or create it."""
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        logger.info(f"Created MLflow experiment: '{name}' (id={exp_id})")
    else:
        exp_id = exp.experiment_id
        logger.info(f"Using existing experiment: '{name}' (id={exp_id})")
    return exp_id


def log_training_run(
    run_period: str,
    data_diagnostics: dict,
    rfm_diagnostics: dict,
    n_clusters: int,
    kmeans_model,
    kmeans_sil: float,
    gmm_sil: float,
    cluster_profile: pd.DataFrame,
    artifact_paths: list,
    tracking_uri: str = "http://127.0.0.1:5001",
):
    """
    Log a full training run to MLflow.

    Returns: run_id (str)
    """
    setup_mlflow(tracking_uri)
    exp_id = get_or_create_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(experiment_id=exp_id, run_name=f"training_{run_period}") as run:
        # ── Tags ──────────────────────────────────────────────────────────────
        mlflow.set_tags({
            "run_period": run_period,
            "pipeline_stage": "training",
            "model_type": "KMeans+GMM",
        })

        # ── Data diagnostics ──────────────────────────────────────────────────
        mlflow.log_params({f"data_{k}": v for k, v in data_diagnostics.items()})

        # ── RFM diagnostics ───────────────────────────────────────────────────
        mlflow.log_metrics({f"rfm_{k}": v for k, v in rfm_diagnostics.items()
                            if isinstance(v, (int, float))})

        # ── Model params ──────────────────────────────────────────────────────
        mlflow.log_params({
            "n_clusters": n_clusters,
            "kmeans_random_state": 42,
            "kmeans_n_init": 10,
        })

        # ── Metrics ───────────────────────────────────────────────────────────
        mlflow.log_metrics({
            "kmeans_silhouette": kmeans_sil,
            "gmm_silhouette": gmm_sil,
        })

        # ── Cluster profiles ──────────────────────────────────────────────────
        for cluster_id, row in cluster_profile.iterrows():
            prefix = f"cluster_{cluster_id}"
            mlflow.log_metrics({
                f"{prefix}_recency_mean": row.get("Recency", np.nan),
                f"{prefix}_frequency_mean": row.get("Frequency", np.nan),
                f"{prefix}_monetary_mean": row.get("Monetary", np.nan),
            })

        # ── KMeans centroids as param ─────────────────────────────────────────
        if hasattr(kmeans_model, "cluster_centers_"):
            centroids = kmeans_model.cluster_centers_
            mlflow.log_param("kmeans_centroids_shape", str(centroids.shape))
            # Save centroids as numpy artifact
            centroid_path = "/tmp/kmeans_centroids.npy"
            np.save(centroid_path, centroids)
            mlflow.log_artifact(centroid_path, artifact_path="model_artifacts")

        # ── Sklearn model ─────────────────────────────────────────────────────
        mlflow.sklearn.log_model(
            kmeans_model,
            artifact_path="kmeans_model",
            registered_model_name=MODEL_NAME,
        )

        # ── Visual artifacts ──────────────────────────────────────────────────
        for path in artifact_paths:
            if path and os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="plots")

        run_id = run.info.run_id
        logger.info(f"Training run logged: run_id={run_id}")
        return run_id


def log_stability_run(
    run_period: str,
    stability_report: dict,
    artifact_paths: list,
    tracking_uri: str = "http://127.0.0.1:5001",
):
    """
    Log a stability monitoring run to its own MLflow experiment.

    Returns: run_id (str)
    """
    setup_mlflow(tracking_uri)
    exp_id = get_or_create_experiment(STABILITY_EXPERIMENT_NAME)

    with mlflow.start_run(
        experiment_id=exp_id, run_name=f"stability_{run_period}"
    ) as run:
        # ── Tags ──────────────────────────────────────────────────────────────
        mlflow.set_tags({
            "run_period": run_period,
            "pipeline_stage": "stability_monitoring",
            "overall_status": stability_report.get("overall_status", "UNKNOWN"),
        })

        # ── Scalar metrics ────────────────────────────────────────────────────
        scalar_keys = [
            "ari", "centroid_shift_mean", "centroid_shift_max",
            "silhouette_previous", "silhouette_current", "silhouette_drop",
        ]
        metrics = {k: stability_report[k] for k in scalar_keys if k in stability_report}
        mlflow.log_metrics(metrics)

        # ── Status params ─────────────────────────────────────────────────────
        mlflow.log_params({
            "ari_status": stability_report.get("ari_status", ""),
            "centroid_shift_status": stability_report.get("centroid_shift_status", ""),
            "silhouette_status": stability_report.get("silhouette_status", ""),
            "overall_status": stability_report.get("overall_status", ""),
        })

        # ── Per-cluster profile flags ─────────────────────────────────────────
        for cluster, flag in stability_report.get("cluster_profile_flags", {}).items():
            mlflow.log_metric(
                f"cluster_{cluster}_profile_max_drift", flag.get("max_drift", 0)
            )
            mlflow.log_param(
                f"cluster_{cluster}_profile_status", flag.get("status", "")
            )

        # ── Artifacts ─────────────────────────────────────────────────────────
        for path in artifact_paths:
            if path and os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="stability_plots")

        run_id = run.info.run_id
        logger.info(f"Stability run logged: run_id={run_id}")
        return run_id
