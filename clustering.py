"""
Clustering — Customer Segmentation Pipeline
Runs KMeans and Gaussian Mixture Model clustering on RFM features.
Includes elbow analysis and silhouette evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import os

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

N_CLUSTERS = 4
RANDOM_STATE = 42


def scale_features(rfm: pd.DataFrame, feature_cols=None) -> tuple:
    """
    Standardise RFM features.

    Returns:
        (scaled_array, scaler, feature_cols)
    """
    if feature_cols is None:
        feature_cols = ["Recency", "Frequency", "Monetary"]
    X = rfm[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, feature_cols


def run_elbow_analysis(X_scaled: np.ndarray, k_range=range(2, 11)) -> dict:
    """
    Run elbow method. Returns distortions dict and saves a plot.
    """
    distortions = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        distortions[k] = km.inertia_
    return distortions


def plot_elbow(distortions: dict, output_dir: str) -> str:
    """Save elbow plot and return file path."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(distortions.keys()), list(distortions.values()), marker="o", color="#2E75B6")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Distortion)")
    ax.set_title("Elbow Method for Optimal k")
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "elbow_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Elbow plot saved: {path}")
    return path


def run_kmeans(X_scaled: np.ndarray, n_clusters: int = N_CLUSTERS) -> tuple:
    """
    Fit KMeans. Returns (labels, centroids, model).
    """
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    logger.info(f"KMeans (k={n_clusters}): silhouette={sil:.4f}")
    return labels, km.cluster_centers_, km, sil


def run_gmm(X_scaled: np.ndarray, n_components: int = N_CLUSTERS) -> tuple:
    """
    Fit GMM. Returns (labels, model, silhouette).
    """
    gmm = GaussianMixture(n_components=n_components, random_state=RANDOM_STATE)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    logger.info(f"GMM (k={n_components}): silhouette={sil:.4f}")
    return labels, gmm, sil


def plot_pca_clusters(X_scaled: np.ndarray, labels: np.ndarray,
                      title: str, output_dir: str, filename: str) -> str:
    """PCA 2D cluster scatter plot."""
    os.makedirs(output_dir, exist_ok=True)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                         cmap="tab10", alpha=0.6, s=20)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PCA plot saved: {path}")
    return path


def plot_tsne_clusters(X_scaled: np.ndarray, labels: np.ndarray,
                       title: str, output_dir: str, filename: str) -> str:
    """t-SNE 2D cluster scatter plot."""
    os.makedirs(output_dir, exist_ok=True)
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
    coords = tsne.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                         cmap="tab10", alpha=0.6, s=20)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"t-SNE plot saved: {path}")
    return path


def get_cluster_profiles(rfm: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """
    Compute mean RFM per cluster.
    Returns a DataFrame for logging / comparison.
    """
    profile = rfm.groupby(cluster_col)[["Recency", "Frequency", "Monetary"]].mean()
    return profile.round(2)
