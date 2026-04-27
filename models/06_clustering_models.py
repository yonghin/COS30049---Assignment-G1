# =============================================================================
# FILE: 06_clustering_models.py
# PURPOSE: Train and evaluate 2 clustering models on malware data
#
# MODELS:
#   1. K-Means  → Group malware into families (MalMem dataset)
#   2. DBSCAN   → Detect anomalous/unknown malware patterns
#
# WHY CLUSTERING?
#   New malware appears daily. Clustering finds patterns WITHOUT needing labels.
#   K-Means groups samples into K clusters (known malware families).
#   DBSCAN marks low-density outliers as anomalies — great for zero-day detection.
#
# OUTPUTS:
#   ../outputs/models/          → saved .pkl model files
#   ../outputs/visualizations/  → elbow curve, cluster scatter plots
#   ../outputs/clustering_results.csv → metrics table
# =============================================================================

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             homogeneity_score, completeness_score)

os.makedirs("../outputs/models", exist_ok=True)
os.makedirs("../outputs/visualizations", exist_ok=True)

COLORS = ['#e74c3c','#2ecc71','#3498db','#f39c12',
          '#9b59b6','#1abc9c','#e67e22','#34495e']

# =============================================================================
# Load data
# =============================================================================
print("="*60)
print("Loading MalMem processed data")
print("="*60)

try:
    df = pd.read_csv("../data/processed/malmem_processed.csv")
    print(f"✓ Loaded: {df.shape}")
except FileNotFoundError:
    print("❌ malmem_processed.csv not found. Run preprocessing first.")
    exit()

drop_cols = [c for c in ['binary_label','category_encoded','category_name']
             if c in df.columns]
feature_cols = [c for c in df.columns if c not in drop_cols]

X_full = df[feature_cols].values

# ── Fix: determine n_classes safely ───────────────────────────────────────
# Use category_name if it has few unique values (real category labels like
# Benign, Ransomware, Spyware, Trojan). Otherwise fall back to binary (2).
from sklearn.preprocessing import LabelEncoder as _LE

if 'category_name' in df.columns and df['category_name'].nunique() <= 20:
    le_tmp = _LE()
    y_true  = le_tmp.fit_transform(df['category_name'].astype(str))
    y_names = df['category_name'].values
    n_classes = len(le_tmp.classes_)
    print(f"Using category_name: {n_classes} categories: {list(le_tmp.classes_)}")
elif 'binary_label' in df.columns:
    y_true  = df['binary_label'].values
    y_names = np.where(y_true == 1, 'Malware', 'Benign')
    n_classes = 2
    print("Using binary_label: 2 classes (Benign / Malware)")
else:
    # Fallback: use K=6 (best silhouette from elbow chart above)
    y_true  = np.zeros(len(df), dtype=int)
    y_names = np.array(['Unknown'] * len(df))
    n_classes = 6
    print("No label column found — using K=6 based on elbow silhouette scores")

print(f"K-Means will use K = {n_classes}")

# Sample for speed
if len(X_full) > 15000:
    idx = np.random.RandomState(42).choice(len(X_full), 15000, replace=False)
    X_full  = X_full[idx]
    y_true  = y_true[idx]
    y_names = y_names[idx]
    print(f"Sampled 15,000 rows for clustering speed")

# =============================================================================
# PCA — reduce dimensions
# 2D  = for plotting only
# 10D = for K-Means training (preserves more variance)
# 5D  = for DBSCAN training
# =============================================================================
print("\n" + "="*60)
print("Dimensionality Reduction with PCA")
print("="*60)

pca2  = PCA(n_components=2,  random_state=42)
pca10 = PCA(n_components=min(10, X_full.shape[1]), random_state=42)
pca5  = PCA(n_components=min(5,  X_full.shape[1]), random_state=42)

X_2d  = pca2.fit_transform(X_full)
X_10d = pca10.fit_transform(X_full)
X_5d  = pca5.fit_transform(X_full)

print(f"Variance explained by 2D  PCA: {pca2.explained_variance_ratio_.sum()*100:.1f}%")
print(f"Variance explained by 10D PCA: {pca10.explained_variance_ratio_.sum()*100:.1f}%")
print(f"Variance explained by 5D  PCA: {pca5.explained_variance_ratio_.sum()*100:.1f}%")

# =============================================================================
# Elbow method — find optimal K for K-Means
# =============================================================================
print("\n" + "="*60)
print("Elbow Method — Finding Optimal K")
print("="*60)

inertias = []
sil_scores = []
k_range = range(2, 9)

for k in k_range:
    km_temp = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
    km_temp.fit(X_10d)
    inertias.append(km_temp.inertia_)
    sil = silhouette_score(X_10d, km_temp.labels_, sample_size=3000, random_state=42)
    sil_scores.append(sil)
    print(f"  K={k}: Inertia={km_temp.inertia_:,.0f}  Silhouette={sil:.4f}")

# Elbow + Silhouette chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=6)
axes[0].set_xlabel('Number of Clusters K')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method', fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_range), sil_scores, 'rs-', linewidth=2, markersize=6)
axes[1].set_xlabel('Number of Clusters K')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score vs K', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/visualizations/kmeans_elbow.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Elbow chart saved: kmeans_elbow.png")

# =============================================================================
# MODEL 1 — K-Means Clustering
# K = number of true malware categories from the dataset
# =============================================================================
print("\n" + "="*60)
print(f"MODEL 1: K-Means (K={n_classes})")
print("="*60)

kmeans = KMeans(
    n_clusters=n_classes,
    random_state=42,
    n_init=10,      # Run 10 times with different seeds, keep best result
    max_iter=300
)
kmeans.fit(X_10d)
km_labels = kmeans.labels_

# Evaluation metrics
sil_km = silhouette_score(X_10d, km_labels, sample_size=5000, random_state=42)
ari_km = adjusted_rand_score(y_true, km_labels)
hom_km = homogeneity_score(y_true, km_labels)
com_km = completeness_score(y_true, km_labels)

print(f"  Silhouette Score:    {sil_km:.4f}  (-1 to 1, higher = better separated)")
print(f"  Adjusted Rand Index: {ari_km:.4f}  (1.0 = perfect match with true labels)")
print(f"  Homogeneity:         {hom_km:.4f}  (each cluster = 1 class)")
print(f"  Completeness:        {com_km:.4f}  (each class = 1 cluster)")

# Cluster sizes
unique, counts = np.unique(km_labels, return_counts=True)
print(f"\n  Cluster sizes:")
for cl, cnt in zip(unique, counts):
    print(f"    Cluster {cl}: {cnt} samples")

# Visualise: K-Means clusters vs true labels side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('K-Means Clustering — Malware (2D PCA)', fontsize=13, fontweight='bold')

for i in range(n_classes):
    mask = km_labels == i
    axes[0].scatter(X_2d[mask,0], X_2d[mask,1],
                    c=COLORS[i % len(COLORS)], s=5, alpha=0.5, label=f'Cluster {i}')
axes[0].set_title('K-Means Predicted Clusters')
axes[0].set_xlabel('PCA 1'); axes[0].set_ylabel('PCA 2')
axes[0].legend(markerscale=3, fontsize=8)

for i, lbl in enumerate(np.unique(y_true)):
    mask = y_true == lbl
    name = y_names[mask][0]
    axes[1].scatter(X_2d[mask,0], X_2d[mask,1],
                    c=COLORS[i % len(COLORS)], s=5, alpha=0.5, label=name)
axes[1].set_title('True Malware Categories (Ground Truth)')
axes[1].set_xlabel('PCA 1'); axes[1].set_ylabel('PCA 2')
axes[1].legend(markerscale=3, fontsize=8)

plt.tight_layout()
plt.savefig('../outputs/visualizations/kmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Cluster chart saved: kmeans_clusters.png")

with open("../outputs/models/kmeans_malware.pkl", "wb") as f:
    pickle.dump({'model': kmeans, 'pca': pca10}, f)
print("✓ Model saved: kmeans_malware.pkl")

# =============================================================================
# MODEL 2 — DBSCAN: Anomaly Detection
# Unlike K-Means, DBSCAN does NOT need K specified upfront.
# It finds dense regions and marks outliers as -1 (anomalies).
# This is very useful for detecting unknown/new malware variants.
# =============================================================================
print("\n" + "="*60)
print("MODEL 2: DBSCAN — Anomaly Detection")
print("="*60)

dbscan = DBSCAN(
    eps=0.8,         # Neighborhood radius — points within eps are neighbors
    min_samples=15,  # Min points to form a dense region (core point)
    n_jobs=-1
)
db_labels = dbscan.fit_predict(X_5d)

n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise       = (db_labels == -1).sum()
noise_pct     = n_noise / len(db_labels) * 100

print(f"  Clusters found:        {n_clusters_db}")
print(f"  Anomalies (label=-1):  {n_noise} ({noise_pct:.1f}%) — potential unknown malware")

# Silhouette only makes sense with 2+ clusters and non-noise points
valid_mask = db_labels != -1
if n_clusters_db >= 2 and valid_mask.sum() > 100:
    sil_db = silhouette_score(X_5d[valid_mask], db_labels[valid_mask], sample_size=3000)
    print(f"  Silhouette Score:      {sil_db:.4f}")
else:
    sil_db = None
    print("  Silhouette Score: N/A (too few clusters)")

# Visualise DBSCAN
fig, ax = plt.subplots(figsize=(8, 6))

# Gray = anomalies
ax.scatter(X_2d[~valid_mask, 0], X_2d[~valid_mask, 1],
           c='lightgray', s=5, alpha=0.3, label=f'Anomalies ({n_noise})')

# Colour each cluster
for i, lbl in enumerate([l for l in np.unique(db_labels) if l != -1][:8]):
    mask = db_labels == lbl
    ax.scatter(X_2d[mask,0], X_2d[mask,1],
               c=COLORS[i % len(COLORS)], s=8, alpha=0.6,
               label=f'Cluster {lbl} ({mask.sum()})')

ax.set_title(f'DBSCAN — {n_clusters_db} clusters, {n_noise} anomalies ({noise_pct:.1f}%)',
             fontweight='bold')
ax.set_xlabel('PCA 1'); ax.set_ylabel('PCA 2')
ax.legend(markerscale=2, fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig('../outputs/visualizations/dbscan_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ DBSCAN chart saved: dbscan_clusters.png")

with open("../outputs/models/dbscan_malware.pkl", "wb") as f:
    pickle.dump({'model': dbscan, 'pca': pca5}, f)
print("✓ Model saved: dbscan_malware.pkl")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("CLUSTERING RESULTS SUMMARY")
print("="*60)

results = pd.DataFrame([
    {
        'Model': 'K-Means',
        'N_Clusters': n_classes,
        'Silhouette': round(sil_km, 4),
        'Adj_Rand_Index': round(ari_km, 4),
        'Homogeneity': round(hom_km, 4),
        'Completeness': round(com_km, 4),
        'Notes': 'Groups malware into known families'
    },
    {
        'Model': 'DBSCAN',
        'N_Clusters': n_clusters_db,
        'Silhouette': round(sil_db, 4) if sil_db else 'N/A',
        'Adj_Rand_Index': 'N/A',
        'Homogeneity': 'N/A',
        'Completeness': 'N/A',
        'Notes': f'{n_noise} anomalies detected ({noise_pct:.1f}%)'
    }
])

print(results.to_string(index=False))
results.to_csv("../outputs/clustering_results.csv", index=False)
print("\n✓ Saved: ../outputs/clustering_results.csv")
print("\n✅ Clustering done! Next: python 07_regression_model.py")
