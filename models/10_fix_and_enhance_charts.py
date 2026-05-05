# =============================================================================
# FILE: 10_fix_and_enhance_charts.py
# PURPOSE: Fix the 2 broken charts + generate additional HD-level visualizations
#
# FIXES:
#   1. DBSCAN chart — dark background, better contrast, no white-on-white issue
#   2. MalMem category distribution — fix x-axis label overlap glitch
#
# NEW CHARTS:
#   3. Feature correlation heatmap (MalMem)
#   4. Feature distribution by malware category (violin plot)
#   5. PCA explained variance chart
#   6. Cluster size comparison (K-Means vs DBSCAN)
#   7. Top spam keywords heatmap (SMS vs Enron)
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
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

os.makedirs("../outputs/visualizations", exist_ok=True)

print("=" * 60)
print("  FIXING AND ENHANCING CHARTS")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────────
df_mal  = pd.read_csv("../data/processed/malmem_processed.csv")
df_spam = pd.read_csv("../data/processed/combined_spam_processed.csv")
df_sms  = pd.read_csv("../data/processed/sms_spam_processed.csv")

drop_mal  = [c for c in ['binary_label','category_encoded','category_name'] if c in df_mal.columns]
feat_cols = [c for c in df_mal.columns if c not in drop_mal]
X_mal     = df_mal[feat_cols].values
y_bin     = df_mal['binary_label'].values

# Rebuild proper category labels
if 'category_name' in df_mal.columns and df_mal['category_name'].nunique() <= 20:
    y_cat   = df_mal['category_name'].values
    le      = LabelEncoder()
    y_enc   = le.fit_transform(y_cat)
    cat_names = list(le.classes_)
else:
    y_cat   = np.where(y_bin == 1, 'Malware', 'Benign')
    y_enc   = y_bin
    cat_names = ['Benign', 'Malware']

print(f"✓ Data loaded — {len(df_mal)} MalMem rows, categories: {cat_names}")

# Load clustering models
with open("../outputs/models/kmeans_malware.pkl", "rb") as f:
    km_data  = pickle.load(f)
    km_model = km_data['model']
    km_pca   = km_data['pca']

with open("../outputs/models/dbscan_malware.pkl", "rb") as f:
    db_data  = pickle.load(f)
    db_model = db_data['model']
    db_pca   = db_data['pca']

# Sample for speed
np.random.seed(42)
if len(X_mal) > 15000:
    idx   = np.random.choice(len(X_mal), 15000, replace=False)
    X_s   = X_mal[idx]
    y_s   = y_enc[idx]
    y_cs  = y_cat[idx]
    y_bs  = y_bin[idx]
else:
    X_s, y_s, y_cs, y_bs = X_mal, y_enc, y_cat, y_bin

# PCA projections
pca2 = PCA(n_components=2, random_state=42)
X_2d = pca2.fit_transform(X_s)
X_db = db_pca.transform(X_s)
db_labels = db_model.fit_predict(X_db)

print(f"DBSCAN: {len(set(db_labels))- (1 if -1 in db_labels else 0)} clusters, "
      f"{(db_labels==-1).sum()} anomalies")

# =============================================================================
# FIX 1: DBSCAN Chart — Dark background, high contrast colours
# =============================================================================
print("\n[1] Fixing DBSCAN scatter plot...")

# Use a dark style so clusters pop against the background
plt.style.use('dark_background')

CLUSTER_COLORS = [
    '#FF6B6B',  # coral red
    '#4ECDC4',  # teal
    '#45B7D1',  # sky blue
    '#FFA07A',  # salmon orange
    '#98D8C8',  # mint
    '#F7DC6F',  # yellow
    '#BB8FCE',  # lavender
    '#82E0AA',  # green
    '#F1948A',  # pink
    '#85C1E9',  # light blue
]

unique_labels = sorted(set(db_labels))
n_clusters    = len([l for l in unique_labels if l != -1])
n_noise       = (db_labels == -1).sum()
noise_pct     = n_noise / len(db_labels) * 100

fig, ax = plt.subplots(figsize=(11, 8))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')

# Plot noise/anomalies first (behind clusters)
noise_mask = db_labels == -1
ax.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1],
           c='#555577', s=6, alpha=0.4,
           label=f'Anomalies — {n_noise} ({noise_pct:.1f}%)', zorder=1)

# Plot each cluster with a distinct bright colour
shown = 0
for i, lbl in enumerate([l for l in unique_labels if l != -1]):
    mask  = db_labels == lbl
    count = mask.sum()
    color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=color, s=10, alpha=0.75,
               label=f'Cluster {lbl} — {count:,} samples',
               zorder=2 + i)
    shown += 1
    if shown >= 8:   # Only show first 8 in legend (rest still plotted)
        break

# Remaining clusters (no legend entry)
for i, lbl in enumerate([l for l in unique_labels if l != -1][8:]):
    mask  = db_labels == lbl
    color = CLUSTER_COLORS[(i + 8) % len(CLUSTER_COLORS)]
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=color, s=10, alpha=0.75, zorder=10)

ax.set_title(f'DBSCAN Clustering — {n_clusters} Clusters, '
             f'{n_noise} Anomalies ({noise_pct:.1f}%)',
             fontsize=14, fontweight='bold', color='white', pad=15)
ax.set_xlabel('PCA Component 1', fontsize=11, color='#aaaacc')
ax.set_ylabel('PCA Component 2', fontsize=11, color='#aaaacc')
ax.tick_params(colors='#aaaacc')
for spine in ax.spines.values():
    spine.set_edgecolor('#444466')

legend = ax.legend(loc='upper left', fontsize=8, framealpha=0.3,
                   facecolor='#1a1a2e', edgecolor='#444466',
                   labelcolor='white', markerscale=2)

# Annotation box
ax.text(0.98, 0.02,
        f'Total samples: {len(db_labels):,}\n'
        f'Clusters found: {n_clusters}\n'
        f'Anomalies: {n_noise} ({noise_pct:.1f}%)\n'
        f'Variance explained: {pca2.explained_variance_ratio_.sum()*100:.1f}%',
        transform=ax.transAxes, fontsize=8, color='#ccccee',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                  edgecolor='#444466', alpha=0.8))

plt.tight_layout()
plt.savefig('../outputs/visualizations/dbscan_clusters_fixed.png',
            dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
plt.style.use('default')   # Reset style for subsequent charts
print("   ✓ Saved: dbscan_clusters_fixed.png")

# =============================================================================
# FIX 2: MalMem Category Distribution — fix x-axis label glitch
# =============================================================================
print("\n[2] Fixing MalMem category distribution chart...")

cat_counts = pd.Series(y_cat).value_counts().sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('CIC-MalMem-2022 — Dataset Analysis', fontsize=14, fontweight='bold')

# Chart 1: Category bar chart (fixed labels)
colors_bar = ['#2ecc71' if c == 'Benign' else '#e74c3c'
               if c == 'Ransomware' else '#3498db'
               if c == 'Spyware' else '#e67e22'
               for c in cat_counts.index]

bars = axes[0].bar(range(len(cat_counts)), cat_counts.values, color=colors_bar, width=0.6)
axes[0].set_xticks(range(len(cat_counts)))
axes[0].set_xticklabels(cat_counts.index, fontsize=11, fontweight='bold')  # No rotation needed
axes[0].set_title('Malware Category Distribution', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Sample Count')
axes[0].set_xlabel('')

# Add count labels on top of bars
for bar, count in zip(bars, cat_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].set_ylim(0, cat_counts.max() * 1.15)
axes[0].grid(True, alpha=0.3, axis='y')

# Chart 2: Pie chart (binary)
binary_counts = pd.Series(y_bin).value_counts()
axes[1].pie(binary_counts.values,
            labels=['Malware' if i == 1 else 'Benign' for i in binary_counts.index],
            autopct='%1.1f%%',
            colors=['#e74c3c', '#2ecc71'],
            startangle=90,
            textprops={'fontsize': 12})
axes[1].set_title('Benign vs Malware (Binary)', fontweight='bold', fontsize=12)

# Chart 3: Top 5 features by variance
top5 = df_mal[feat_cols].var().nlargest(5)
axes[2].barh(range(5), top5.values, color='#3498db', height=0.5)
axes[2].set_yticks(range(5))
axes[2].set_yticklabels([c[:25] for c in top5.index], fontsize=9)
axes[2].set_title('Top 5 Features by Variance', fontweight='bold', fontsize=12)
axes[2].set_xlabel('Variance')
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../outputs/visualizations/malmem_analysis_fixed.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: malmem_analysis_fixed.png")

# =============================================================================
# NEW CHART 3: Feature Correlation Heatmap (top 12 MalMem features)
# =============================================================================
print("\n[3] Feature correlation heatmap...")

top12_cols = df_mal[feat_cols].var().nlargest(12).index.tolist()
corr       = df_mal[top12_cols].corr()

# Shorten labels for readability
short_labels = [c.split('.')[-1][:14] for c in top12_cols]

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # Show lower triangle only

sns.heatmap(corr,
            ax=ax,
            cmap='RdYlBu_r',
            annot=True,
            fmt='.2f',
            annot_kws={'size': 8},
            xticklabels=short_labels,
            yticklabels=short_labels,
            vmin=-1, vmax=1,
            linewidths=0.5,
            linecolor='#dddddd',
            cbar_kws={'label': 'Pearson Correlation Coefficient'})

ax.set_title('CIC-MalMem-2022 — Feature Correlation Heatmap\n(Top 12 Features by Variance)',
             fontsize=13, fontweight='bold', pad=15)
ax.tick_params(axis='x', rotation=40, labelsize=9)
ax.tick_params(axis='y', rotation=0,  labelsize=9)

plt.tight_layout()
plt.savefig('../outputs/visualizations/malmem_correlation_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: malmem_correlation_heatmap.png")

# =============================================================================
# NEW CHART 4: Feature Distribution by Category (Box plot)
# Shows how key features differ across malware families
# =============================================================================
print("\n[4] Feature distribution by malware category (box plot)...")

top3_cols = df_mal[feat_cols].var().nlargest(3).index.tolist()

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('CIC-MalMem-2022 — Top Feature Distributions by Malware Category',
             fontsize=13, fontweight='bold')

palette = {'Benign': '#2ecc71', 'Ransomware': '#e74c3c',
           'Spyware': '#3498db', 'Trojan': '#e67e22'}

plot_df = pd.DataFrame({
    'category': y_cat,
    **{col: df_mal[col].values for col in top3_cols}
})

for i, col in enumerate(top3_cols):
    present_cats = [c for c in cat_names if c in plot_df['category'].values]
    present_pal  = {c: palette.get(c, '#9b59b6') for c in present_cats}

    sns.boxplot(data=plot_df, x='category', y=col,
                palette=present_pal, ax=axes[i],
                order=present_cats, width=0.5,
                flierprops=dict(marker='o', markersize=2, alpha=0.3))

    short_name = col.split('.')[-1]
    axes[i].set_title(f'Feature: {short_name}', fontweight='bold')
    axes[i].set_xlabel('Category')
    axes[i].set_ylabel('Normalised Value')
    axes[i].tick_params(axis='x', rotation=15)
    axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../outputs/visualizations/malmem_feature_boxplot.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: malmem_feature_boxplot.png")

# =============================================================================
# NEW CHART 5: PCA Explained Variance (Scree Plot)
# Justifies why we used PCA before clustering
# =============================================================================
print("\n[5] PCA scree plot...")

pca_full = PCA(random_state=42)
pca_full.fit(X_mal[:5000])   # Fit on subset for speed

explained     = pca_full.explained_variance_ratio_
cumulative    = np.cumsum(explained)
n_components  = min(20, len(explained))

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.bar(range(1, n_components+1), explained[:n_components]*100,
        color='#3498db', alpha=0.7, label='Individual variance (%)')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Individual Explained Variance (%)', color='#3498db')
ax1.tick_params(axis='y', labelcolor='#3498db')

ax2 = ax1.twinx()
ax2.plot(range(1, n_components+1), cumulative[:n_components]*100,
         'ro-', linewidth=2, markersize=5, label='Cumulative variance (%)')
ax2.axhline(90, color='gray', linestyle='--', alpha=0.6)
ax2.axhline(95, color='orange', linestyle='--', alpha=0.6)
ax2.text(n_components*0.6, 91, '90% threshold', fontsize=9, color='gray')
ax2.text(n_components*0.6, 96, '95% threshold', fontsize=9, color='orange')
ax2.set_ylabel('Cumulative Explained Variance (%)', color='#e74c3c')
ax2.tick_params(axis='y', labelcolor='#e74c3c')
ax2.set_ylim(0, 105)

# Find where cumulative hits 90% and 95%
idx90 = np.argmax(cumulative >= 0.90) + 1
idx95 = np.argmax(cumulative >= 0.95) + 1
ax2.axvline(idx90, color='gray',   linestyle=':', alpha=0.7)
ax2.axvline(idx95, color='orange', linestyle=':', alpha=0.7)
ax2.text(idx90+0.2, 50, f'n={idx90}\n({cumulative[idx90-1]*100:.0f}%)',
         fontsize=8, color='gray')

ax1.set_title('PCA Scree Plot — CIC-MalMem-2022\n'
              'Variance Explained per Principal Component',
              fontsize=13, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='center right', fontsize=9)

plt.tight_layout()
plt.savefig('../outputs/visualizations/pca_scree_plot.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: pca_scree_plot.png")

# =============================================================================
# NEW CHART 6: Spam Keyword Heatmap
# Shows which spam keywords appear most in spam vs ham
# =============================================================================
print("\n[6] Spam keyword heatmap...")

keyword_cols = [c for c in df_sms.columns if c.startswith('has_')]

if keyword_cols:
    kw_names = [c.replace('has_', '') for c in keyword_cols]

    spam_freq = df_sms[df_sms['label_encoded']==1][keyword_cols].mean() * 100
    ham_freq  = df_sms[df_sms['label_encoded']==0][keyword_cols].mean() * 100

    heat_df = pd.DataFrame({
        'Spam (%)': spam_freq.values,
        'Ham (%)':  ham_freq.values
    }, index=kw_names)

    heat_df = heat_df.sort_values('Spam (%)', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(heat_df,
                ax=ax,
                cmap='YlOrRd',
                annot=True,
                fmt='.1f',
                annot_kws={'size': 10},
                linewidths=0.5,
                cbar_kws={'label': 'Frequency (%)'},
                vmin=0)

    ax.set_title('Spam Keyword Frequency Heatmap\n'
                 '(% of messages containing each keyword)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Message Class')
    ax.set_ylabel('Keyword')
    ax.tick_params(axis='x', rotation=0, labelsize=11)
    ax.tick_params(axis='y', rotation=0, labelsize=10)

    plt.tight_layout()
    plt.savefig('../outputs/visualizations/spam_keyword_heatmap.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: spam_keyword_heatmap.png")
else:
    print("   ⚠ No keyword columns found in SMS dataset — skipping")

# =============================================================================
# NEW CHART 7: K-Means vs DBSCAN Cluster Size Comparison
# =============================================================================
print("\n[7] Cluster size comparison chart...")

km_pca_full = PCA(n_components=min(10, X_s.shape[1]), random_state=42)
X_km        = km_pca_full.fit_transform(X_s)
km_labels   = km_model.predict(X_km)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Clustering Results — K-Means vs DBSCAN',
             fontsize=13, fontweight='bold')

# K-Means cluster sizes
km_unique, km_counts = np.unique(km_labels, return_counts=True)
km_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12',
             '#9b59b6', '#1abc9c', '#e67e22', '#34495e'][:len(km_unique)]
bars = axes[0].bar([f'Cluster {i}' for i in km_unique],
                   km_counts, color=km_colors, width=0.5)
for bar, count in zip(bars, km_counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{count:,}', ha='center', va='bottom', fontsize=9)
axes[0].set_title('K-Means — Cluster Sizes', fontweight='bold')
axes[0].set_ylabel('Number of Samples')
axes[0].set_xlabel('Cluster')
axes[0].grid(True, alpha=0.3, axis='y')

# DBSCAN cluster sizes (exclude noise)
db_valid   = db_labels[db_labels != -1]
db_unique, db_counts = np.unique(db_valid, return_counts=True)
# Sort by count descending
sort_idx   = np.argsort(db_counts)[::-1]
db_unique  = db_unique[sort_idx][:8]
db_counts  = db_counts[sort_idx][:8]

db_colors = ['#e74c3c', '#4ECDC4', '#45B7D1', '#FFA07A',
             '#BB8FCE', '#F7DC6F', '#82E0AA', '#F1948A'][:len(db_unique)]
bars2 = axes[1].bar([f'C{i}' for i in db_unique],
                    db_counts, color=db_colors, width=0.5)
for bar, count in zip(bars2, db_counts):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{count:,}', ha='center', va='bottom', fontsize=9)

n_noise_db = (db_labels == -1).sum()
axes[1].set_title(f'DBSCAN — Cluster Sizes\n'
                  f'(+{n_noise_db} anomalies not shown)',
                  fontweight='bold')
axes[1].set_ylabel('Number of Samples')
axes[1].set_xlabel('Cluster (top 8 shown)')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../outputs/visualizations/cluster_size_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: cluster_size_comparison.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("ALL CHARTS GENERATED")
print("=" * 60)

charts = [
    ("dbscan_clusters_fixed.png",       "FIX: DBSCAN dark background, high contrast"),
    ("malmem_analysis_fixed.png",       "FIX: MalMem category labels corrected"),
    ("malmem_correlation_heatmap.png",  "NEW: Feature correlation heatmap"),
    ("malmem_feature_boxplot.png",      "NEW: Feature distributions by malware type"),
    ("pca_scree_plot.png",              "NEW: PCA variance explained (scree plot)"),
    ("spam_keyword_heatmap.png",        "NEW: Spam keyword frequency heatmap"),
    ("cluster_size_comparison.png",     "NEW: K-Means vs DBSCAN cluster sizes"),
]

for fname, desc in charts:
    path   = f"../outputs/visualizations/{fname}"
    exists = "✓" if os.path.exists(path) else "✗"
    size   = f"({os.path.getsize(path)//1024} KB)" if os.path.exists(path) else ""
    print(f"  {exists} {desc:<45} {size}")

print("\n✅ Done! Use the fixed versions in your report.")
print("   Old charts can be replaced with the *_fixed.png versions.")
