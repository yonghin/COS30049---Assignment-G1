# =============================================================================
# FILE: 02_preprocess_malmem.py
# PURPOSE: Clean and preprocess the CIC-MalMem-2022 malware dataset
# WHAT IT DOES:
#   - Loads the malware memory analysis CSV
#   - Checks for and handles missing values
#   - Removes irrelevant columns
#   - Normalizes numerical features (so all numbers are on same scale)
#   - Encodes category labels for multi-class classification
#   - Saves cleaned dataset ready for ML models
# =============================================================================

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — no popup windows
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer   # Fills in missing values

# --- Setup output folders ---
os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../outputs/visualizations", exist_ok=True)

print("=" * 60)
print("STEP 1: Loading CIC-MalMem-2022 Dataset")
print("=" * 60)

# --- Load the dataset ---
# The CIC-MalMem-2022 dataset from Kaggle is a CSV file
# It contains memory features extracted from Windows processes
try:
    df = pd.read_csv("../data/raw/Obfuscated-MalMem2022.csv")
    print(f"✓ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    # Try alternative filename
    df = pd.read_csv("../data/raw/MalMemAnalysis.csv")
    print(f"✓ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nFirst 3 rows:")
print(df.head(3))
print("\nColumn names:")
print(df.columns.tolist())

# =============================================================================
# STEP 2: Understand the dataset structure
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Dataset Overview")
print("=" * 60)

print(f"\nData types:\n{df.dtypes.value_counts()}")
print(f"\nMissing values per column (top 10):")
missing = df.isnull().sum()
print(missing[missing > 0].head(10))
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# Check what the label column looks like
# CIC-MalMem has columns like 'Category' and 'Class'
# 'Category' = type of malware (Spyware, Ransomware, Trojan, Benign)
# 'Class' = binary label (Malware or Benign)
label_col = None
for possible_label in ['Category', 'Class', 'Label', 'label', 'category']:
    if possible_label in df.columns:
        label_col = possible_label
        break

print(f"\nLabel column found: '{label_col}'")
print(f"Unique labels:\n{df[label_col].value_counts()}")

# =============================================================================
# STEP 3: Separate features from labels
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Separating Features and Labels")
print("=" * 60)

# Save the label columns before we drop them
# Binary label: Malware (1) or Benign (0)
# Multi-class label: Spyware, Ransomware, Trojan, Benign

# Identify all non-numeric / label columns to exclude from features
non_feature_cols = []
for col in df.columns:
    if df[col].dtype == 'object':   # Text columns are not features
        non_feature_cols.append(col)

print(f"Non-numeric (text) columns: {non_feature_cols}")

# Create binary label: 1 = malware, 0 = benign
if 'Class' in df.columns:
    df['binary_label'] = (df['Class'].str.lower() == 'malware').astype(int)
elif 'Category' in df.columns:
    df['binary_label'] = (df['Category'].str.lower() != 'benign').astype(int)

# Create multi-class label (for clustering and multi-class classification)
le_multi = LabelEncoder()
df['category_encoded'] = le_multi.fit_transform(df[label_col])
category_mapping = dict(zip(le_multi.classes_, le_multi.transform(le_multi.classes_)))
print(f"\nCategory encoding: {category_mapping}")

# Feature columns = all numeric columns (excluding labels)
feature_cols = [col for col in df.columns
                if col not in non_feature_cols
                and col not in ['binary_label', 'category_encoded']]

print(f"\nNumber of feature columns: {len(feature_cols)}")

# =============================================================================
# STEP 4: Handle missing values
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Handling Missing Values")
print("=" * 60)

X = df[feature_cols].copy()

# Strategy: fill missing numeric values with the MEDIAN of that column
# Median is better than mean when there are outliers
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=feature_cols
)

print(f"Missing values before: {X.isnull().sum().sum()}")
print(f"Missing values after:  {X_imputed.isnull().sum().sum()}")

# =============================================================================
# STEP 5: Remove low-variance features
# Features that barely change across samples don't help the model
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Removing Low-Variance Features")
print("=" * 60)

# Calculate variance for each feature
variances = X_imputed.var()

# Keep only features with variance > 0.01 (very low threshold)
high_var_cols = variances[variances > 0.01].index.tolist()
X_filtered = X_imputed[high_var_cols]

print(f"Features before: {X_imputed.shape[1]}")
print(f"Features after removing low-variance: {X_filtered.shape[1]}")
print(f"Removed {X_imputed.shape[1] - X_filtered.shape[1]} low-variance features")

# =============================================================================
# STEP 6: Normalize/Scale features
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Normalizing Features (StandardScaler)")
print("=" * 60)

# WHY SCALING? Different features have very different ranges.
# Example: one feature might be 0-1, another might be 0-1,000,000
# Scaling makes them all comparable (mean=0, std=1)
# This is important for models like SVM and K-Means

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_filtered),
    columns=X_filtered.columns
)

print(f"Before scaling - Feature 0: mean={X_filtered.iloc[:,0].mean():.2f}, std={X_filtered.iloc[:,0].std():.2f}")
print(f"After scaling  - Feature 0: mean={X_scaled.iloc[:,0].mean():.2f}, std={X_scaled.iloc[:,0].std():.2f}")

# =============================================================================
# STEP 7: Build and save the final processed dataset
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Saving Processed Dataset")
print("=" * 60)

# Combine scaled features with labels
final_df = X_scaled.copy()
final_df['binary_label'] = df['binary_label'].values
final_df['category_encoded'] = df['category_encoded'].values
final_df['category_name'] = df[label_col].values   # Keep human-readable label too

# Save
final_df.to_csv("../data/processed/malmem_processed.csv", index=False)
print(f"✓ Saved: ../data/processed/malmem_processed.csv")
print(f"  Shape: {final_df.shape}")
print(f"  Benign samples:  {(final_df['binary_label']==0).sum()}")
print(f"  Malware samples: {(final_df['binary_label']==1).sum()}")

# Save the category mapping for reference
# NOTE: Convert int64 values to regular Python int first — JSON can't handle numpy int64
import json
category_mapping_serializable = {k: int(v) for k, v in category_mapping.items()}
with open("../data/processed/category_mapping.json", "w") as f:
    json.dump(category_mapping_serializable, f, indent=2)
print(f"✓ Saved category mapping: ../data/processed/category_mapping.json")

# Also save the scaler — we need it later to scale new data for predictions
import pickle
with open("../data/processed/malmem_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print(f"✓ Saved scaler: ../data/processed/malmem_scaler.pkl")

# =============================================================================
# STEP 8: Visualizations
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Creating Visualizations")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('CIC-MalMem-2022 - Dataset Analysis', fontsize=14, fontweight='bold')

# Chart 1: Malware category distribution
category_counts = df[label_col].value_counts()
colors = ['#2ecc71', '#e74c3c', '#e67e22', '#9b59b6']
axes[0].bar(category_counts.index, category_counts.values, color=colors[:len(category_counts)])
axes[0].set_title('Malware Category Distribution')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=15)

# Chart 2: Benign vs Malware (binary)
binary_counts = final_df['binary_label'].value_counts()
axes[1].pie(binary_counts, labels=['Benign', 'Malware'],
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
axes[1].set_title('Benign vs Malware (Binary)')

# Chart 3: Top 5 most important features by variance (bar chart — much faster than heatmap)
top5 = X_filtered.var().nlargest(5)
axes[2].barh(range(5), top5.values, color='#3498db')
axes[2].set_yticks(range(5))
axes[2].set_yticklabels([c[:20] for c in top5.index], fontsize=8)
axes[2].set_title('Top 5 Features by Variance')
axes[2].set_xlabel('Variance')

plt.tight_layout()
plt.savefig('../outputs/visualizations/malmem_analysis.png', dpi=150, bbox_inches='tight')
plt.close()   # Close silently — chart is saved to file, no popup window
print("✓ Saved chart: ../outputs/visualizations/malmem_analysis.png")

print("\n✅ MalMem preprocessing complete!")
