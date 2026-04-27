# =============================================================================
# FILE: 04_preprocess_basic_datasets.py
# PURPOSE: Preprocess the two basic datasets provided by the unit
#   - emails_inti.csv  → spam/ham email dataset
#   - Malware_dataset.csv → malware classification dataset
# =============================================================================

import pandas as pd
import numpy as np
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../outputs/visualizations", exist_ok=True)

# =============================================================================
# PART A: emails_inti.csv
# =============================================================================
print("=" * 60)
print("PART A: Processing emails_inti.csv (Basic Email Dataset)")
print("=" * 60)

try:
    df_email = pd.read_csv("../data/raw/emails_inti.csv", encoding='latin-1')
    print(f"✓ Loaded {len(df_email)} rows")
    print(f"Columns: {df_email.columns.tolist()}")
    print(df_email.head(3))
except FileNotFoundError:
    print("❌ emails_inti.csv not found in ../data/raw/")
    df_email = None

if df_email is not None:
    # --- Auto-detect label and text columns ---
    label_col = None
    text_col = None
    for col in df_email.columns:
        if col.lower() in ['label', 'spam', 'class', 'category', 'spam/ham', 'type', 'target']:
            label_col = col
        if col.lower() in ['message', 'text', 'body', 'email', 'content', 'subject']:
            text_col = col

    print(f"\nDetected label column: '{label_col}'")
    print(f"Detected text column:  '{text_col}'")

    # If text column not found, use the column with longest average text
    if text_col is None:
        text_cols = df_email.select_dtypes(include='object').columns
        if label_col:
            text_cols = [c for c in text_cols if c != label_col]
        text_col = max(text_cols, key=lambda c: df_email[c].astype(str).str.len().mean())
        print(f"Auto-selected text column: '{text_col}'")

    df_email = df_email[[text_col, label_col]].copy() if label_col else df_email[[text_col]].copy()
    df_email = df_email.rename(columns={text_col: 'message'})
    df_email = df_email.dropna(subset=['message'])

    # Standardize label to 0/1
    if label_col:
        df_email = df_email.rename(columns={label_col: 'label_raw'})
        def to_binary(v):
            s = str(v).lower().strip()
            if s in ['1', 'spam', 'true', 'yes', 'malicious']:
                return 1
            elif s in ['0', 'ham', 'false', 'no', 'legitimate', 'not spam']:
                return 0
            return None
        df_email['label'] = df_email['label_raw'].apply(to_binary)
        df_email = df_email.dropna(subset=['label'])
        df_email['label'] = df_email['label'].astype(int)
        print(f"\nLabel distribution:\n{df_email['label'].value_counts()}")

    # Clean text
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df_email['cleaned_message'] = df_email['message'].apply(clean_text)
    df_email['message_length'] = df_email['message'].apply(len)
    df_email['word_count'] = df_email['cleaned_message'].apply(lambda x: len(x.split()))

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=300, stop_words='english', ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(df_email['cleaned_message'])

    # Save
    save_cols = ['label', 'cleaned_message', 'message_length', 'word_count'] if label_col else ['cleaned_message', 'message_length', 'word_count']
    df_email[save_cols].to_csv("../data/processed/emails_inti_processed.csv", index=False)
    print(f"✓ Saved: ../data/processed/emails_inti_processed.csv ({len(df_email)} rows)")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('emails_inti - Basic Dataset Analysis', fontsize=13, fontweight='bold')

    if label_col:
        counts = df_email['label'].value_counts()
        axes[0].pie(counts, labels=['Ham','Spam'] if counts.index[0]==0 else ['Spam','Ham'],
                    autopct='%1.1f%%', colors=['#2ecc71','#e74c3c'])
        axes[0].set_title('Spam vs Ham')
    else:
        axes[0].text(0.5, 0.5, 'No label column', ha='center', va='center')
        axes[0].set_title('Labels')

    df_email['message_length'].clip(upper=3000).hist(ax=axes[1], bins=40, color='#3498db')
    axes[1].set_title('Message Length Distribution')
    axes[1].set_xlabel('Length (characters)')

    plt.tight_layout()
    plt.savefig('../outputs/visualizations/emails_inti_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved chart: ../outputs/visualizations/emails_inti_analysis.png")

# =============================================================================
# PART B: Malware_dataset.csv
# =============================================================================
print("\n" + "=" * 60)
print("PART B: Processing Malware_dataset.csv (Basic Malware Dataset)")
print("=" * 60)

try:
    df_mal = pd.read_csv("../data/raw/Malware dataset.csv", encoding='latin-1')
    print(f"✓ Loaded {len(df_mal)} rows, {df_mal.shape[1]} columns")
    print(f"Columns: {df_mal.columns.tolist()[:10]}...")
    print(df_mal.head(3))
except FileNotFoundError:
    print("❌ Malware_dataset.csv not found in ../data/raw/")
    df_mal = None

if df_mal is not None:
    # --- Auto-detect label column ---
    label_col = None
    for col in df_mal.columns:
        if col.lower() in ['label', 'class', 'category', 'type', 'target',
                           'malware', 'benign', 'classification']:
            label_col = col
            break

    # If still not found, check last column (common convention)
    if label_col is None:
        last_col = df_mal.columns[-1]
        if df_mal[last_col].nunique() < 20:   # Likely a label if few unique values
            label_col = last_col
            print(f"Using last column as label: '{label_col}'")

    print(f"\nLabel column: '{label_col}'")
    if label_col:
        print(f"Unique labels: {df_mal[label_col].unique()[:10]}")

    # Separate numeric features
    non_feature = [label_col] if label_col else []
    feature_cols = [c for c in df_mal.columns
                    if c not in non_feature and df_mal[c].dtype in ['int64','float64']]

    print(f"Numeric feature columns: {len(feature_cols)}")

    if len(feature_cols) == 0:
        # All object columns — try converting
        for col in df_mal.columns:
            if col != label_col:
                df_mal[col] = pd.to_numeric(df_mal[col], errors='coerce')
        feature_cols = [c for c in df_mal.columns
                        if c != label_col and df_mal[c].dtype in ['float64','int64']]
        print(f"After conversion, numeric features: {len(feature_cols)}")

    X = df_mal[feature_cols].copy()

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    # Remove low-variance
    variances = X_imputed.var()
    high_var = variances[variances > 0.001].index.tolist()
    X_filtered = X_imputed[high_var]
    print(f"Features after variance filter: {X_filtered.shape[1]}")

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_filtered), columns=X_filtered.columns)

    # Encode label
    final_df = X_scaled.copy()
    if label_col:
        le = LabelEncoder()
        final_df['label_encoded'] = le.fit_transform(df_mal[label_col].astype(str))
        final_df['label_name'] = df_mal[label_col].values
        print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    final_df.to_csv("../data/processed/malware_basic_processed.csv", index=False)
    print(f"✓ Saved: ../data/processed/malware_basic_processed.csv")
    print(f"  Shape: {final_df.shape}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Malware_dataset - Basic Dataset Analysis', fontsize=13, fontweight='bold')

    if label_col:
        counts = final_df['label_name'].value_counts().head(8)
        axes[0].bar(range(len(counts)), counts.values, color='#e74c3c')
        axes[0].set_xticks(range(len(counts)))
        axes[0].set_xticklabels([str(l)[:12] for l in counts.index], rotation=30, fontsize=8)
        axes[0].set_title('Class Distribution')
        axes[0].set_ylabel('Count')
    else:
        axes[0].text(0.5, 0.5, 'No label found', ha='center', va='center')

    # Feature variance plot
    top5 = X_filtered.var().nlargest(5)
    axes[1].barh(range(5), top5.values, color='#3498db')
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels([c[:20] for c in top5.index], fontsize=8)
    axes[1].set_title('Top 5 Features by Variance')

    plt.tight_layout()
    plt.savefig('../outputs/visualizations/malware_basic_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved chart: ../outputs/visualizations/malware_basic_analysis.png")

print("\n✅ Basic datasets preprocessing complete!")
