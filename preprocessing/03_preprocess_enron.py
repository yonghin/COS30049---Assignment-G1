# =============================================================================
# FILE: 03_preprocess_enron.py
# PURPOSE: Clean and preprocess the Enron Email Spam dataset
# WHAT IT DOES:
#   - Loads the Enron email CSV from Kaggle
#   - Cleans email text (removes headers, HTML, special chars)
#   - Extracts email-specific features (subject line, body length, etc.)
#   - Applies TF-IDF vectorization
#   - Saves processed data ready for ML models
# =============================================================================

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Setup output folders ---
os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../outputs/visualizations", exist_ok=True)

print("=" * 60)
print("STEP 1: Loading Enron Email Dataset")
print("=" * 60)

# --- Load the Enron dataset ---
# CORRECT DATASET: "Enron Spam Data" by marcelwiechmann on Kaggle
# URL: https://www.kaggle.com/datasets/marcelwiechmann/enron-spam-data
# This has ~33,716 emails with a 'Spam/Ham' label column.
# DO NOT use: wcukierski/enron-email-dataset (that one has 517k unlabeled emails)

try:
    df = pd.read_csv("../data/raw/enron_spam_data.csv")
    print(f"✓ Loaded {len(df)} emails")
    print(f"Columns: {df.columns.tolist()}")
except FileNotFoundError:
    print("❌ File not found!")
    print("Please download the CORRECT Enron dataset:")
    print("  → https://www.kaggle.com/datasets/marcelwiechmann/enron-spam-data")
    print("  → Save the file as: ../data/raw/enron_spam_data.csv")
    exit()

print("\nFirst 2 rows:")
print(df.head(2))

# =============================================================================
# STEP 2: Standardize column names
# marcelwiechmann dataset columns: Message ID, Subject, Message, Spam/Ham, Date
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Standardizing Column Names")
print("=" * 60)

print(f"All columns found: {df.columns.tolist()}")

# Find the message/text column
message_col = None
for col in df.columns:
    if col.lower() in ['message', 'text', 'body', 'email', 'content']:
        message_col = col
        break

# Find the label column — marcelwiechmann uses 'Spam/Ham'
label_col = None
for col in df.columns:
    if col.lower() in ['spam/ham', 'spam', 'label', 'class', 'category', 'target', 'is_spam']:
        label_col = col
        break

print(f"Message column detected: '{message_col}'")
print(f"Label column detected:   '{label_col}'")

if label_col is None or message_col is None:
    print("❌ Could not auto-detect columns. Columns in your file:")
    print(df.columns.tolist())
    print("Update message_col and label_col manually above.")
    exit()

# Rename to standard names used throughout this project
df = df.rename(columns={message_col: 'message', label_col: 'label_raw'})
df = df[['message', 'label_raw']].copy()
df = df.dropna()
print(f"After removing empty rows: {len(df)} emails")

# =============================================================================
# STEP 3: Standardize labels
# Different versions use: 0/1, "spam"/"ham", "spam"/"not spam", True/False
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Standardizing Labels")
print("=" * 60)

print(f"Unique label values found: {df['label_raw'].unique()[:10]}")

# Convert everything to binary: 1 = spam, 0 = not spam (ham)
def standardize_label(val):
    """Converts any label format to 0 (ham) or 1 (spam)"""
    val_str = str(val).lower().strip()
    if val_str in ['1', 'spam', 'true', 'yes']:
        return 1
    elif val_str in ['0', 'ham', 'false', 'no', 'not spam']:
        return 0
    else:
        return None  # Unknown labels we'll drop

df['label'] = df['label_raw'].apply(standardize_label)
df = df.dropna(subset=['label'])       # Drop rows with unknown labels
df['label'] = df['label'].astype(int)

print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"Spam: {df['label'].sum()}, Ham: {(df['label']==0).sum()}")

# =============================================================================
# STEP 4: Clean email text
# Emails have a lot of noise: headers, HTML tags, reply chains, etc.
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Cleaning Email Text")
print("=" * 60)

def clean_email(text):
    """
    Cleans a raw email message.
    Removes: headers, HTML, URLs, email addresses, numbers, punctuation
    """
    text = str(text)

    # Remove common email header patterns
    text = re.sub(r'(From|To|Subject|Date|Cc|Bcc|Reply-To):\s.*\n', '', text)

    # Remove HTML tags (some emails are HTML formatted)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)

    # Remove special characters, keep only letters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Lowercase and remove extra whitespace
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df['cleaned_message'] = df['message'].apply(clean_email)

# Remove very short cleaned messages (less than 10 characters after cleaning)
# These are likely empty emails that won't help training
df = df[df['cleaned_message'].str.len() > 10]
print(f"After removing too-short messages: {len(df)} emails")

print("\nCleaning example:")
print(f"BEFORE: {df['message'].iloc[0][:200]}...")
print(f"AFTER:  {df['cleaned_message'].iloc[0][:200]}...")

# =============================================================================
# STEP 5: Feature Engineering (email-specific features)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Feature Engineering")
print("=" * 60)

# Email length (spam emails tend to be longer or shorter than normal)
df['email_length'] = df['message'].apply(len)

# Word count
df['word_count'] = df['cleaned_message'].apply(lambda x: len(x.split()))

# Number of capital letters (spam often uses CAPS for emphasis)
df['capital_ratio'] = df['message'].apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
)

# Contains typical spam keywords
spam_words = ['free', 'money', 'win', 'winner', 'cash', 'prize',
              'click', 'offer', 'deal', 'discount', 'unsubscribe',
              'urgent', 'limited', 'guaranteed', 'credit', 'loan']
for word in spam_words:
    df[f'has_{word}'] = df['cleaned_message'].apply(
        lambda x: 1 if f' {word} ' in f' {x} ' else 0
    )

print(f"Total features created: {len(df.columns)}")
print("Sample features:", df.columns.tolist()[:10])

# =============================================================================
# STEP 6: TF-IDF Vectorization
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: TF-IDF Vectorization")
print("=" * 60)

tfidf = TfidfVectorizer(
    max_features=1000,     # Top 1000 words (emails have more vocabulary than SMS)
    stop_words='english',  # Remove common English words
    ngram_range=(1, 2),    # Single words + word pairs
    min_df=2               # Ignore words that appear in less than 2 emails
)

tfidf_matrix = tfidf.fit_transform(df['cleaned_message'])
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f'tfidf_{w}' for w in tfidf.get_feature_names_out()]
)
tfidf_df.index = df.index   # Match index so we can combine later

print(f"TF-IDF matrix: {tfidf_df.shape[0]} emails × {tfidf_df.shape[1]} word features")

# =============================================================================
# STEP 7: Save processed datasets
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Saving Processed Data")
print("=" * 60)

# Save main features (no TF-IDF)
feature_cols = ['label', 'cleaned_message', 'email_length', 'word_count', 'capital_ratio'] + \
               [f'has_{w}' for w in spam_words]

df[feature_cols].to_csv("../data/processed/enron_processed.csv", index=False)
print(f"✓ Saved: ../data/processed/enron_processed.csv ({len(df)} rows)")

# Save TF-IDF separately
tfidf_df['label'] = df['label'].values
tfidf_df.to_csv("../data/processed/enron_tfidf.csv", index=False)
print(f"✓ Saved TF-IDF: ../data/processed/enron_tfidf.csv")

# =============================================================================
# STEP 8: Visualizations
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Enron Email Dataset - Analysis', fontsize=14, fontweight='bold')

# Chart 1: Spam vs Ham
counts = df['label'].value_counts()
axes[0].pie(counts, labels=['Ham', 'Spam'], autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'])
axes[0].set_title('Spam vs Ham Distribution')

# Chart 2: Email length by class
df[df['label']==0]['email_length'].clip(upper=5000).hist(
    ax=axes[1], alpha=0.6, color='#2ecc71', label='Ham', bins=40)
df[df['label']==1]['email_length'].clip(upper=5000).hist(
    ax=axes[1], alpha=0.6, color='#e74c3c', label='Spam', bins=40)
axes[1].set_title('Email Length Distribution')
axes[1].set_xlabel('Email Length (characters, capped at 5000)')
axes[1].legend()

# Chart 3: Top spam keyword frequency in spam emails
keyword_spam_freq = {w: df[df['label']==1][f'has_{w}'].mean() * 100
                     for w in spam_words}
keyword_spam_freq = dict(sorted(keyword_spam_freq.items(),
                                key=lambda x: x[1], reverse=True)[:10])
axes[2].barh(list(keyword_spam_freq.keys()),
             list(keyword_spam_freq.values()), color='#e74c3c')
axes[2].set_title('Spam Keyword Frequency in Spam Emails (%)')
axes[2].set_xlabel('% of Spam Emails Containing Keyword')

plt.tight_layout()
plt.savefig('../outputs/visualizations/enron_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved chart: ../outputs/visualizations/enron_analysis.png")

print("\n✅ Enron email preprocessing complete!")
