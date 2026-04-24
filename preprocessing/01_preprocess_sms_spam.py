# =============================================================================
# FILE: 01_preprocess_sms_spam.py
# PURPOSE: Clean and preprocess the UCI SMS Spam Collection dataset
# WHAT IT DOES:
#   - Loads the raw SMS spam CSV file
#   - Cleans the text (removes punctuation, lowercases, etc.)
#   - Extracts useful features (message length, word count, etc.)
#   - Converts text to numbers using TF-IDF (so ML models can read it)
#   - Saves the cleaned dataset as a new CSV file
# =============================================================================

# --- Import required libraries ---
import pandas as pd           # For loading and manipulating data (like Excel in Python)
import numpy as np            # For numerical operations
import re                     # For text cleaning using regular expressions
import os                     # For file and folder operations
import matplotlib
matplotlib.use('Agg')         # Non-interactive backend — charts save to file, no popups
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Setup: Create output folder if it doesn't exist ---
os.makedirs("../data/processed", exist_ok=True)   # Where we save cleaned data
os.makedirs("../outputs/visualizations", exist_ok=True)  # Where we save charts

print("=" * 60)
print("STEP 1: Loading SMS Spam Dataset")
print("=" * 60)

# --- Load the dataset ---
# The UCI SMS Spam file uses tab (\t) as separator and has no header row
# Column 0 = label (spam/ham), Column 1 = message text
try:
    df = pd.read_csv(
        "../data/raw/SMSSpamCollection",   # Path to your downloaded file
        sep='\t',                          # Tab-separated
        header=None,                       # No column names in file
        names=['label', 'message'],        # We give our own column names
        encoding='latin-1'                 # Encoding to handle special characters
    )
    print(f"✓ Loaded {len(df)} SMS messages successfully!")
except FileNotFoundError:
    # If file has a different name, try the CSV version
    df = pd.read_csv(
        "../data/raw/spam.csv",
        encoding='latin-1',
        usecols=[0, 1]          # Only use first two columns
    )
    df.columns = ['label', 'message']
    print(f"✓ Loaded {len(df)} SMS messages from spam.csv!")

# --- Preview the data ---
print("\nFirst 5 rows of raw data:")
print(df.head())
print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

# =============================================================================
# STEP 2: Check for data quality issues
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Checking Data Quality")
print("=" * 60)

# Check for missing values (empty cells)
print(f"Missing values:\n{df.isnull().sum()}")

# Check how many spam vs ham messages we have
print(f"\nLabel distribution:")
print(df['label'].value_counts())
print(f"\nSpam percentage: {df['label'].value_counts(normalize=True)['spam']*100:.1f}%")

# Remove any duplicate messages
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"\nRemoved {before - after} duplicate rows. {after} rows remaining.")

# =============================================================================
# STEP 3: Clean the text messages
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Cleaning Text Messages")
print("=" * 60)

def clean_text(text):
    """
    This function cleans a single text message.
    Steps:
    1. Convert to lowercase (so "FREE" and "free" are treated the same)
    2. Remove URLs (links don't help identify spam patterns)
    3. Remove email addresses
    4. Remove numbers (usually not meaningful for spam detection)
    5. Remove punctuation and special characters
    6. Remove extra whitespace
    """
    text = str(text).lower()                          # Lowercase everything
    text = re.sub(r'http\S+|www\S+', '', text)       # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)              # Remove emails
    text = re.sub(r'\d+', '', text)                  # Remove numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)          # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()         # Remove extra spaces
    return text

# Apply cleaning to every message in the dataset
df['cleaned_message'] = df['message'].apply(clean_text)

print("Example - Before cleaning:")
print(f"  '{df['message'].iloc[0]}'")
print("After cleaning:")
print(f"  '{df['cleaned_message'].iloc[0]}'")

# =============================================================================
# STEP 4: Feature Engineering
# Creating extra useful columns that help ML models detect spam
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Feature Engineering")
print("=" * 60)

# Feature 1: Length of the message (spam messages tend to be longer)
df['message_length'] = df['message'].apply(len)

# Feature 2: Number of words
df['word_count'] = df['message'].apply(lambda x: len(x.split()))

# Feature 3: Number of capital letters (spam often uses CAPS)
df['capital_count'] = df['message'].apply(lambda x: sum(1 for c in x if c.isupper()))

# Feature 4: Number of special characters like !, $, ?
df['special_char_count'] = df['message'].apply(
    lambda x: len(re.findall(r'[!$?]', x))
)

# Feature 5: Does the message contain "free", "win", "prize" etc.? (common spam words)
spam_keywords = ['free', 'win', 'winner', 'prize', 'call', 'txt', 'claim', 'urgent', 'guaranteed']
for keyword in spam_keywords:
    df[f'has_{keyword}'] = df['cleaned_message'].apply(
        lambda x: 1 if keyword in x.split() else 0
    )

print("New features created:")
print(df[['label', 'message_length', 'word_count', 'capital_count', 'special_char_count']].head())

# =============================================================================
# STEP 5: Encode labels
# ML models need numbers, not words like "spam" and "ham"
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Encoding Labels")
print("=" * 60)

# Convert "spam" -> 1, "ham" -> 0
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
# This should print: {'ham': 0, 'spam': 1}

# =============================================================================
# STEP 6: TF-IDF Vectorization
# TF-IDF converts text into numbers that represent how important each word is
# TF = how often a word appears in THIS message
# IDF = how rare the word is across ALL messages (rare words = more informative)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: TF-IDF Text Vectorization")
print("=" * 60)

tfidf = TfidfVectorizer(
    max_features=500,    # Keep only top 500 most important words
    stop_words='english', # Remove common words like "the", "is", "and"
    ngram_range=(1, 2)   # Consider single words AND pairs of words
)

# Fit and transform the cleaned messages
tfidf_matrix = tfidf.fit_transform(df['cleaned_message'])

# Convert to a dataframe so we can see it
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f'tfidf_{word}' for word in tfidf.get_feature_names_out()]
)

print(f"TF-IDF matrix shape: {tfidf_df.shape}")
print("(Each row = 1 message, each column = 1 word's importance score)")

# =============================================================================
# STEP 7: Create final processed dataset
# Combine the original features + TF-IDF features
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Saving Processed Dataset")
print("=" * 60)

# Save the main processed dataset (without TF-IDF - that's for model training)
processed_df = df[['label', 'label_encoded', 'cleaned_message',
                    'message_length', 'word_count', 'capital_count',
                    'special_char_count'] +
                   [f'has_{kw}' for kw in spam_keywords]]

processed_df.to_csv("../data/processed/sms_spam_processed.csv", index=False)
print(f"✓ Saved processed SMS data: ../data/processed/sms_spam_processed.csv")
print(f"  Shape: {processed_df.shape}")

# Save TF-IDF features separately (used during model training)
tfidf_df['label_encoded'] = df['label_encoded'].values
tfidf_df.to_csv("../data/processed/sms_spam_tfidf.csv", index=False)
print(f"✓ Saved TF-IDF features: ../data/processed/sms_spam_tfidf.csv")

# =============================================================================
# STEP 8: Visualizations
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Creating Visualizations")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('SMS Spam Dataset - Exploratory Analysis', fontsize=14, fontweight='bold')

# Chart 1: Spam vs Ham distribution (pie chart)
counts = df['label'].value_counts()
axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'])
axes[0].set_title('Spam vs Ham Distribution')

# Chart 2: Message length distribution
df[df['label']=='ham']['message_length'].hist(
    ax=axes[1], alpha=0.7, color='#2ecc71', label='Ham', bins=30)
df[df['label']=='spam']['message_length'].hist(
    ax=axes[1], alpha=0.7, color='#e74c3c', label='Spam', bins=30)
axes[1].set_title('Message Length Distribution')
axes[1].set_xlabel('Message Length (characters)')
axes[1].legend()

# Chart 3: Average word count by label
df.groupby('label')['word_count'].mean().plot(
    kind='bar', ax=axes[2], color=['#2ecc71', '#e74c3c'])
axes[2].set_title('Average Word Count by Label')
axes[2].set_xlabel('Label')
axes[2].set_ylabel('Average Word Count')
axes[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('../outputs/visualizations/sms_spam_analysis.png', dpi=150, bbox_inches='tight')
plt.close()   # Save silently, no popup window
print("✓ Saved chart: ../outputs/visualizations/sms_spam_analysis.png")

print("\n✅ SMS Spam preprocessing complete!")
print(f"   Final dataset: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
