# =============================================================================
# FILE: 00_run_all_preprocessing.py
# PURPOSE: Master script - runs ALL preprocessing steps in order
# HOW TO USE: Just run this ONE file and it will process everything!
#   python 00_run_all_preprocessing.py
# =============================================================================

import pandas as pd
import numpy as np
import os
import subprocess
import sys

print("=" * 60)
print("  SPAM & MALWARE DETECTION - DATA PREPROCESSING PIPELINE")
print("=" * 60)

# --- Create all required folders ---
folders = [
    "../data/raw",
    "../data/processed",
    "../outputs/visualizations",
    "../outputs/models"
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✓ Folder ready: {folder}")

print("\n")

# =============================================================================
# RUN EACH PREPROCESSING SCRIPT IN ORDER
# =============================================================================

scripts = [
    ("01_preprocess_sms_spam.py",  "SMS Spam Collection (UCI)"),
    ("02_preprocess_malmem.py",    "CIC-MalMem-2022 (Malware)"),
    ("03_preprocess_enron.py",     "Enron Email Dataset"),
]

results = {}

for script, name in scripts:
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Script:  {script}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            results[name] = "✅ Success"
        else:
            results[name] = "❌ Failed"
    except Exception as e:
        results[name] = f"❌ Error: {e}"

# =============================================================================
# STEP: Merge SMS Spam + Enron into one combined spam dataset
# This gives us more data for better model training
# =============================================================================
print(f"\n{'='*60}")
print("MERGING: Combining SMS Spam + Enron into one spam dataset")
print(f"{'='*60}")

try:
    # Load both processed spam datasets
    sms_df = pd.read_csv("../data/processed/sms_spam_processed.csv")
    enron_df = pd.read_csv("../data/processed/enron_processed.csv")

    # Standardize column names so we can combine them
    # Both should have: label (0/1), cleaned_message, message_length or email_length

    # Rename Enron's email_length to message_length for consistency
    if 'email_length' in enron_df.columns:
        enron_df = enron_df.rename(columns={'email_length': 'message_length'})

    # Keep only the common columns that exist in both datasets
    common_cols = ['label', 'cleaned_message', 'message_length', 'word_count']

    # Add source column so we know where data came from
    sms_subset = sms_df[['label_encoded', 'cleaned_message', 'message_length', 'word_count']].copy()
    sms_subset = sms_subset.rename(columns={'label_encoded': 'label'})
    sms_subset['source'] = 'sms'

    enron_subset = enron_df[['label', 'cleaned_message', 'message_length', 'word_count']].copy()
    enron_subset['source'] = 'email'

    # Combine both datasets
    combined_spam = pd.concat([sms_subset, enron_subset], ignore_index=True)
    combined_spam = combined_spam.dropna()
    combined_spam = combined_spam.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    combined_spam.to_csv("../data/processed/combined_spam_processed.csv", index=False)

    print(f"✓ Combined spam dataset saved!")
    print(f"  SMS messages:    {len(sms_subset)}")
    print(f"  Enron emails:    {len(enron_subset)}")
    print(f"  TOTAL combined:  {len(combined_spam)}")
    print(f"  Spam:            {combined_spam['label'].sum()}")
    print(f"  Ham:             {(combined_spam['label']==0).sum()}")

    results["Combined Spam Dataset"] = "✅ Success"

except Exception as e:
    print(f"❌ Failed to merge: {e}")
    results["Combined Spam Dataset"] = f"❌ Failed: {e}"

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{'='*60}")
print("PREPROCESSING SUMMARY")
print(f"{'='*60}")

for name, status in results.items():
    print(f"  {status}  {name}")

print(f"\n{'='*60}")
print("OUTPUT FILES CREATED:")
print(f"{'='*60}")

output_files = [
    ("../data/processed/sms_spam_processed.csv",     "SMS spam features"),
    ("../data/processed/sms_spam_tfidf.csv",         "SMS spam TF-IDF features"),
    ("../data/processed/malmem_processed.csv",        "MalMem malware features"),
    ("../data/processed/malmem_scaler.pkl",           "MalMem scaler (for predictions)"),
    ("../data/processed/enron_processed.csv",         "Enron email features"),
    ("../data/processed/combined_spam_processed.csv", "Combined SMS + Enron spam"),
]

for filepath, description in output_files:
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  ✓ {filepath:<50} ({size_kb:.0f} KB) - {description}")
    else:
        print(f"  ✗ {filepath:<50} NOT FOUND - {description}")

print("\n🎉 All preprocessing complete! Ready for Step 3: Model Training")
print("\nNEXT STEP: Run the model training scripts in /models/")
