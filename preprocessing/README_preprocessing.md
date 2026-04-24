# Spam & Malware Detection - Project Structure

## Folder Layout
```
spam_malware_project/
│
├── data/
│   ├── raw/                        ← Put your downloaded datasets HERE
│   │   ├── SMSSpamCollection       (from UCI)
│   │   ├── Obfuscated-MalMem2022.csv (from Kaggle)
│   │   └── enron_spam_data.csv     (from Kaggle)
│   │
│   └── processed/                  ← Cleaned data (auto-created)
│       ├── sms_spam_processed.csv
│       ├── sms_spam_tfidf.csv
│       ├── malmem_processed.csv
│       ├── enron_processed.csv
│       └── combined_spam_processed.csv
│
├── preprocessing/                  ← Data cleaning scripts (YOU ARE HERE)
│   ├── 00_run_all_preprocessing.py ← RUN THIS FIRST
│   ├── 01_preprocess_sms_spam.py
│   ├── 02_preprocess_malmem.py
│   └── 03_preprocess_enron.py
│
├── models/                         ← ML model training (Step 3)
│
└── outputs/
    └── visualizations/             ← Charts and graphs (auto-created)
```

## How to Run Preprocessing

### 1. Install required libraries
```bash
conda create -n spam_malware python=3.10
conda activate spam_malware
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Put raw data files in data/raw/

### 3. Run master preprocessing script
```bash
cd preprocessing
python 00_run_all_preprocessing.py
```
