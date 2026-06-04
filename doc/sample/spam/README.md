# Spam Detector — Sample Files

Ready-to-upload samples for the **Spam Detector** page (`/spam`, *Batch Upload* tab).

**Accepted formats** (see `backend/routers/spam.py`):
- **`.txt`** — one message per line.
- **`.csv`** — must contain a **`message`** column (other columns are ignored).

You can also paste any single line into the *Single Message* tab.

| File | Format | Rows | Contents |
| ---- | ------ | ---- | -------- |
| `spam_samples.txt` | txt | 10 | Realistic spam (prize scams, phishing, fake parcels, dodgy offers) |
| `ham_samples.txt` | txt | 10 | Normal everyday messages (the "ham" class) |
| `mixed_samples.csv` | csv | 16 | 8 spam + 8 ham mixed — the best all-round demo |
| `rf_spam_samples.csv` | csv | 8 | Spam tuned for the **Random Forest** model (see note below) |

## Detection results per model

These counts come from running each file through the **real backend models**
(`backend/services/spam_service.py`):

| File | Random Forest | Naive Bayes | Logistic Regression |
| ---- | ------------- | ----------- | ------------------- |
| `spam_samples.txt` (10 spam) | 1 / 10 | **7 / 10** | **7 / 10** |
| `ham_samples.txt` (10 ham) | 0 / 10 ✓ | 1 / 10 | 0 / 10 ✓ |
| `rf_spam_samples.csv` (8 spam) | **8 / 8** | 7 / 8 | 7 / 8 |

> **Use Naive Bayes or Logistic Regression for content-based spam detection** — they read the
> message text (TF-IDF) and are the strong models here.

## ⚠️ About the Random Forest model

The trained Random Forest spam model only uses **two numeric features** —
`message_length` and `word_count` — it does **not** read the words at all
(confirmed in `spam_service.py::_predict_rf`, where `feature_cols == ['message_length', 'word_count']`).

Consequences:
- RF essentially classifies by **length**. It flags SPAM mainly for longer promotional-style
  messages (~24-28 words / ~130-155 characters) and calls most short messages HAM regardless of
  how "spammy" the words are.
- That's why a short, obvious scam can read as HAM on RF but SPAM on NB/LR.

`rf_spam_samples.csv` therefore contains **real spam messages from the SMS dataset that the RF model
actually flags** (RF spam-probability 61-95%), so you can see RF return SPAM. For everything else,
prefer **Naive Bayes** or **Logistic Regression**.

## How to use

1. Start the backend (`:8000`) and frontend (`:5173`).
2. Open **Spam Detector → Batch Upload**.
3. Upload a file, pick a model (try **Naive Bayes**), and click **Analyze** to see the spam/ham
   split, the probability histogram, the ham-vs-spam donut, and the per-message results table.
