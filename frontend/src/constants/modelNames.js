// G1 — shared display-name map for model keys, so raw keys (rf_spam, lr_spam, …)
// never leak into the UI. Used across Dashboard, Spam Detector and Model Analytics.
export const MODEL_LABELS = {
  rf_spam: 'Random Forest',
  nb_spam: 'Naive Bayes',
  lr_spam: 'Logistic Regression',
  logistic_regression_spam: 'Logistic Regression',
  svm_malware: 'SVM (Malware)',
  kmeans_malware: 'K-Means (Malware)',
  dbscan_malware: 'DBSCAN (Malware)',
}

// Falls back to the raw key so unknown models still render something.
export const modelLabel = (key) => MODEL_LABELS[key] ?? key
