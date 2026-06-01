import pickle
import logging
from typing import TypedDict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class ModelRegistry(TypedDict):
    rf_spam:             dict
    nb_spam:             dict
    lr_spam:             dict
    svm_malware:         object   # raw SVC
    kmeans_malware:      dict
    dbscan_malware:      dict
    spam_tfidf:          TfidfVectorizer
    malmem_feature_cols: list


def load_models(
    models_dir: str = "outputs/models",
    processed_dir: str = "data/processed",
    spam_corpus_path: str = "data/processed/sms_spam_processed.csv",
    malmem_path: str = "data/processed/malmem_processed.csv",
) -> ModelRegistry:
    def _load_pkl(path: str):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Missing required file: {path}")

    registry = {}
    registry["rf_spam"]        = _load_pkl(f"{models_dir}/rf_spam.pkl")
    registry["nb_spam"]        = _load_pkl(f"{models_dir}/nb_spam.pkl")
    registry["lr_spam"]        = _load_pkl(f"{models_dir}/logistic_regression_spam.pkl")
    registry["svm_malware"]    = _load_pkl(f"{models_dir}/svm_malware.pkl")
    registry["kmeans_malware"] = _load_pkl(f"{models_dir}/kmeans_malware.pkl")
    registry["dbscan_malware"] = _load_pkl(f"{models_dir}/dbscan_malware.pkl")

    try:
        corpus_df = pd.read_csv(spam_corpus_path)
    except FileNotFoundError:
        raise RuntimeError(f"Missing required file: {spam_corpus_path}")
    tfidf = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    tfidf.fit(corpus_df["cleaned_message"].fillna("").astype(str))
    registry["spam_tfidf"] = tfidf

    try:
        malmem_df = pd.read_csv(malmem_path)
    except FileNotFoundError:
        raise RuntimeError(f"Missing required file: {malmem_path}")
    drop_cols = [c for c in ["binary_label", "category_encoded", "category_name"] if c in malmem_df.columns]
    registry["malmem_feature_cols"] = [c for c in malmem_df.columns if c not in drop_cols]

    logger.info("ModelRegistry loaded: %d keys", len(registry))
    return registry
