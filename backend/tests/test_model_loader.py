import pytest
from backend.services.model_loader import load_models
from sklearn.svm import SVC


@pytest.fixture(scope="module")
def registry():
    return load_models()


def test_all_model_keys_present(registry):
    for key in ["rf_spam", "nb_spam", "lr_spam", "svm_malware", "kmeans_malware", "dbscan_malware"]:
        assert key in registry


def test_tfidf_has_500_features(registry):
    assert len(registry["spam_tfidf"].vocabulary_) == 500


def test_malmem_feature_cols_excludes_labels(registry):
    cols = registry["malmem_feature_cols"]
    assert "binary_label" not in cols
    assert "category_encoded" not in cols
    assert "category_name" not in cols
    assert len(cols) > 0


def test_svm_is_raw_svc(registry):
    assert isinstance(registry["svm_malware"], SVC)


def test_missing_pkl_raises_runtime_error():
    with pytest.raises(RuntimeError, match="Missing required file"):
        load_models(models_dir="nonexistent/path")
