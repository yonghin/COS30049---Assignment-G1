import pytest
from backend.services.model_loader import load_models
from backend.services.analytics_service import initialize


@pytest.fixture(scope="module")
def cache():
    return initialize(load_models())


def test_all_four_models_populated(cache):
    for key in ["rf_spam", "nb_spam", "logistic_regression_spam", "svm_malware"]:
        assert cache[key] is not None


def test_confusion_matrix_shape(cache):
    cm = cache["rf_spam"]["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2


def test_rf_feature_importance_non_empty(cache):
    fi = cache["rf_spam"]["feature_importance"]
    assert fi is not None and len(fi) > 0


def test_nb_feature_importance_is_none(cache):
    assert cache["nb_spam"]["feature_importance"] is None


def test_auc_in_range(cache):
    for key in ["rf_spam", "nb_spam", "logistic_regression_spam", "svm_malware"]:
        auc = cache[key]["roc"]["auc"]
        assert 0.0 <= auc <= 1.0
