import pytest
from backend.services.model_loader import load_models
from backend.services.spam_service import predict_single, predict_batch, clean_text


@pytest.fixture(scope="module")
def registry():
    return load_models()


@pytest.mark.parametrize("model_name", ["nb_spam", "logistic_regression_spam"])
def test_spam_text_classified_as_spam(registry, model_name):
    result = predict_single("Congratulations! You WON a FREE iPhone! Call now to claim!", model_name, registry)
    assert result.label == "SPAM"
    assert result.spam_prob > 0.5


@pytest.mark.parametrize("model_name", ["nb_spam", "logistic_regression_spam"])
def test_ham_text_classified_as_ham(registry, model_name):
    result = predict_single("Are you coming to lunch today?", model_name, registry)
    assert result.label == "HAM"


def test_rf_spam_returns_valid_result(registry):
    # RF uses only message_length + word_count — no content semantics, so only check structure
    result = predict_single("Congratulations! You WON a FREE iPhone! Call now to claim!", "rf_spam", registry)
    assert result.label in ("SPAM", "HAM")
    assert 0.0 <= result.spam_prob <= 1.0
    assert 0.0 <= result.ham_prob <= 1.0
    assert result.confidence == max(result.spam_prob, result.ham_prob)


def test_unknown_model_raises_value_error(registry):
    with pytest.raises(ValueError, match="Unknown model"):
        predict_single("hello world", "bad_model", registry)


def test_predict_batch_empty_returns_empty(registry):
    assert predict_batch([], "rf_spam", registry) == []


def test_predict_batch_count(registry):
    results = predict_batch(["Win a prize!", "Hello there", "Free money"], "rf_spam", registry)
    assert len(results) == 3


def test_clean_text_removes_urls():
    assert "http" not in clean_text("Visit http://win.com for prizes")


def test_clean_text_removes_emails():
    assert "@" not in clean_text("Email bob@example.com for details")


def test_confidence_is_max_prob(registry):
    result = predict_single("hello world how are you", "rf_spam", registry)
    assert result.confidence == max(result.spam_prob, result.ham_prob)
