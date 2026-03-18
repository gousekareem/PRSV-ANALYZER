from app.config import settings
from ml.infer_svm import predict_with_svm


def test_predict_with_svm_returns_valid_output() -> None:
    feature_vector = [0.4, 0.5, 0.3, 0.4, 0.2, 0.1, 0.5]
    result = predict_with_svm(feature_vector, settings)

    assert result.prediction in {"Healthy", "Diseased"}
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.feature_vector_used) == 7