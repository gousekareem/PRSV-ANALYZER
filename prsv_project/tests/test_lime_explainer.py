from app.config import settings
from xai.lime_explainer import explain_with_lime


def test_lime_explainer_never_raises_regardless_of_availability() -> None:
    """
    On a fresh checkout with no trained model/background artifacts (or
    without the optional `lime` package installed), explain_with_lime must
    return None rather than raising - same best-effort contract as
    ml/shap_explainer.explain_prediction.
    """
    feature_vector = [0.4, 0.3, 0.5, 0.6, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.5, 0.6]

    result = explain_with_lime(feature_vector, settings)

    assert result is None or isinstance(result, dict)
