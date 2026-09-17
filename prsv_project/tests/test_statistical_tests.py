import numpy as np

from evaluation.statistical_tests import calibration_report, mcnemar_test


def test_mcnemar_identical_predictions_not_significant() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred_a = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred_b = np.array([0, 1, 0, 1, 1, 0, 1, 0])

    result = mcnemar_test(y_true, y_pred_a, y_pred_b)

    assert result.significant_at_0_05 is False
    assert result.p_value == 1.0


def test_mcnemar_detects_large_discordance() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)

    # Model A always correct, model B always wrong -> maximally discordant.
    y_pred_a = y_true.copy()
    y_pred_b = 1 - y_true

    result = mcnemar_test(y_true, y_pred_a, y_pred_b)

    assert result.significant_at_0_05 is True
    assert result.p_value < 0.05


def test_calibration_report_perfect_calibration() -> None:
    rng = np.random.default_rng(1)
    y_prob = rng.uniform(0, 1, size=500)
    y_true = (rng.uniform(0, 1, size=500) < y_prob).astype(int)

    report = calibration_report(y_true, y_prob, output_dir=None)

    assert 0.0 <= report.brier_score <= 1.0
    assert len(report.bin_true_frequencies) == len(report.bin_predicted_probabilities)
