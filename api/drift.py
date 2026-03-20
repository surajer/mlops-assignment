import numpy as np

# Baseline (training stats)
baseline_mean = np.array([5.8, 3.0, 3.7, 1.1])
drift_threshold = 1.0


def detect_drift(input_data: np.ndarray) -> bool:
    """
    Detects drift by comparing input mean with baseline mean.
    Returns True if drift is detected.
    """
    input_mean = np.mean(input_data, axis=0)
    diff = np.abs(input_mean - baseline_mean)

    return bool(np.any(diff > drift_threshold))