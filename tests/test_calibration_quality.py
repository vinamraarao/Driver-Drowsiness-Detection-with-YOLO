"""Tests for calibration quality safeguard."""
import math
import sys
from unittest.mock import Mock

# Add the project root to the path
sys.path.insert(0, '.')

from calibration import CalibrationResult, _is_valid_ear_sample, derive_thresholds
import config


def test_is_valid_ear_sample():
    """Test EAR sample validation logic."""
    # Valid samples
    assert _is_valid_ear_sample(0.25) == True
    assert _is_valid_ear_sample(0.30) == True
    assert _is_valid_ear_sample(0.21) == True  # Default threshold

    # Invalid samples - too low
    assert _is_valid_ear_sample(0.05) == False
    assert _is_valid_ear_sample(0.068) == False  # The problematic value from the issue
    assert _is_valid_ear_sample(0.07) == False
    assert _is_valid_ear_sample(0.08) == True   # Boundary

    # Invalid samples - too high
    assert _is_valid_ear_sample(0.46) == False
    assert _is_valid_ear_sample(0.50) == False
    assert _is_valid_ear_sample(0.45) == True   # Boundary

    # Invalid samples - non-finite
    assert _is_valid_ear_sample(float('inf')) == False
    assert _is_valid_ear_sample(float('-inf')) == False
    assert _is_valid_ear_sample(float('nan')) == False
    assert _is_valid_ear_sample(None) == False
    assert _is_valid_ear_sample("invalid") == False


def test_valid_ear_calibration():
    """Test that valid EAR samples produce calibrated threshold."""
    # Create samples that should produce a reasonable EAR threshold
    # Normal open-eye EAR values around 0.28-0.32
    ear_samples = [0.29, 0.30, 0.28, 0.31, 0.29, 0.30, 0.28, 0.30, 0.29, 0.30]
    mar_samples = [0.20, 0.22, 0.18, 0.21, 0.20, 0.23, 0.19, 0.21, 0.20, 0.22]  # Normal MAR

    result = derive_thresholds(ear_samples, mar_samples)

    # Should use calibrated thresholds (not fall back to defaults)
    assert result.calibrated == True
    assert result.ear_threshold != config.EAR_THRESHOLD or result.mar_threshold != config.MAR_THRESHOLD
    assert result.calibration_quality in ["good", "marginal"]
    assert result.ear_valid_samples == len(ear_samples)  # All samples should be valid
    assert result.ear_rejection_rate == 0.0


def test_very_low_ear_samples_rejected():
    """Test that very low EAR samples are rejected during calibration."""
    # Mix of valid and invalid (very low) EAR samples
    ear_samples = [0.29, 0.068, 0.30, 0.05, 0.28, 0.04, 0.31, 0.29, 0.07, 0.30]  # 4 invalid low samples
    mar_samples = [0.20, 0.22, 0.18, 0.21, 0.20, 0.23, 0.19, 0.21, 0.20, 0.22]  # Normal MAR

    result = derive_thresholds(ear_samples, mar_samples)

    # Should reject the invalid samples
    assert result.ear_valid_samples == 6  # 6 valid samples out of 10
    assert result.ear_rejection_rate == 0.4  # 40% rejection rate

    # With 6 valid samples (minimum required is 5) and 40% rejection, should be marginal or good
    assert result.calibration_quality in ["marginal", "good"]

    # Should still calibrate since we have enough valid samples and reasonable result
    # The exact behavior depends on the calculated threshold, but it should not automatically fall back


def test_insufficient_valid_ear_samples():
    """Test fallback when insufficient valid EAR samples."""
    # Too many invalid samples
    ear_samples = [0.068, 0.05, 0.04, 0.07, 0.06, 0.08, 0.09, 0.05, 0.04, 0.06]  # Mostly invalid
    mar_samples = [0.20, 0.22, 0.18, 0.21, 0.20, 0.23, 0.19, 0.21, 0.20, 0.22]  # Normal MAR

    result = derive_thresholds(ear_samples, mar_samples)

    # Should have very few valid samples
    assert result.ear_valid_samples <= 2  # At most 2 valid samples (0.08 and 0.09)
    assert result.ear_rejection_rate >= 0.8  # At least 80% rejection rate

    # Should fall back to safe defaults for EAR due to insufficient valid samples
    # MAR should still be calibrated independently (requirement #7)
    assert result.ear_threshold == config.EAR_THRESHOLD
    assert result.calibration_quality == "poor"
    assert "insufficient" in result.reason.lower() or "poor" in result.reason.lower()


def test_successful_fallback():
    """Test that fallback to defaults works correctly when EAR samples are invalid."""
    # No valid EAR samples at all
    ear_samples = [0.068, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00, 0.07, 0.06, 0.05]
    mar_samples = [0.20, 0.22, 0.18, 0.21, 0.20, 0.23, 0.19, 0.21, 0.20, 0.22]

    result = derive_thresholds(ear_samples, mar_samples)

    # Should fall back to defaults for EAR
    assert result.ear_threshold == config.EAR_THRESHOLD
    assert result.ear_valid_samples == 0
    assert result.ear_rejection_rate == 1.0
    assert result.calibration_quality == "poor"
    # MAR should still be calibrated independently when possible
    assert result.mar_threshold != config.MAR_THRESHOLD  # MAR calibrated from valid samples
    assert result.calibrated == True  # Overall calibrated=True because MAR was calibrated
    assert "MAR calibrated" in result.reason
    assert "all samples rejected" in result.reason


def test_valid_mar_calibration_independent():
    """Test that MAR calibration works independently of EAR quality."""
    # Valid EAR samples but unusual MAR samples that should still calibrate MAR
    ear_samples = [0.29, 0.30, 0.28, 0.31, 0.29, 0.30, 0.28, 0.30, 0.29, 0.30]  # All valid
    # MAR samples that indicate a person with high baseline mouth aspect ratio
    mar_samples = [0.40, 0.42, 0.38, 0.41, 0.40, 0.43, 0.39, 0.41, 0.40, 0.42]  # Higher than normal

    result = derive_thresholds(ear_samples, mar_samples)

    # Should calibrate since we have good EAR samples
    assert result.calibrated == True
    assert result.ear_valid_samples == 10
    assert result.ear_rejection_rate == 0.0

    # MAR threshold should be adjusted based on the higher baseline
    assert result.mar_threshold != config.MAR_THRESHOLD  # Should be calibrated

    # Both thresholds should be within their respective bounds
    assert config.EAR_CALIB_MIN <= result.ear_threshold <= config.EAR_CALIB_MAX
    assert config.MAR_CALIB_MIN <= result.mar_threshold <= config.MAR_CALIB_MAX


def test_mar_calibration_with_poor_ear():
    """Test that MAR calibrates independently when EAR calibration is poor."""
    # Poor EAR samples but valid MAR samples
    ear_samples = [0.068, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00, 0.07, 0.06, 0.05]  # All invalid
    mar_samples = [0.40, 0.42, 0.38, 0.41, 0.40, 0.43, 0.39, 0.41, 0.40, 0.42]  # Valid MAR samples

    result = derive_thresholds(ear_samples, mar_samples)

    # EAR should fall back to defaults
    assert result.ear_threshold == config.EAR_THRESHOLD
    assert result.ear_valid_samples == 0
    assert result.ear_rejection_rate == 1.0

    # But MAR should still be calibrated based on valid samples
    assert result.mar_threshold != config.MAR_THRESHOLD  # MAR should be calibrated
    assert result.calibrated == True  # Overall calibrated=True because MAR was calibrated
    assert "MAR calibrated" in result.reason
    assert "poor EAR calibration quality" in result.reason


def test_both_ear_mar_poor_fallback():
    """Test fallback to defaults when both EAR and MAR calibration are poor."""
    # Very poor samples for both
    ear_samples = [0.068, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00, 0.07, 0.06, 0.05]  # All invalid EAR
    mar_samples = [0.05, 0.04, 0.03, 0.02, 0.01, 0.00, 0.06, 0.05, 0.04, 0.03]  # All invalid MAR

    result = derive_thresholds(ear_samples, mar_samples)

    # Should fall back to defaults for both
    assert result.ear_threshold == config.EAR_THRESHOLD
    assert result.mar_threshold == config.MAR_THRESHOLD
    assert result.ear_valid_samples == 0
    assert result.calibrated == False  # Nothing was calibrated
    assert result.calibration_quality == "poor"
    assert "safe defaults" in result.reason.lower()


def test_calibration_result_attributes():
    """Test that CalibrationResult includes new quality attributes."""
    result = CalibrationResult(
        ear_threshold=0.22,
        mar_threshold=0.50,
        ear_baseline=0.28,
        mar_baseline=0.25,
        ear_samples=[0.28, 0.29, 0.27],
        mar_samples=[0.25, 0.26, 0.24],
        calibrated=True,
        reason="test calibration",
        ear_valid_samples=3,
        ear_rejection_rate=0.0,
        calibration_quality="good"
    )

    assert result.ear_threshold == 0.22
    assert result.mar_threshold == 0.50
    assert result.ear_valid_samples == 3
    assert result.ear_rejection_rate == 0.0
    assert result.calibration_quality == "good"


if __name__ == "__main__":
    # Run tests manually if called directly
    test_is_valid_ear_sample()
    print("OK: test_is_valid_ear_sample passed")

    test_valid_ear_calibration()
    print("OK: test_valid_ear_calibration passed")

    test_very_low_ear_samples_rejected()
    print("OK: test_very_low_ear_samples_rejected passed")

    test_insufficient_valid_ear_samples()
    print("OK: test_insufficient_valid_ear_samples passed")

    test_successful_fallback()
    print("OK: test_successful_fallback passed")

    test_valid_mar_calibration_independent()
    print("OK: test_valid_mar_calibration_independent passed")

    test_mar_calibration_with_poor_ear()
    print("OK: test_mar_calibration_with_poor_ear passed")

    test_both_ear_mar_poor_fallback()
    print("OK: test_both_ear_mar_poor_fallback passed")

    test_calibration_result_attributes()
    print("OK: test_calibration_result_attributes passed")

    print("\nAll calibration quality tests passed!")