# Driver Monitoring System - Calibration Quality Safeguard Fix

## Problem Solved
Fixed two critical issues:
1. **REAL-CAMERA runtime crash** - Missing import of `hands_off_wheel_alert` in main.py
2. **Poor calibration quality handling** - Invalid EAR samples (like 0.068) were being used to generate unusable thresholds

## Root Cause Analysis
The EAR value of 0.068 observed during calibration is abnormally low:
- Normal open-eye EAR range: 0.25-0.30
- Eyes-closed threshold: Typically < 0.20
- Value of 0.068 suggests either:
  1. Eyes were mostly closed during baseline collection
  2. Face landmark detection failure (detecting incorrect landmarks)
  3. Systematic error in eye landmark localization

## Solution Implemented

### 1. Crash Fix Verified
- `hands_off_wheel_alert` function exists in `audio_alert.py` (lines 192-196)
- Properly imported in `main.py` (line 20)
- Correctly called in alert logic (line 418)
- All modules compile successfully

### 2. Calibration Quality Safeguard
Updated `calibration.py` with:

**EAR Sample Validation:**
- Reject samples < 0.08 (eyes mostly closed or detection failure)
- Reject samples > 0.45 (landmark detection failure)
- Reject non-finite values

**Independent EAR/MAR Calibration:**
- Each modality calibrated separately
- One can fail while the other succeeds
- Overall `calibrated` flag = True if at least one threshold was calibrated

**Quality Assessment:**
- `good`: ≥50% valid samples, <50% rejection rate
- `marginal`: ≥50% valid samples, ≥50% rejection rate  
- `poor`: <50% valid samples or all samples rejected

**Fallback Behavior:**
- Poor EAR quality → EAR threshold falls back to default (0.21)
- MAR calibration continues independently when possible
- Clear reasoning in calibration result

**Enhanced CalibrationResult:**
- `ear_valid_samples`: Count of samples passing validation
- `ear_rejection_rate`: Percentage of samples rejected
- `calibration_quality`: "good"/"marginal"/"poor"

## Test Results
All 9 calibration quality tests pass:
```
test_is_valid_ear_sample              PASSED
test_valid_ear_calibration            PASSED
test_very_low_ear_samples_rejected    PASSED
test_insufficient_valid_ear_samples   PASSED
test_successful_fallback              PASSED
test_valid_mar_calibration_independent PASSED
test_mar_calibration_with_poor_ear    PASSED
test_both_ear_mar_poor_fallback       PASSED
test_calibration_result_attributes    PASSED
```

## System Behavior Examples

**Case 1: Valid calibration (normal operation)**
- EAR samples: [0.29, 0.30, 0.28, 0.31, 0.29, 0.30, 0.28, 0.30, 0.29, 0.30]
- Result: Calibrated EAR≈0.22, MAR≈0.20, quality="good", calibrated=True

**Case 2: Poor EAR quality (original issue)**
- EAR samples: [0.068, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00, 0.07, 0.06, 0.05] 
- MAR samples: [0.20, 0.22, 0.18, 0.21, 0.20, 0.23, 0.19, 0.21, 0.20, 0.22]
- Result: EAR=0.21 (default), MAR≈0.25 (calibrated), quality="poor", calibrated=True

**Case 3: Complete detection failure**
- EAR samples: [0.068, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00, 0.07, 0.06, 0.05]
- MAR samples: [0.05, 0.04, 0.03, 0.02, 0.01, 0.00, 0.06, 0.05, 0.04, 0.03]
- Result: EAR=0.21, MAR=0.55 (both defaults), quality="poor", calibrated=False

## Verification Completed
✅ **Import verification**: `hands_off_wheel_alert` properly imported
✅ **Compile verification**: All Python modules compile without errors  
✅ **Unit test verification**: 9/9 calibration quality tests pass
✅ **Logic verification**: Independent EAR/MAR calibration works correctly
✅ **Fallback verification**: Proper behavior when calibration quality poor
✅ **Crash simulation**: No crash when hand-off-wheel alerts triggered

## Recommendation for User
The system will now:
1. Never crash due to missing hand-off-wheel alert function
2. Automatically detect and reject poor calibration samples
3. Fall back to safe defaults when calibration quality is insufficient
4. Continue to calibrate the unaffected modality (EAR or MAR) when possible
5. Provide clear calibration quality feedback in logs

No manual recalibration is needed - the safeguard handles poor quality inputs automatically.

The EAR=0.068 value was correctly identified as invalid and will be rejected during calibration, preventing the generation of unusable thresholds.