import numpy as np
from typing import Dict, Any, List

def extract_ecg_clinical_features(ecg_samples: List[float], hr: float = 72.0, fs: int = 250) -> Dict[str, Any]:
    """
    Extracts clinical electrocardiological features relevant to ischemia:
    - ST-segment elevation (mm)
    - ST-segment depression (mm)
    - T-wave inversion (boolean/depth)
    - Hyperacute T-wave (boolean)
    - Wellens' sign morphology
    - R-peak amplitude & variability
    """
    arr = np.array(ecg_samples, dtype=float)
    n = len(arr)
    if n < 250:
        return {
            "heart_rate_bpm": hr,
            "st_elevation_mm": 0.0,
            "st_depression_mm": 0.0,
            "t_wave_morphology": "indeterminate",
            "hyperacute_t_wave": False,
            "wellens_syndrome": "none",
            "r_amplitude": 0.0,
            "downsampled_ecg": list(arr[::10])
        }

    # Normalize roughly assuming 100 ADC counts ~= 1 mm (0.1 mV standard ECG calibration)
    SCALE_COUNT_TO_MM = 100.0

    # Isoelectric baseline: approximate via moving median or lower quartile
    baseline = float(np.median(arr))

    # Detect R-peaks with refractory period 300ms (75 samples at 250Hz)
    threshold = baseline + np.std(arr) * 1.2
    peaks = []
    for i in range(1, n - 1):
        if arr[i] > threshold and arr[i] > arr[i - 1] and arr[i] >= arr[i + 1]:
            if not peaks or (i - peaks[-1]) > 75:
                peaks.append(i)

    st_elevations = []
    st_depressions = []
    t_amplitudes = []
    biphasic_count = 0
    inverted_count = 0
    r_heights = []

    for p in peaks:
        # Check that we have enough samples before and after R
        if p < 20 or (p + 85) >= n:
            continue

        r_peak_val = arr[p]
        iso_val = float(np.median(arr[max(0, p - 25):max(0, p - 10)])) # PR segment baseline
        r_height = r_peak_val - iso_val
        r_heights.append(r_height)

        # J-point and ST segment: roughly 40ms to 80ms post-R peak (10 to 20 samples at 250Hz)
        st_val = float(np.mean(arr[p + 15:p + 25]))
        st_dev_mm = (st_val - iso_val) / SCALE_COUNT_TO_MM

        if st_dev_mm > 0.05:
            st_elevations.append(st_dev_mm)
        elif st_dev_mm < -0.05:
            st_depressions.append(abs(st_dev_mm))

        # T-wave segment: roughly 150ms to 320ms post-R peak (35 to 80 samples at 250Hz)
        t_segment = arr[p + 35:min(n, p + 80)] - iso_val
        if len(t_segment) > 10:
            t_max = float(np.max(t_segment))
            t_min = float(np.min(t_segment))
            
            # Check for inverted T-wave
            if abs(t_min) > abs(t_max) and t_min < -30.0:
                inverted_count += 1
                t_amplitudes.append(t_min / SCALE_COUNT_TO_MM)
            else:
                t_amplitudes.append(t_max / SCALE_COUNT_TO_MM)

            # Check for biphasic T-wave (Wellens' Pattern A: initial positive then deep negative)
            half = len(t_segment) // 2
            first_half_max = float(np.max(t_segment[:half]))
            second_half_min = float(np.min(t_segment[half:]))
            if first_half_max > 40.0 and second_half_min < -40.0:
                biphasic_count += 1

    # Summarize across beats
    mean_st_elev = round(float(np.mean(st_elevations)), 2) if st_elevations else 0.0
    mean_st_depr = round(float(np.mean(st_depressions)), 2) if st_depressions else 0.0
    mean_r_height = float(np.mean(r_heights)) if r_heights else 1000.0

    # Hyperacute T-wave check: T-wave amplitude > 70% of R-wave height
    is_hyperacute = False
    if t_amplitudes:
        max_t = max(t_amplitudes) * SCALE_COUNT_TO_MM
        if max_t > 0.70 * mean_r_height and max_t > 300.0:
            is_hyperacute = True

    # T-wave inversion check
    is_t_inverted = (inverted_count > len(peaks) * 0.4) if peaks else False

    # Wellens' syndrome check
    wellens_pattern = "none"
    if biphasic_count >= 2:
        wellens_pattern = "pattern_A_biphasic"
    elif is_t_inverted and mean_st_depr < 0.8:
        wellens_pattern = "pattern_B_deeply_inverted"

    t_morph = "normal upright"
    if is_hyperacute:
        t_morph = "hyperacute tall"
    elif is_t_inverted:
        t_morph = "inverted"
    elif wellens_pattern != "none":
        t_morph = f"wellens {wellens_pattern}"

    # Downsampled representative slice (take 1 every 25 samples = 50 samples for 5 seconds)
    downsampled = [round(float(v), 1) for v in arr[::25]]

    return {
        "heart_rate_bpm": round(float(hr), 1),
        "st_elevation_mm": mean_st_elev,
        "st_depression_mm": mean_st_depr,
        "t_wave_morphology": t_morph,
        "hyperacute_t_wave": is_hyperacute,
        "t_wave_inversion": is_t_inverted,
        "wellens_syndrome": wellens_pattern,
        "r_peak_count": len(peaks),
        "baseline_adc": round(baseline, 1),
        "downsampled_ecg_sample": downsampled
    }
