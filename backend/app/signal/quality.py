import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class QualityAssessment:
    quality_score: float                # 0.00 to 1.00
    status: str                         # "CLEAN", "DEGRADED", "REJECTED"
    is_rejected: bool
    rejection_reason: Optional[str]
    detected_artifacts: List[str]
    warnings: List[str]
    metrics: Dict[str, float]


class SignalQualityAssessor:
    """
    Comprehensive multi-metric ECG Signal Quality Index (SQI) and Artifact Evaluator.
    Computes statistical, spectral, and hardware indicators to output measurable quality scores.
    """

    def __init__(
        self,
        sampling_rate: int = 250,
        clean_threshold: float = 0.78,
        degraded_threshold: float = 0.48,
        motion_threshold: float = 0.18,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.clean_threshold = clean_threshold
        self.degraded_threshold = degraded_threshold
        self.motion_threshold = motion_threshold

    def evaluate_chunk(
        self,
        raw_ecg: List[float],
        cleaned_ecg: List[float],
        motion_samples: Optional[List[float]] = None,
        lead_off_flags: Optional[List[int]] = None,
        gap_detected: bool = False,
        gap_max_ms: float = 0.0,
    ) -> QualityAssessment:
        n = len(raw_ecg)
        if n < 50:
            return QualityAssessment(
                quality_score=0.0,
                status="REJECTED",
                is_rejected=True,
                rejection_reason="Insufficient data length (< 50 samples)",
                detected_artifacts=["DATA_TOO_SHORT"],
                warnings=["Window has insufficient samples."],
                metrics={"sample_count": float(n)},
            )

        raw = np.array(raw_ecg, dtype=np.float64)
        clean = np.array(cleaned_ecg, dtype=np.float64) if (cleaned_ecg is not None and len(cleaned_ecg) > 0) else raw

        detected_artifacts = []
        warnings = []

        # -------------------------------------------------------------
        # 1. Hardware & Saturation Indicators
        # -------------------------------------------------------------
        # Railing / Clipping check (ADC counts <= 50 or >= 4045 for 12-bit ADC)
        clipped_count = np.sum((raw <= 50) | (raw >= 4045))
        clip_ratio = float(clipped_count / n)
        if clip_ratio > 0.05:
            detected_artifacts.append("CLIPPING_SATURATION")
            warnings.append(f"ADC clipping/saturation on {clip_ratio * 100:.1f}% of samples.")

        # Flatline check (consecutive samples with near-zero diff)
        diffs = np.abs(np.diff(raw))
        zero_diffs = np.sum(diffs < 2.0)
        flatline_ratio = float(zero_diffs / (n - 1))
        signal_std = float(np.std(raw))

        if signal_std < 12.0 or flatline_ratio > 0.60:
            detected_artifacts.append("FLATLINE_DISCONNECT")
            warnings.append("Flatline / detached electrode detected.")

        # Hardware Lead-Off pins
        lead_off_ratio = 0.0
        if lead_off_flags and len(lead_off_flags) > 0:
            lo_count = sum(1 for x in lead_off_flags if x > 0)
            lead_off_ratio = float(lo_count / len(lead_off_flags))
            if lead_off_ratio > 0.10:
                detected_artifacts.append("HARDWARE_LEAD_OFF")
                warnings.append(f"AD8232 lead-off detected on {lead_off_ratio * 100:.1f}% of window.")

        # Motion / Accelerometer check
        max_motion_dev = 0.0
        motion_noise_ratio = 0.0
        if motion_samples and len(motion_samples) > 0:
            m_arr = np.array(motion_samples, dtype=np.float64)
            devs = np.abs(m_arr - 1.0)
            max_motion_dev = float(np.max(devs))
            motion_noisy = np.sum(devs > self.motion_threshold)
            motion_noise_ratio = float(motion_noisy / len(motion_samples))
            if motion_noise_ratio > 0.30 or max_motion_dev > 0.40:
                detected_artifacts.append("EXCESSIVE_MOTION")
                warnings.append(f"Accelerometer motion artifact ({motion_noise_ratio * 100:.1f}% contaminated, peak dev {max_motion_dev:.2f}g).")

        # Data gap check
        if gap_detected or gap_max_ms > 40.0:
            detected_artifacts.append("DATA_GAP")
            warnings.append(f"Critical data transmission gap detected ({gap_max_ms:.1f}ms).")

        # -------------------------------------------------------------
        # 2. Statistical Signal Quality Indices (SQI)
        # -------------------------------------------------------------
        # Kurtosis (kSQI): clean QRS produces high kurtosis (> 5.0)
        try:
            k_sqi = float(kurtosis(clean, fisher=True)) # Gaussian = 0.0, ECG > 3.0
            # Normalize to 0..1 range where normal ECG (kurtosis 3 to 15) scores high
            k_norm = min(1.0, max(0.0, (k_sqi + 1.0) / 10.0))
        except Exception:
            k_sqi = 0.0
            k_norm = 0.5

        # Skewness (sSQI)
        try:
            s_sqi = float(skew(clean))
        except Exception:
            s_sqi = 0.0

        # -------------------------------------------------------------
        # 3. Spectral Power SQIs (pSQI, basSQI, hfSQI)
        # -------------------------------------------------------------
        try:
            freqs, psd = welch(clean, fs=self.sampling_rate, nperseg=min(n, 256))
            total_power = float(np.sum(psd)) + 1e-12

            # QRS Band Power (5 to 15 Hz) vs total ECG Band (1 to 40 Hz)
            qrs_band = (freqs >= 5.0) & (freqs <= 15.0)
            ecg_band = (freqs >= 1.0) & (freqs <= 40.0)
            p_qrs = np.sum(psd[qrs_band])
            p_ecg = np.sum(psd[ecg_band]) + 1e-12
            p_sqi = float(p_qrs / p_ecg) # clean ECG typically 0.4 to 0.85

            # Baseline Wander Ratio (0 to 0.5 Hz vs total)
            bw_band = (freqs >= 0.0) & (freqs < 0.5)
            bas_sqi = float(np.sum(psd[bw_band]) / total_power)
            if bas_sqi > 0.45:
                detected_artifacts.append("HIGH_BASELINE_WANDER")
                warnings.append(f"High baseline wander spectral power ({bas_sqi * 100:.1f}% of total).")

            # High Frequency Noise Ratio (> 35 Hz vs total)
            hf_band = (freqs >= 35.0)
            hf_sqi = float(np.sum(psd[hf_band]) / total_power)
            if hf_sqi > 0.30:
                detected_artifacts.append("HIGH_FREQUENCY_EMG")
                warnings.append(f"High frequency muscle/electronic noise ({hf_sqi * 100:.1f}% of total).")

            # Estimated SNR (dB) in QRS band vs out-of-band noise
            noise_power = total_power - p_qrs
            snr_db = round(float(10.0 * np.log10(max(p_qrs, 1e-12) / max(noise_power, 1e-12))), 1)

        except Exception:
            p_sqi = 0.5
            bas_sqi = 0.2
            hf_sqi = 0.1
            snr_db = 10.0

        # -------------------------------------------------------------
        # 4. Composite Quality Score Calculation (0.0 to 1.0)
        # -------------------------------------------------------------
        # Base quality from spectral & statistical indicators
        base_score = 0.35 * min(1.0, p_sqi / 0.50) + 0.25 * k_norm + 0.20 * (1.0 - min(1.0, bas_sqi * 2.0)) + 0.20 * (1.0 - min(1.0, hf_sqi * 2.5))
        base_score = min(1.0, max(0.0, base_score))

        # Apply artifact penalties
        penalty = 0.0
        if "CLIPPING_SATURATION" in detected_artifacts:
            penalty += clip_ratio * 1.5
        if "FLATLINE_DISCONNECT" in detected_artifacts:
            penalty += 1.0
        if "HARDWARE_LEAD_OFF" in detected_artifacts:
            penalty += lead_off_ratio * 1.2
        if "EXCESSIVE_MOTION" in detected_artifacts:
            penalty += motion_noise_ratio * 0.7
        if "DATA_GAP" in detected_artifacts:
            penalty += 0.8
        if "HIGH_FREQUENCY_EMG" in detected_artifacts:
            penalty += hf_sqi * 0.5

        final_score = max(0.0, min(1.0, base_score - penalty))
        final_score = round(final_score, 3)

        # Determine Categorical Status
        is_fatal = any(a in ["FLATLINE_DISCONNECT", "HARDWARE_LEAD_OFF", "DATA_GAP"] for a in detected_artifacts) or (clip_ratio > 0.15)

        if is_fatal or final_score < self.degraded_threshold:
            status = "REJECTED"
            is_rejected = True
            rejection_reason = (
                "Flatline / Lead Detached" if "FLATLINE_DISCONNECT" in detected_artifacts
                else "Hardware Lead Off" if "HARDWARE_LEAD_OFF" in detected_artifacts
                else "Critical Data Gap" if "DATA_GAP" in detected_artifacts
                else f"ADC Clipping / Saturation ({clip_ratio * 100:.1f}%)" if clip_ratio > 0.15
                else f"Severe Noise / Artifact Distortion (Quality: {final_score:.2f})"
            )
        elif final_score >= self.clean_threshold:
            status = "CLEAN"
            is_rejected = False
            rejection_reason = None
        else:
            status = "DEGRADED"
            is_rejected = False
            rejection_reason = None

        metrics_dict = {
            "quality_score": final_score,
            "snr_db": snr_db,
            "p_sqi": round(p_sqi, 3),
            "k_sqi": round(k_sqi, 2),
            "bas_sqi": round(bas_sqi, 3),
            "hf_sqi": round(hf_sqi, 3),
            "clip_ratio": round(clip_ratio, 3),
            "flatline_ratio": round(flatline_ratio, 3),
            "motion_noise_ratio": round(motion_noise_ratio, 3),
            "signal_std": round(signal_std, 1),
        }

        return QualityAssessment(
            quality_score=final_score,
            status=status,
            is_rejected=is_rejected,
            rejection_reason=rejection_reason,
            detected_artifacts=detected_artifacts,
            warnings=warnings,
            metrics=metrics_dict,
        )
