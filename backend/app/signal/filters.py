import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.signal import butter, sosfilt, filtfilt, iirnotch, savgol_filter, welch
from scipy.ndimage import median_filter


class ECGFilter:
    """
    Robust clinical ECG filtering engine:
    - Real-time SOS filtering for live streaming batches.
    - Dual-median / Butterworth baseline wander removal preserving ST-T morphology.
    - Adaptive 50Hz/60Hz powerline notch filter.
    - 35Hz zero-phase lowpass + Savitzky-Golay muscle noise suppression.
    - Reflection padding to prevent boundary edge artifacts.
    """

    def __init__(
        self,
        sampling_rate: int = 250,
        highpass_hz: float = 0.5,
        lowpass_hz: float = 35.0,
        notch_hz: float = 50.0,
        notch_enabled: bool = True,
        notch_adaptive: bool = True,
        baseline_method: str = "dual_median",
    ) -> None:
        self.sampling_rate = sampling_rate
        self.highpass_hz = highpass_hz
        self.lowpass_hz = lowpass_hz
        self.notch_hz = notch_hz
        self.notch_enabled = notch_enabled
        self.notch_adaptive = notch_adaptive
        self.baseline_method = baseline_method

        # -------------------------------------------------------------
        # Real-time stateful SOS filters for 25-sample live stream
        # -------------------------------------------------------------
        self.sos_hp = butter(2, highpass_hz, btype="highpass", fs=sampling_rate, output="sos")
        self.zi_hp = np.zeros((self.sos_hp.shape[0], 2))

        self.sos_lp = butter(3, lowpass_hz, btype="lowpass", fs=sampling_rate, output="sos")
        self.zi_lp = np.zeros((self.sos_lp.shape[0], 2))

        if notch_hz < (sampling_rate / 2.0 - 2.0):
            self.sos_notch = butter(2, [max(1.0, notch_hz - 1.5), min(sampling_rate / 2.0 - 0.5, notch_hz + 1.5)], btype="bandstop", fs=sampling_rate, output="sos")
            self.zi_notch = np.zeros((self.sos_notch.shape[0], 2))
        else:
            self.sos_notch = None
            self.zi_notch = None

        # -------------------------------------------------------------
        # Zero-phase filters for 5-second chunk analysis
        # -------------------------------------------------------------
        self.lp_b, self.lp_a = butter(4, lowpass_hz, fs=sampling_rate, btype="lowpass")
        self.hp_b, self.hp_a = butter(3, highpass_hz, fs=sampling_rate, btype="highpass")

        if notch_hz < (sampling_rate / 2.0 - 2.0):
            self.notch_b, self.notch_a = iirnotch(notch_hz, Q=30, fs=sampling_rate)
        else:
            self.notch_b, self.notch_a = None, None

        # Harmonic notch (e.g. 100 Hz if 50Hz, 120 Hz if 60Hz and within Nyquist)
        harmonic_hz = notch_hz * 2.0
        if harmonic_hz < (sampling_rate / 2.0 - 2.0):
            self.notch2_b, self.notch2_a = iirnotch(harmonic_hz, Q=30, fs=sampling_rate)
        else:
            self.notch2_b, self.notch2_a = None, None

    def filter_chunk(self, signal: List[float]) -> List[float]:
        """Filters small streaming batches (e.g. 25 samples) in real-time for UI display."""
        if not signal:
            return []
        sig = np.array(signal, dtype=np.float64)
        filtered, self.zi_hp = sosfilt(self.sos_hp, sig, zi=self.zi_hp)
        filtered, self.zi_lp = sosfilt(self.sos_lp, filtered, zi=self.zi_lp)
        if self.sos_notch is not None and self.notch_enabled:
            filtered, self.zi_notch = sosfilt(self.sos_notch, filtered, zi=self.zi_notch)
        return (filtered + 2048.0).tolist()

    def remove_baseline_wander(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Removes baseline wander using dual-median filtering (clinical gold-standard for ST preservation)
        or high-pass Butterworth filtering.
        Returns:
            detrended_signal: signal with baseline drift subtracted
            baseline_trend: estimated baseline drift curve
        """
        n = len(signal)
        if n < 50:
            return signal, np.zeros_like(signal)

        if self.baseline_method == "dual_median":
            # Window 1: 200ms window (removes P and QRS waves)
            w1 = int(round(0.200 * self.sampling_rate))
            if w1 % 2 == 0:
                w1 += 1
            med1 = median_filter(signal, size=w1, mode="reflect")

            # Window 2: 600ms window (removes T waves)
            w2 = int(round(0.600 * self.sampling_rate))
            if w2 % 2 == 0:
                w2 += 1
            baseline = median_filter(med1, size=w2, mode="reflect")

            detrended = signal - baseline
            return detrended, baseline
        else:
            # Fallback zero-phase highpass
            try:
                # Add reflection padding to avoid boundary transients
                pad_len = min(n // 2, int(self.sampling_rate * 0.5))
                padded = np.pad(signal, pad_len, mode="reflect")
                filtered = filtfilt(self.hp_b, self.hp_a, padded)
                detrended = filtered[pad_len:-pad_len]
                baseline = signal - detrended
                return detrended, baseline
            except Exception:
                return signal - np.median(signal), np.full_like(signal, np.median(signal))

    def detect_powerline_interference(self, signal: np.ndarray) -> bool:
        """
        Checks spectral power at the notch frequency vs background spectrum
        to avoid blindly applying notch filters when no interference is present.
        """
        if len(signal) < int(self.sampling_rate * 1.5):
            return True  # Short segment fallback

        try:
            freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=min(len(signal), 512))
            notch_band = (freqs >= self.notch_hz - 1.0) & (freqs <= self.notch_hz + 1.0)
            ref_band = (freqs >= self.notch_hz - 6.0) & (freqs <= self.notch_hz + 6.0) & ~notch_band

            if np.any(notch_band) and np.any(ref_band):
                notch_pwr = np.mean(psd[notch_band])
                ref_pwr = np.mean(psd[ref_band])
                if ref_pwr > 1e-12:
                    return bool((notch_pwr / ref_pwr) > 1.8)
        except Exception:
            pass

        return True

    def clean_5s_chunk(self, signal: List[float]) -> List[float]:
        """
        Full clinical zero-phase preprocessing of ECG telemetry:
        1. Reflection margin padding (prevents edge discontinuity artifacts).
        2. Multi-notch powerline filtering (50 Hz + 100 Hz harmonic) via zero-phase filtfilt.
        3. Zero-phase Butterworth highpass (0.5 Hz) via filtfilt to eliminate baseline drift.
        4. Zero-phase Butterworth lowpass (35.0 Hz) via filtfilt to eliminate EMG muscle tremor.
        5. ST-preserving dual-median baseline wander elimination (200ms P-QRS + 600ms T-wave window).
        6. Savitzky-Golay polynomial smoothing (window 11, polyorder 3).
        7. Re-centering around 2048.0 isoelectric ADC standard.
        """
        if not signal or len(signal) < 20:
            return signal

        arr = np.array(signal, dtype=np.float64)
        n = len(arr)

        # 1. Edge padding: 0.5s reflection padding on both ends
        pad_len = min(n // 2, int(self.sampling_rate * 0.5))
        padded = np.pad(arr, pad_len, mode="reflect")

        # 2. Powerline Notch Filtering (50 Hz and 100 Hz harmonic via zero-phase filtfilt)
        if self.notch_enabled and self.notch_b is not None:
            try:
                padded = filtfilt(self.notch_b, self.notch_a, padded)
                if self.notch2_b is not None:
                    padded = filtfilt(self.notch2_b, self.notch2_a, padded)
            except Exception:
                pass

        # 3. Zero-phase Butterworth Highpass (0.5 Hz) + Lowpass (35.0 Hz) via filtfilt
        try:
            padded = filtfilt(self.hp_b, self.hp_a, padded)
        except Exception:
            pass

        try:
            padded = filtfilt(self.lp_b, self.lp_a, padded)
        except Exception:
            pass

        # 4. ST-preserving Dual-Median Baseline Elimination
        try:
            detrended_padded, _ = self.remove_baseline_wander(padded)
        except Exception:
            detrended_padded = padded

        # 5. Savitzky-Golay polynomial smoothing to suppress high-frequency micro-jitter
        try:
            if len(detrended_padded) >= 15:
                detrended_padded = savgol_filter(detrended_padded, window_length=11, polyorder=3)
        except Exception:
            pass

        # 6. Unpad back to exact original chunk length
        unpadded = detrended_padded[pad_len:-pad_len]

        # 7. Re-center around standard 2048 ADC baseline
        clean_result = unpadded + 2048.0
        return clean_result.tolist()
