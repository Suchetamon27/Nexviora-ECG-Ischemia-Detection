import numpy as np
from scipy.signal import butter, sosfilt, filtfilt, iirnotch

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False


class ECGFilter:
    def __init__(
        self,
        sampling_rate: int = 250,
        highpass_hz: float = 0.5,
        lowpass_hz: float = 40.0,
        notch_hz: float = 50.0,
    ) -> None:
        self.sampling_rate = sampling_rate

        # Real-time sos filter for continuous 25-sample stream
        self.sos_hp = butter(3, highpass_hz, btype="highpass", fs=sampling_rate, output="sos")
        self.zi_hp = np.zeros((self.sos_hp.shape[0], 2))

        self.sos_lp = butter(3, lowpass_hz, btype="lowpass", fs=sampling_rate, output="sos")
        self.zi_lp = np.zeros((self.sos_lp.shape[0], 2))

        self.sos_notch = butter(3, [notch_hz - 1.5, notch_hz + 1.5], btype="bandstop", fs=sampling_rate, output="sos")
        self.zi_notch = np.zeros((self.sos_notch.shape[0], 2))

        # SciPy zero-phase filter for batch chunks
        self.bp_b, self.bp_a = butter(3, [highpass_hz, lowpass_hz], fs=sampling_rate, btype="band")
        self.notch_b, self.notch_a = iirnotch(notch_hz, Q=30, fs=sampling_rate)

    def filter_chunk(self, signal: list[float]) -> list[float]:
        """Filters small streaming batches (e.g. 25 samples) for live display."""
        if not signal:
            return []
        sig = np.array(signal, dtype=np.float64)
        filtered, self.zi_hp = sosfilt(self.sos_hp, sig, zi=self.zi_hp)
        filtered, self.zi_lp = sosfilt(self.sos_lp, filtered, zi=self.zi_lp)
        filtered, self.zi_notch = sosfilt(self.sos_notch, filtered, zi=self.zi_notch)
        return filtered.tolist()

    def clean_5s_chunk(self, signal: list[float]) -> list[float]:
        """
        Deep clinical cleaning of a full 5-second chunk (1,250 samples):
        - Eliminates 50 Hz powerline mains hum
        - Removes respiratory baseline wander and DC drift
        - Suppresses high-frequency EMG muscle noise
        - Preserves true P-wave, QRS complex, and ST-T segment morphology
        """
        if not signal or len(signal) < 20:
            return signal

        arr = np.array(signal, dtype=np.float64)

        if HAS_NEUROKIT and len(arr) >= 250:
            try:
                # NeuroKit2 medical-grade zero-phase filtering
                cleaned = nk.ecg_clean(arr, sampling_rate=self.sampling_rate, method="neurokit")
                # Center around standard 2048 ADC baseline
                return (cleaned + 2048.0).tolist()
            except Exception as e:
                pass

        # Fallback to zero-phase filtfilt + iirnotch
        try:
            median_val = np.median(arr)
            centered = arr - median_val
            clean = filtfilt(self.bp_b, self.bp_a, centered)
            clean = filtfilt(self.notch_b, self.notch_a, clean)
            return (clean + 2048.0).tolist()
        except Exception:
            return signal
