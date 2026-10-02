import numpy as np
from scipy.signal import butter, sosfilt

class ECGFilter:
    def __init__(
        self,
        sampling_rate: int,
        highpass_hz: float,
        lowpass_hz: float,
        notch_hz: float,
    ) -> None:
        self.sampling_rate = sampling_rate

        self.sos_hp = butter(4, highpass_hz, btype="highpass", fs=sampling_rate, output="sos")
        self.zi_hp = np.zeros((self.sos_hp.shape[0], 2))

        self.sos_lp = butter(4, lowpass_hz, btype="lowpass", fs=sampling_rate, output="sos")
        self.zi_lp = np.zeros((self.sos_lp.shape[0], 2))

        # Notch using butterworth bandstop for simplicity with sosfilt
        self.sos_notch = butter(4, [notch_hz - 1, notch_hz + 1], btype="bandstop", fs=sampling_rate, output="sos")
        self.zi_notch = np.zeros((self.sos_notch.shape[0], 2))

    def filter_chunk(self, signal: list[float]) -> list[float]:
        if not signal:
            return []
        
        sig = np.array(signal, dtype=np.float64)
        
        filtered, self.zi_hp = sosfilt(self.sos_hp, sig, zi=self.zi_hp)
        filtered, self.zi_lp = sosfilt(self.sos_lp, filtered, zi=self.zi_lp)
        filtered, self.zi_notch = sosfilt(self.sos_notch, filtered, zi=self.zi_notch)

        return filtered.tolist()