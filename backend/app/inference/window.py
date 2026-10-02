from collections import deque
from typing import List, Optional, Dict, Any

class ECGWindow:
    def __init__(
        self,
        sampling_rate: int = 250,
        window_seconds: float = 5.0,
        step_seconds: float = 5.0,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.window_size = int(sampling_rate * window_seconds) # 1250 samples
        self.step_size = int(sampling_rate * step_seconds)     # 1250 samples

        self.ecg_samples = deque(maxlen=self.window_size)
        self.raw_ecg_samples = deque(maxlen=self.window_size)
        self.motion_samples = deque(maxlen=self.window_size)
        self.lead_off_samples = deque(maxlen=self.window_size)
        self.ppg_samples = deque(maxlen=self.window_size)
        self.timestamps = deque(maxlen=self.window_size)

        self.last_hr = 72.0
        self.last_spo2 = 98.0
        self.last_pitch = 0.0
        self.last_roll = 0.0
        self.last_motion = 1.0

        self._since_last_window = 0

    def add_packet(self, filtered_ecg: List[float], raw_ecg: List[float], packet) -> Optional[Dict[str, Any]]:
        """
        Adds samples from a DevicePacket. When 5 seconds of samples accumulate,
        returns the packaged 5-second chunk.
        """
        n = len(filtered_ecg)
        self.last_hr = packet.ecgHr if packet.ecgHr and packet.ecgHr > 0 else (packet.hr if packet.hr and packet.hr > 0 else self.last_hr)
        self.last_spo2 = packet.spo2 if packet.spo2 and packet.spo2 > 0 else self.last_spo2
        self.last_pitch = packet.pitch
        self.last_roll = packet.roll
        self.last_motion = packet.motion
        lo_flag = 1 if (packet.loP != 0 or packet.loM != 0) else 0

        for i in range(n):
            self.ecg_samples.append(filtered_ecg[i])
            if i < len(raw_ecg):
                self.raw_ecg_samples.append(raw_ecg[i])
            else:
                self.raw_ecg_samples.append(filtered_ecg[i])

            if i < len(packet.ir):
                self.ppg_samples.append(packet.ir[i])
            else:
                self.ppg_samples.append(0.0)

            self.motion_samples.append(packet.motion)
            self.lead_off_samples.append(lo_flag)
            self.timestamps.append(packet.timestamp_ms if packet.timestamp_ms else 0)

            self._since_last_window += 1

        if len(self.ecg_samples) == self.window_size and self._since_last_window >= self.step_size:
            self._since_last_window = 0
            return {
                "ecg": list(self.ecg_samples),
                "raw_ecg": list(self.raw_ecg_samples),
                "ppg": list(self.ppg_samples),
                "motion": list(self.motion_samples),
                "lead_off": list(self.lead_off_samples),
                "timestamps": list(self.timestamps),
                "hr": self.last_hr,
                "spo2": self.last_spo2,
                "pitch": self.last_pitch,
                "roll": self.last_roll,
                "motion_latest": self.last_motion,
            }

        return None
