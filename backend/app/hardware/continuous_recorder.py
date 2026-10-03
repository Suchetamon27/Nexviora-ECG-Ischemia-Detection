import os
import csv
import time
from datetime import datetime
from typing import List, Optional
from .models import DevicePacket
from ..config import settings


class ContinuousCSVRecorder:
    """
    Manages continuous live streaming CSV recording files.
    Every time a new stream/recording session starts, opens a BRAND NEW timestamped CSV file
    and continuously records raw and preprocessed ECG, PPG, SpO2, HR, motion, and quality metrics.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        if output_dir is None:
            self.output_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", settings.recordings_folder)
            )
        else:
            self.output_dir = os.path.abspath(output_dir)

        os.makedirs(self.output_dir, exist_ok=True)

        self.current_file_path: Optional[str] = None
        self._file_handle = None
        self._writer = None
        self.sample_counter: int = 0
        self.is_recording: bool = False

    def start_new_recording(self) -> str:
        """Closes any active recording file and creates a BRAND NEW CSV file."""
        self.stop_recording()

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_prefix = settings.hardware_mode
        filename = f"stream_{mode_prefix}_{timestamp_str}.csv"
        self.current_file_path = os.path.join(self.output_dir, filename)

        headers = [
            "timestamp_ms",
            "sample_seq",
            "ecg_raw",
            "ecg_cleaned",
            "ppg_dc_ir",
            "ecg_bpm",
            "spo2_pct",
            "pitch_deg",
            "roll_deg",
            "motion_g",
            "signal_quality",
        ]

        self._file_handle = open(self.current_file_path, mode="w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file_handle)
        self._writer.writerow(headers)
        self._file_handle.flush()

        self.sample_counter = 0
        self.is_recording = True
        print(f"[ContinuousCSVRecorder] Started NEW recording file: {self.current_file_path}")
        return self.current_file_path

    def record_packet(
        self,
        packet: DevicePacket,
        cleaned_samples: Optional[List[float]] = None,
        quality_status: str = "GOOD",
    ) -> None:
        """Continuously writes all samples in a DevicePacket to the active CSV file."""
        if not self.is_recording or self._writer is None:
            # Auto-start a recording if none is active
            self.start_new_recording()

        try:
            n_raw = len(packet.ecg)
            n_clean = len(cleaned_samples) if cleaned_samples else 0
            n_ir = len(packet.ir)

            for i in range(n_raw):
                raw_val = packet.ecg[i]
                clean_val = cleaned_samples[i] if i < n_clean else raw_val
                ir_val = packet.ir[i] if i < n_ir else 0.0

                self.sample_counter += 1
                row = [
                    packet.timestamp_ms,
                    self.sample_counter,
                    raw_val,
                    clean_val,
                    ir_val,
                    packet.hr,
                    packet.spo2,
                    packet.pitch,
                    packet.roll,
                    packet.motion,
                    quality_status,
                ]
                self._writer.writerow(row)

            # Flush periodically
            if self._file_handle and self.sample_counter % 250 == 0:
                self._file_handle.flush()

        except Exception as e:
            print(f"[ContinuousCSVRecorder] Write error: {e}")

    def stop_recording(self) -> Optional[str]:
        """Flushes and closes active CSV file."""
        saved_path = self.current_file_path
        if self._file_handle and not self._file_handle.closed:
            try:
                self._file_handle.flush()
                self._file_handle.close()
                print(f"[ContinuousCSVRecorder] Finalized recording ({self.sample_counter} samples): {saved_path}")
            except Exception:
                pass

        self._file_handle = None
        self._writer = None
        self.is_recording = False
        return saved_path


continuous_recorder = ContinuousCSVRecorder()
