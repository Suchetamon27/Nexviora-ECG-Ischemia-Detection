from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from ..hardware.models import DevicePacket
from .filters import ECGFilter
from ..config import settings


@dataclass(slots=True)
class ProcessedChunk:
    timestamp_us: int
    raw_ecg: List[float]
    filtered_ecg: List[float]
    signal_quality: str
    noise_percentage: float
    chance_good_data: float
    chance_noise_movement: float
    packet: DevicePacket


class SignalPipeline:
    def __init__(self) -> None:
        self.motion_threshold = settings.motion_threshold
        self.noise_rejection_threshold = settings.noise_rejection_threshold
        self.filter = ECGFilter(
            sampling_rate=settings.sampling_rate,
            highpass_hz=settings.highpass_hz,
            lowpass_hz=settings.lowpass_hz,
            notch_hz=settings.notch_hz,
        )

    def process(self, packet: DevicePacket, timestamp_us: int) -> ProcessedChunk:
        filtered = self.filter.filter_chunk(packet.ecg)
        
        # Check motion deviation
        motion_dev = abs(packet.motion - 1.0)
        is_moving = motion_dev > self.motion_threshold

        # Check lead railing
        railed_count = sum(1 for v in packet.ecg if v <= 100 or v >= 4000)
        railed_ratio = railed_count / max(len(packet.ecg), 1)

        # Compute instant noise index for this batch
        motion_noise = min(1.0, motion_dev / 0.5) if is_moving else 0.0
        lead_noise = 1.0 if (packet.loP != 0 or packet.loM != 0 or railed_ratio > 0.2) else (railed_ratio * 2.0)
        
        noise_ratio = min(1.0, max(motion_noise, lead_noise))
        noise_pct = round(noise_ratio * 100.0, 1)
        good_pct = round(max(0.0, 100.0 - noise_pct), 1)

        if packet.loP != 0 or packet.loM != 0 or railed_ratio > 0.3:
            quality = "LEAD_OFF"
        elif is_moving or noise_pct > 65.0:
            quality = "MOTION_CONTAMINATED"
        else:
            quality = "GOOD"

        return ProcessedChunk(
            timestamp_us=timestamp_us,
            raw_ecg=packet.ecg,
            filtered_ecg=filtered,
            signal_quality=quality,
            noise_percentage=noise_pct,
            chance_good_data=good_pct,
            chance_noise_movement=noise_pct,
            packet=packet
        )

    def evaluate_5s_window_quality(self, ecg_samples: List[float], motion_samples: List[float], lead_off_flags: List[int]) -> dict:
        """
        Pass 1: Comprehensive quality evaluation over a full 5-second chunk (1,250 samples).
        Tolerates up to 60-70% noise window as requested by user.
        """
        total = max(len(ecg_samples), 1)
        
        # 1. Railing or detached leads
        railed_samples = sum(1 for v in ecg_samples if v <= 100 or v >= 4000)
        lead_off_samples = sum(1 for lo in lead_off_flags if lo > 0)
        
        # 2. Movement / accelerometer variance
        motion_noisy_samples = sum(1 for m in motion_samples if abs(m - 1.0) > self.motion_threshold)

        # 3. Baseline instability
        ecg_arr = np.array(ecg_samples)
        std_val = float(np.std(ecg_arr))
        is_flat = std_val < 15.0 # flatline / sensor unplugged
        
        # Combined noise ratio over the 5-second chunk
        corrupted_samples = max(railed_samples, lead_off_samples, motion_noisy_samples)
        if is_flat:
            corrupted_samples = total

        noise_ratio = min(1.0, corrupted_samples / total)
        noise_pct = round(noise_ratio * 100.0, 1)
        good_pct = round(max(0.0, 100.0 - noise_pct), 1)

        # User specified: "take like 60 to 70 percent for noise as window"
        # Reject chunk if noise exceeds 65%
        is_rejected = noise_pct > (self.noise_rejection_threshold * 100.0) or is_flat

        rejection_reason = None
        if is_rejected:
            if is_flat:
                rejection_reason = "Signal Flatline / Detached Electrode"
            elif motion_noisy_samples > (self.noise_rejection_threshold * total):
                rejection_reason = f"Excessive Motion Artifact ({noise_pct}% contaminated)"
            else:
                rejection_reason = f"High Baseline Noise / Electrode Railing ({noise_pct}%)"

        return {
            "is_rejected": is_rejected,
            "rejection_reason": rejection_reason,
            "noise_percentage": noise_pct,
            "chance_good_data": good_pct,
            "chance_noise_movement": noise_pct,
            "signal_std": round(std_val, 1)
        }
