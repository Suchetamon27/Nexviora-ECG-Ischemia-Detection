from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

from ..hardware.models import DevicePacket
from .filters import ECGFilter
from .sampling import SamplingValidator, SamplingReport
from .quality import SignalQualityAssessor, QualityAssessment
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
        self.sampling_validator = SamplingValidator(
            nominal_fs=settings.sampling_rate,
            jitter_tolerance_ms=settings.sampling_jitter_tolerance_ms,
            max_interpolatable_gap_ms=settings.max_interpolatable_gap_ms,
            critical_gap_threshold_ms=settings.critical_gap_threshold_ms,
        )
        self.filter = ECGFilter(
            sampling_rate=settings.sampling_rate,
            highpass_hz=settings.highpass_hz,
            lowpass_hz=settings.lowpass_hz,
            notch_hz=settings.notch_hz,
            notch_enabled=settings.notch_enabled,
            notch_adaptive=settings.notch_adaptive,
            baseline_method=settings.baseline_method,
        )
        self.quality_assessor = SignalQualityAssessor(
            sampling_rate=settings.sampling_rate,
            clean_threshold=settings.sqi_clean_threshold,
            degraded_threshold=settings.sqi_degraded_threshold,
            motion_threshold=settings.motion_threshold,
        )

    def process(self, packet: DevicePacket, timestamp_us: int) -> ProcessedChunk:
        filtered = self.filter.filter_chunk(packet.ecg)

        # Check motion deviation
        motion_dev = abs(packet.motion - 1.0)
        is_moving = motion_dev > settings.motion_threshold

        # Check lead railing
        railed_count = sum(1 for v in packet.ecg if v <= 100 or v >= 4000)
        railed_ratio = railed_count / max(len(packet.ecg), 1)

        # Compute instant noise index for live display
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
            packet=packet,
        )

    def evaluate_5s_window_quality(
        self,
        raw_ecg_samples: List[float],
        cleaned_ecg_samples: List[float],
        motion_samples: List[float],
        lead_off_flags: List[int],
        timestamps_ms: Optional[List[float]] = None,
    ) -> QualityAssessment:
        """
        Pass 1: Comprehensive multi-metric SQI & artifact evaluation over a full 5-second chunk.
        """
        # 1. Sampling & Gap validation
        _, samp_report = self.sampling_validator.validate_and_align(raw_ecg_samples, timestamps_ms)

        # 2. Multi-Metric SQI evaluation
        assessment = self.quality_assessor.evaluate_chunk(
            raw_ecg=raw_ecg_samples,
            cleaned_ecg=cleaned_ecg_samples,
            motion_samples=motion_samples,
            lead_off_flags=lead_off_flags,
            gap_detected=samp_report.has_critical_gap,
            gap_max_ms=samp_report.max_gap_ms,
        )

        return assessment

