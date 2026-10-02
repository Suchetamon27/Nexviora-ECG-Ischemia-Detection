from dataclasses import dataclass
from typing import List, Optional

from ..hardware.models import DevicePacket
from .filters import ECGFilter
from ..config import settings


@dataclass(slots=True)
class ProcessedChunk:
    timestamp_us: int
    raw_ecg: List[float]
    filtered_ecg: List[float]
    signal_quality: str
    packet: DevicePacket


class SignalPipeline:
    def __init__(self) -> None:
        self.motion_threshold = settings.motion_threshold
        self.filter = ECGFilter(
            sampling_rate=settings.sampling_rate,
            highpass_hz=settings.highpass_hz,
            lowpass_hz=settings.lowpass_hz,
            notch_hz=settings.notch_hz,
        )

    def process(self, packet: DevicePacket, timestamp_us: int) -> ProcessedChunk:
        filtered = self.filter.filter_chunk(packet.ecg)
        
        if packet.loP != 0 or packet.loM != 0:
            quality = "LEAD_OFF"
        elif packet.motion > self.motion_threshold:
            quality = "MOTION_CONTAMINATED"
        else:
            quality = "GOOD"

        return ProcessedChunk(
            timestamp_us=timestamp_us,
            raw_ecg=packet.ecg,
            filtered_ecg=filtered,
            signal_quality=quality,
            packet=packet
        )