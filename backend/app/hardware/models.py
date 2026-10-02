from typing import List, Optional
from pydantic import BaseModel

class DevicePacket(BaseModel):
    next: int = 0
    timestamp_ms: Optional[int] = None
    ecg: List[float] = []
    ir: List[float] = []
    ecgHr: Optional[float] = 0.0
    hr: Optional[float] = 0.0
    spo2: Optional[float] = 0.0
    loP: int = 0
    loM: int = 0
    pitch: float = 0.0
    roll: float = 0.0
    motion: float = 1.0
