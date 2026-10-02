from typing import List, Optional
from pydantic import BaseModel

class DevicePacket(BaseModel):
    next: int
    ecg: List[float]
    ir: List[float]
    ecgHr: Optional[float] = None
    hr: Optional[float] = None
    spo2: Optional[float] = None
    loP: int = 0
    loM: int = 0
    pitch: float = 0.0
    roll: float = 0.0
    motion: float = 0.0