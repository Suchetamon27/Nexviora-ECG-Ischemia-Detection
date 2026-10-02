import asyncio
import json
import ast
import random
import time
import math
import serial
from typing import AsyncGenerator

from .models import DevicePacket
from ..config import settings


class SerialECGReader:
    def __init__(self, port: str, baudrate: int) -> None:
        self.port = port
        self.baudrate = baudrate
        self._serial: serial.Serial | None = None
        self.mock_mode = settings.hardware_mode == "mock"

    def connect(self) -> None:
        if self.mock_mode:
            return
        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=1,
        )

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def _parse_line(self, line: str) -> DevicePacket | None:
        # Try JSON first
        try:
            data = json.loads(line)
            return DevicePacket(**data)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try CSV with nested arrays (ast.literal_eval is safe for lists)
        parts = line.strip().split(',"', 1)
        if len(parts) == 1:
            return None # Failed to parse
        # Very rudimentary fallback if it's not JSON
        return None

    async def read(self) -> AsyncGenerator[DevicePacket, None]:
        if self.mock_mode:
            seq = 0
            while True:
                await asyncio.sleep(1.0)
                t = time.time()
                ecg = []
                for i in range(250):
                    val = 2000 + 500 * math.sin(2 * math.pi * 1.2 * (t + i/250.0)) + random.uniform(-50, 50)
                    ecg.append(val)
                
                # simulate motion and lead off sometimes
                motion = 1.0 + random.uniform(-0.1, 0.1)
                loP, loM = 0, 0
                if random.random() < 0.05:
                    motion = 2.5
                if random.random() < 0.02:
                    loP = 1
                
                packet = DevicePacket(
                    next=seq,
                    ecg=ecg,
                    ir=[40000 + random.uniform(-100, 100) for _ in range(250)],
                    ecgHr=72.0,
                    hr=70.0,
                    spo2=98.0,
                    loP=loP,
                    loM=loM,
                    pitch=0.0,
                    roll=0.0,
                    motion=motion
                )
                seq += 1
                yield packet

        if self._serial is None:
            self.connect()

        while True:
            line = await asyncio.to_thread(
                self._serial.readline
            )

            if not line:
                continue

            try:
                decoded = line.decode("utf-8")
            except UnicodeDecodeError:
                continue

            sample = self._parse_line(decoded)

            if sample is not None:
                yield sample