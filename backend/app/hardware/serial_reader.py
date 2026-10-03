import asyncio
import json
import time
import math
import random
import re
import serial
from typing import AsyncGenerator, Optional, List

from .models import DevicePacket
from ..config import settings


class SerialECGReader:
    def __init__(self, port: str, baudrate: int) -> None:
        self.port = port
        self.baudrate = baudrate
        self._serial: Optional[serial.Serial] = None
        self.mock_mode = settings.hardware_mode == "mock"
        self.is_streaming = False
        self.discovered_wifi_ip: Optional[str] = None

    def connect(self) -> bool:
        if self.mock_mode:
            return True
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.2,
            )
            print(f"[SerialReader] Connected to {self.port} at {self.baudrate} baud.")
            return True
        except Exception as e:
            self._serial = None
            return False

    def close(self) -> None:
        if self._serial is not None and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def send_command(self, cmd: str) -> bool:
        """Sends a text command (e.g. #REC_START or #REC_STOP) to ESP32."""
        if self._serial and self._serial.is_open:
            try:
                msg = (cmd.strip() + "\n").encode("utf-8")
                self._serial.write(msg)
                self._serial.flush()
                print(f"[SerialReader] Sent command: {cmd}")
                return True
            except Exception as e:
                print(f"[SerialReader] Error sending command: {e}")
                return False
        return False

    def start_stream(self) -> bool:
        self.is_streaming = True
        return self.send_command("#REC_START")

    def stop_stream(self) -> bool:
        self.is_streaming = False
        return self.send_command("#REC_STOP")

    def _parse_rec_line(self, line: str) -> Optional[dict]:
        """
        Parses format: REC,timestamp_ms,ecg_filtered,ppg_dc_ir,ecg_bpm,spo2_pct,pitch_deg,roll_deg,motion_g
        """
        parts = line.strip().split(",")
        if len(parts) >= 9 and parts[0] == "REC":
            try:
                return {
                    "ts": int(parts[1]),
                    "ecg": float(parts[2]),
                    "ppg": float(parts[3]),
                    "ecg_hr": float(parts[4]),
                    "spo2": float(parts[5]),
                    "pitch": float(parts[6]),
                    "roll": float(parts[7]),
                    "motion": float(parts[8]),
                }
            except (ValueError, IndexError):
                return None
        return None

    async def read(self) -> AsyncGenerator[DevicePacket, None]:
        batch_ecg: List[float] = []
        batch_ir: List[float] = []
        last_hr = 72.0
        last_spo2 = 98.0
        last_pitch = 0.0
        last_roll = 0.0
        last_motion = 1.0
        seq = 0

        while True:
            # Mock mode or fallback when physical port not found
            if self.mock_mode or (self._serial is None and not self.connect()):
                if not self.mock_mode:
                    await asyncio.sleep(1.0)
                    # Try connecting again next cycle
                    continue

                # Mock generation at 10Hz packets (25 samples per packet = 250Hz)
                await asyncio.sleep(0.1)
                t_base = time.time()
                sim_ecg = []
                for i in range(25):
                    t = t_base + i / 250.0
                    # Normal sinus rhythm waveform simulation
                    phase = (t * 1.2) % 1.0  # ~72 bpm
                    # QRS complex
                    if 0.20 <= phase < 0.24:
                        v = 2048 + 1200 * math.sin((phase - 0.20) / 0.04 * math.pi)  # R peak
                    elif 0.18 <= phase < 0.20:
                        v = 2048 - 200 * math.sin((phase - 0.18) / 0.02 * math.pi)   # Q dip
                    elif 0.24 <= phase < 0.27:
                        v = 2048 - 350 * math.sin((phase - 0.24) / 0.03 * math.pi)   # S dip
                    elif 0.08 <= phase < 0.16:
                        v = 2048 + 180 * math.sin((phase - 0.08) / 0.08 * math.pi)   # P wave
                    elif 0.35 <= phase < 0.55:
                        v = 2048 + 320 * math.sin((phase - 0.35) / 0.20 * math.pi)   # T wave
                    else:
                        v = 2048 + random.uniform(-15, 15)  # Baseline noise
                    sim_ecg.append(round(v, 1))

                motion_val = 1.0 + random.uniform(-0.04, 0.04)
                loP = 0
                loM = 0

                packet = DevicePacket(
                    next=seq,
                    timestamp_ms=int(t_base * 1000),
                    ecg=sim_ecg,
                    ir=[32000 + random.uniform(-50, 50) for _ in range(25)],
                    ecgHr=72.0,
                    hr=72.0,
                    spo2=98.0,
                    loP=loP,
                    loM=loM,
                    pitch=1.5,
                    roll=-0.8,
                    motion=round(motion_val, 2)
                )
                seq += 1
                yield packet
                continue

            # Real serial reading
            try:
                line = await asyncio.to_thread(self._serial.readline)
                if not line:
                    await asyncio.sleep(0.01)
                    continue

                decoded = line.decode("utf-8", errors="ignore").strip()
                if "Local IP:" in decoded or "http://" in decoded:
                    m = re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", decoded)
                    if m:
                        ip = m.group(1)
                        if ip != "192.168.4.1":
                            self.discovered_wifi_ip = f"http://{ip}"
                            print(f"[SerialReader] Intercepted ESP32 Wi-Fi IP from Serial: {self.discovered_wifi_ip}")

                if decoded.startswith("REC,") or decoded.startswith("#"):
                    pass # regular traffic
                else:
                    print(f"[SerialReader Debug] Received: {decoded[:80]}")

                parsed = self._parse_rec_line(decoded)
                if parsed:
                    batch_ecg.append(parsed["ecg"])
                    batch_ir.append(parsed["ppg"])
                    last_hr = parsed["ecg_hr"] if parsed["ecg_hr"] > 0 else last_hr
                    last_spo2 = parsed["spo2"] if parsed["spo2"] > 0 else last_spo2
                    last_pitch = parsed["pitch"]
                    last_roll = parsed["roll"]
                    last_motion = parsed["motion"]

                    # Emit a packet every 25 samples (10 packets/second at 250Hz)
                    if len(batch_ecg) >= 25:
                        loP = 1 if (parsed["ecg"] >= 4050) else 0
                        loM = 1 if (parsed["ecg"] <= 50) else 0

                        packet = DevicePacket(
                            next=seq,
                            timestamp_ms=parsed["ts"],
                            ecg=list(batch_ecg),
                            ir=list(batch_ir),
                            ecgHr=last_hr,
                            hr=last_hr,
                            spo2=last_spo2,
                            loP=loP,
                            loM=loM,
                            pitch=last_pitch,
                            roll=last_roll,
                            motion=last_motion
                        )
                        batch_ecg.clear()
                        batch_ir.clear()
                        seq += 1
                        if seq % 10 == 0:
                            print(f"[SerialReader] Emitted 10 packets (250 samples/s). Latest HR: {last_hr} BPM")
                        yield packet

            except serial.SerialException as se:
                print(f"[SerialReader] Disconnected: {se}. Reconnecting...")
                self.close()
                await asyncio.sleep(1.5)
            except Exception as e:
                print(f"[SerialReader] Read error: {e}")
                await asyncio.sleep(0.05)
