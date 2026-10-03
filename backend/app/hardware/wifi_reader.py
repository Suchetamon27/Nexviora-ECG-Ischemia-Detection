import asyncio
import time
import subprocess
import re
import httpx
from typing import AsyncGenerator, Optional, List, Dict, Any

from .models import DevicePacket
from ..config import settings


class WifiECGReader:
    """
    Asynchronous Wi-Fi Telemetry Reader for ESP32.
    - NO FAKE DATA: Strictly reads real hardware telemetry from ESP32.
    - AUTO-DISCOVERY: Scans local network subnets, 192.168.4.1 (ESP32 AP),
      and mDNS endpoints to find the ESP32's IP address automatically.
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.target_url = (base_url or settings.esp_wifi_url).rstrip("/")
        self.discovered_ip: Optional[str] = None
        self.is_connected = False
        self.is_streaming = True

        self.last_seq: int = 0
        self.last_pseq: int = 0
        self.client: Optional[httpx.AsyncClient] = None
        self.is_scanning = False
        self.status_message = "Disconnected / Initializing..."

    def set_esp_url(self, new_url: str) -> None:
        self.target_url = new_url.rstrip("/")
        self.discovered_ip = self.target_url.replace("http://", "").replace("https://", "")
        self.is_connected = False
        print(f"[WifiECGReader] Set target ESP32 Wi-Fi URL to: {self.target_url}")

    def get_candidate_ips(self) -> List[str]:
        """Gathers potential ESP32 IP targets (configured IP, mDNS, AP default, subnets)."""
        candidates = []

        # 1. Configured target URL IP / Hostname
        if self.target_url:
            clean_target = self.target_url.replace("http://", "").replace("https://", "").split("/")[0]
            candidates.append(clean_target)

        # 2. mDNS hostname on local Wi-Fi
        candidates.append("nexviora-ecg.local")

        # 3. ESP32 SoftAP default IP
        candidates.append("192.168.4.1")

        # 4. Active subnets from system routing table (supports 10.x.x.x, 192.168.x.x, 172.x.x.x)
        common_prefixes = ["10.70.94", "192.168.43", "172.20.10", "192.168.1", "192.168.0", "192.168.137", "10.42.0"]
        try:
            output = subprocess.check_output(["ip", "route"], stderr=subprocess.DEVNULL).decode("utf-8")
            for line in output.splitlines():
                if "docker" in line or "virbr" in line or "br-" in line or "tailscale" in line:
                    continue
                m = re.search(r"(\d+\.\d+\.\d+)\.0/\d+", line)
                if m:
                    prefix = m.group(1)
                    if prefix not in common_prefixes:
                        common_prefixes.insert(0, prefix)
        except Exception:
            pass

        for prefix in common_prefixes:
            for i in range(1, 255):
                candidates.append(f"{prefix}.{i}")

        # Deduplicate while preserving priority order
        return list(dict.fromkeys(candidates))

    async def _test_ip_endpoint(self, ip_or_url: str, client: httpx.AsyncClient) -> Optional[str]:
        """Tests if an IP address hosts the ESP32 /data endpoint."""
        url = ip_or_url if ip_or_url.startswith("http") else f"http://{ip_or_url}"
        try:
            resp = await client.get(f"{url}/data", timeout=0.5)
            if resp.status_code == 200:
                data = resp.json()
                if "ecg" in data or "next" in data:
                    return url
        except Exception:
            pass
        return None

    async def auto_discover_esp32_ip(self) -> Optional[str]:
        """Scans local network targets to automatically discover the ESP32 IP."""
        if self.is_scanning:
            return None

        self.is_scanning = True
        self.status_message = "Auto-scanning local network for ESP32 Wi-Fi IP..."
        candidates = self.get_candidate_ips()

        async with httpx.AsyncClient() as scan_client:
            # Batch scan in chunks of 50 IPs
            chunk_size = 50
            for i in range(0, len(candidates), chunk_size):
                batch = candidates[i : i + chunk_size]
                tasks = [self._test_ip_endpoint(ip, scan_client) for ip in batch]
                results = await asyncio.gather(*tasks)

                for res in results:
                    if res:
                        self.discovered_ip = res
                        self.target_url = res
                        self.is_connected = True
                        self.is_scanning = False
                        self.status_message = f"Connected to ESP32 Wi-Fi at {res}"
                        print(f"*** [WifiECGReader] AUTO-DISCOVERED ESP32 AT: {res} ***")
                        self.start_stream()
                        return res

        self.is_scanning = False
        self.status_message = "ESP32 Wi-Fi device not found on local network. Ensure ESP32 is powered on and connected."
        return None

    async def connect(self) -> bool:
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=2.0, read=4.0, write=2.0, pool=2.0)
            )

        # First try explicit target URL
        found_url = await self._test_ip_endpoint(self.target_url, self.client)
        if found_url:
            self.discovered_ip = found_url
            self.is_connected = True
            self.status_message = f"Connected to ESP32 at {found_url}"
            self.start_stream()
            return True

        # Run auto-discovery if target failed
        found_url = await self.auto_discover_esp32_ip()
        if found_url:
            return True

        self.is_connected = False
        return False

    async def close(self) -> None:
        if self.client and not self.client.is_closed:
            await self.client.aclose()
        self.client = None
        self.is_connected = False

    async def send_command(self, action: str) -> bool:
        """Sends HTTP control command to ESP32 (on=start or on=stop)."""
        if self.client is None or self.client.is_closed:
            await self.connect()

        if self.client and self.is_connected:
            try:
                resp = await self.client.post(f"{self.target_url}/rec", data={"on": action})
                if resp.status_code == 200:
                    print(f"[WifiECGReader] Sent Wi-Fi command: /rec on={action}")
                    return True
                resp = await self.client.get(f"{self.target_url}/reading?on={action}")
                if resp.status_code == 200:
                    print(f"[WifiECGReader] Sent Wi-Fi command: /reading on={action}")
                    return True
            except Exception as e:
                print(f"[WifiECGReader] Error sending Wi-Fi command '{action}': {e}")

        return False

    def start_stream(self) -> bool:
        self.is_streaming = True
        asyncio.create_task(self.send_command("start"))
        return True

    def stop_stream(self) -> bool:
        self.is_streaming = False
        asyncio.create_task(self.send_command("stop"))
        return True

    async def read(self) -> AsyncGenerator[DevicePacket, None]:
        batch_ecg: List[float] = []
        batch_ir: List[float] = []
        last_hr = 72.0
        last_spo2 = 98.0
        last_pitch = 0.0
        last_roll = 0.0
        last_motion = 1.0
        seq = 0

        poll_interval_s = settings.esp_poll_interval_ms / 1000.0

        while True:
            # -------------------------------------------------------------
            # NO FAKE DATA: Strictly wait for real ESP32 connection
            # -------------------------------------------------------------
            if not self.is_connected:
                connected = await self.connect()
                if not connected:
                    # Wait 3 seconds before retrying discovery (NO FAKE GENERATION)
                    await asyncio.sleep(3.0)
                    continue

            # -------------------------------------------------------------
            # Real ESP32 Wi-Fi HTTP Stream
            # -------------------------------------------------------------
            try:
                url = f"{self.target_url}/data?after={self.last_seq}&pafter={self.last_pseq}"
                resp = await self.client.get(url)

                if resp.status_code == 200:
                    data = resp.json()
                    new_ecg = data.get("ecg", [])
                    new_ir = data.get("ir", [])

                    self.last_seq = data.get("next", self.last_seq)
                    self.last_pseq = data.get("pnext", self.last_pseq)

                    ecg_hr = float(data.get("ecgHr", 0.0))
                    hr_ppg = float(data.get("hr", 0.0))
                    spo2_val = float(data.get("spo2", 0.0))
                    loP_val = int(data.get("loP", 0))
                    loM_val = int(data.get("loM", 0))
                    pitch_val = float(data.get("pitch", 0.0))
                    roll_val = float(data.get("roll", 0.0))
                    motion_val = float(data.get("motion", 1.0))

                    dc_ir_val = float(data.get("dcIr", 0.0))
                    amp_ir_val = float(data.get("ampIr", 0.0))
                    reading_val = bool(data.get("reading", False))
                    finger_detected = (dc_ir_val > 8000.0) or reading_val

                    # ECG HR from chest/arm electrodes (AD8232)
                    if ecg_hr > 0:
                        last_hr = ecg_hr

                    # Optical PPG Pulse HR & SpO2 strictly active ONLY when finger is physically detected
                    current_optical_hr = hr_ppg if (finger_detected and hr_ppg > 0) else 0.0
                    current_spo2 = spo2_val if (finger_detected and spo2_val > 0) else 0.0

                    last_pitch = pitch_val
                    last_roll = roll_val
                    last_motion = motion_val

                    batch_ecg.extend(new_ecg)
                    batch_ir.extend(new_ir if finger_detected else [0.0] * len(new_ecg))

                    while len(batch_ecg) >= 25:
                        sub_ecg = batch_ecg[:25]
                        sub_ir = batch_ir[:25] if len(batch_ir) >= 25 else ([0.0] * 25)

                        batch_ecg = batch_ecg[25:]
                        batch_ir = batch_ir[25:] if len(batch_ir) >= 25 else []

                        packet = DevicePacket(
                            next=seq,
                            timestamp_ms=int(time.time() * 1000),
                            ecg=sub_ecg,
                            ir=sub_ir,
                            ecgHr=last_hr,
                            hr=current_optical_hr,
                            spo2=current_spo2,
                            loP=loP_val,
                            loM=loM_val,
                            pitch=last_pitch,
                            roll=last_roll,
                            motion=last_motion,
                            dcIr=dc_ir_val if finger_detected else 0.0,
                            ampIr=amp_ir_val if finger_detected else 0.0,
                            fingerDetected=finger_detected,
                        )
                        seq += 1
                        yield packet

            except Exception as e:
                print(f"[WifiECGReader] Lost connection to {self.target_url}: {e}")
                self.is_connected = False
                await self.close()
                await asyncio.sleep(2.0)

            await asyncio.sleep(poll_interval_s)
