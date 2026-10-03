import asyncio
import os
import glob
import serial
from typing import AsyncGenerator, Optional, List
from .models import DevicePacket
from .wifi_reader import WifiECGReader
from .serial_reader import SerialECGReader
from ..config import settings


class HybridECGReader:
    """
    Hybrid Hardware Telemetry Reader:
    - Primary: Wi-Fi stream from ESP32.
    - Fallback: Scans USB Serial ports (/dev/ttyACM*, /dev/ttyUSB*) if Wi-Fi is disconnected.
    - Strict No-Fake-Data Guarantee: Reads real physical telemetry only.
    """

    def __init__(self, default_wifi_url: str = "http://192.168.4.1", default_serial_port: str = "/dev/ttyACM0"):
        self.wifi_reader = WifiECGReader(base_url=default_wifi_url)
        self.serial_port = default_serial_port
        self.serial_baudrate = settings.serial_baudrate
        self.active_mode = "disconnected"
        self.status_message = "Initializing hybrid reader (Wi-Fi + USB Serial)..."
        self.is_streaming = True
        self.serial_reader: Optional[SerialECGReader] = None
        self.discovered_ip = None

    @property
    def is_connected(self) -> bool:
        return self.active_mode in ["wifi", "usb_serial"]

    def set_esp_url(self, new_url: str):
        self.wifi_reader.set_esp_url(new_url)

    async def auto_discover_esp32_ip(self):
        return await self.wifi_reader.auto_discover_esp32_ip()

    def start_stream(self):
        self.is_streaming = True
        self.wifi_reader.start_stream()
        if self.serial_reader:
            self.serial_reader.start_stream()
        return True

    def stop_stream(self):
        self.is_streaming = False
        self.wifi_reader.stop_stream()
        if self.serial_reader:
            self.serial_reader.stop_stream()
        return True

    def _find_available_serial_ports(self) -> List[str]:
        """Finds active USB serial ports on Linux."""
        ports = []
        for pattern in ["/dev/ttyACM*", "/dev/ttyUSB*", "/dev/tty.usbmodem*", "/dev/tty.usbserial*"]:
            ports.extend(glob.glob(pattern))
        if self.serial_port and self.serial_port not in ports and os.path.exists(self.serial_port):
            ports.insert(0, self.serial_port)
        return list(dict.fromkeys(ports))

    async def read(self) -> AsyncGenerator[DevicePacket, None]:
        while True:
            # -------------------------------------------------------------
            # STEP 1: Try Wi-Fi Connection
            # -------------------------------------------------------------
            wifi_ok = await self.wifi_reader.connect()
            if wifi_ok:
                self.active_mode = "wifi"
                self.discovered_ip = self.wifi_reader.discovered_ip
                self.status_message = f"Connected to ESP32 over Wi-Fi ({self.wifi_reader.target_url})"
                print(f"[HybridECGReader] Active mode: Wi-Fi ({self.wifi_reader.target_url})")

                try:
                    async for packet in self.wifi_reader.read():
                        yield packet
                except Exception as e:
                    print(f"[HybridECGReader] Wi-Fi stream error: {e}")

                self.wifi_reader.is_connected = False
                await self.wifi_reader.close()

            # -------------------------------------------------------------
            # STEP 2: Fallback to USB Serial if Wi-Fi unavailable
            # -------------------------------------------------------------
            candidate_ports = self._find_available_serial_ports()
            connected_serial = False

            for port in candidate_ports:
                print(f"[HybridECGReader] Attempting USB Serial connection on {port}...")
                s_reader = SerialECGReader(port=port, baudrate=self.serial_baudrate)
                s_reader.mock_mode = False  # Strictly read real USB hardware

                if s_reader.connect():
                    self.serial_reader = s_reader
                    self.active_mode = "usb_serial"
                    self.status_message = f"Connected to ESP32 over USB Serial ({port})"
                    print(f"[HybridECGReader] Active mode: USB Serial ({port})")
                    connected_serial = True

                    try:
                        async for packet in s_reader.read():
                            if s_reader.discovered_wifi_ip:
                                self.wifi_reader.set_esp_url(s_reader.discovered_wifi_ip)
                                s_reader.discovered_wifi_ip = None
                            yield packet
                    except Exception as e:
                        print(f"[HybridECGReader] USB Serial stream error on {port}: {e}")

                    s_reader.close()
                    self.serial_reader = None
                    break

            if not connected_serial:
                self.active_mode = "disconnected"
                self.status_message = "Neither Wi-Fi nor USB Serial ESP32 hardware detected. Retrying discovery..."
                await asyncio.sleep(3.0)
