from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    hardware_mode: str = "wifi"  # "wifi", "serial", or "mock"
    esp_wifi_url: str = "http://192.168.4.1"  # ESP32 Access Point IP
    esp_poll_interval_ms: float = 50.0        # Poll ESP32 HTTP data endpoint every 50ms
    serial_port: str = "/dev/ttyACM0"
    serial_baudrate: int = 115200
    recordings_folder: str = "recordings"


    sampling_rate: int = 250
    sampling_jitter_tolerance_ms: float = 8.0
    max_interpolatable_gap_ms: float = 20.0
    critical_gap_threshold_ms: float = 40.0

    # Filtering & Preprocessing Configuration
    highpass_hz: float = 0.5
    lowpass_hz: float = 35.0
    notch_enabled: bool = True
    notch_hz: float = 50.0  # 50.0 Hz or 60.0 Hz
    notch_adaptive: bool = True  # Check spectral peak before applying notch
    baseline_method: str = "dual_median"  # "dual_median" (clinical ST-preserving) or "butterworth"

    # Motion & Quality Thresholds
    motion_threshold: float = 0.18        # |motion_g - 1.0| > 0.18 indicates significant movement
    noise_rejection_threshold: float = 0.65 # Threshold window

    # Signal Quality Index (SQI) Thresholds
    sqi_clean_threshold: float = 0.70     # Score >= 0.70 -> CLEAN (70% good data gate)
    sqi_degraded_threshold: float = 0.50  # Score 0.50 to 0.69 -> DEGRADED

    inference_window_seconds: float = 3.0 # 3-second chunk (750 samples @ 250Hz)
    inference_step_seconds: float = 3.0

    laya_model: str = "z-ai/glm-5.3"
    model_temperature: float = 0.5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

