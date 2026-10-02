from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    hardware_mode: str = "serial"  # "serial" or "mock"
    serial_port: str = "/dev/ttyACM0"
    serial_baudrate: int = 115200

    sampling_rate: int = 250

    highpass_hz: float = 0.5
    lowpass_hz: float = 40.0
    notch_hz: float = 50.0

    motion_threshold: float = 0.18        # |motion_g - 1.0| > 0.18 indicates significant movement
    noise_rejection_threshold: float = 0.65 # Pass 1: reject chunk if > 65% of window is noisy

    inference_window_seconds: float = 5.0 # 5-second chunk (1,250 samples)
    inference_step_seconds: float = 5.0

    laya_model: str = "convaiinnovations/laya"
    model_temperature: float = 0.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
