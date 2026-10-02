from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    hardware_mode: str = "mock"  # "mock" or "serial"
    serial_port: str = "/dev/ttyACM0"
    serial_baudrate: int = 115200

    sampling_rate: int = 250

    highpass_hz: float = 0.5
    lowpass_hz: float = 40.0
    notch_hz: float = 50.0

    motion_threshold: float = 1.5

    inference_window_seconds: float = 10.0
    inference_step_seconds: float = 1.0

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "ecg-ischemia"

    model_temperature: float = 0.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

settings = Settings()