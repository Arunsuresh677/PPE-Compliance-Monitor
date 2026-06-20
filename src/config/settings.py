"""
src/config/settings.py — Centralised configuration via environment variables.

All settings can be overridden with PPE_* env vars or a .env file:
    PPE_MODEL_PATH=weights/best.pt
    PPE_CONF_THRESHOLD=0.6
    PPE_DB_PATH=/data/violations.db
    PPE_API_PORT=8000
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PPE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Model
    model_url: str = (
        "https://huggingface.co/Arunsuresh677/ppe-compliance-monitor"
        "/resolve/main/best.pt"
    )
    model_path: str = "best.pt"

    # Inference
    conf_threshold: float = 0.5
    imgsz: int = 640

    # Tracking
    stale_timeout_secs: float = 3.0

    # Database
    database_url: str = "postgresql://ppe_user:ppe_pass@localhost:5432/ppe_db"

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Alerts
    slack_webhook_url:     str   = ""
    alert_violation_secs:  float = 30.0   # fire alert after this many seconds of continuous violation

    # Logging
    log_level: str = "INFO"


settings = Settings()
