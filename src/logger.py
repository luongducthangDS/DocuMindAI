import sys
from pathlib import Path

from loguru import logger

from src.config import get_settings

# Windows console defaults to cp1252 which cannot encode Vietnamese or emoji.
# Reconfigure before loguru sets up its stderr sink so all log output is UTF-8.
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def setup_logger() -> None:
    settings = get_settings()
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    # Console — human-readable in dev, JSON in prod
    if settings.environment == "development":
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
            level="DEBUG",
            colorize=True,
        )
    else:
        logger.add(
            sys.stderr,
            format="{time:YYYY-MM-DDTHH:mm:ssZ} | {level} | {name}:{line} | {message}",
            level="INFO",
            serialize=True,  # structured JSON
        )

    # Rotating file log — never log secrets
    logger.add(
        settings.logs_dir / "documind_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        compression="gz",
        level="INFO",
        format="{time:YYYY-MM-DDTHH:mm:ssZ} | {level} | {name}:{line} | {message}",
        filter=_redact_secrets,
    )


def _redact_secrets(record: dict) -> bool:
    """Strip API keys and passwords from log messages."""
    msg = record.get("message", "")
    for keyword in ("api_key", "password", "token", "secret", "Bearer "):
        if keyword.lower() in msg.lower():
            record["message"] = "[REDACTED — contains sensitive data]"
            break
    return True


setup_logger()
