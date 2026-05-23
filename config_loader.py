import os
from pathlib import Path

from dotenv import load_dotenv


def load_env_file(env_name: str) -> None:
    """Load variables from env/<name>.env into the process environment."""
    root = Path(__file__).resolve().parent
    path = root / "env" / f"{env_name}.env"
    load_dotenv(path, override=True)


def require_env(key: str) -> str:
    """Return a required environment variable or raise."""
    value = os.environ.get(key)
    if value is None or value == "":
        raise RuntimeError(f"Missing environment variable: {key}")
    return value


def env_int(key: str, default: int) -> int:
    """Parse an integer environment variable."""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def env_float(key: str, default: float) -> float:
    """Parse a float environment variable."""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return float(raw)
