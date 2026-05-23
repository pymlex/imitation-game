import time

import httpx

from config_loader import require_env
from schemas import DetectRequest, DetectResponse


def wait_for_detector(poll_s: float = 2.0) -> None:
    """Poll the detector health endpoint until the service is ready."""
    base_url = require_env("DETECTOR_API_URL")
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        while True:
            response = client.get("/health")
            if response.status_code == 200:
                return
            time.sleep(poll_s)


def detect_logits(texts: list[str]) -> list[float]:
    """Call the local detector API and return logits."""
    base_url = require_env("DETECTOR_API_URL")
    payload = DetectRequest(texts=texts)

    with httpx.Client(base_url=base_url, timeout=None) as client:
        response = client.post("/detect", json=payload.model_dump())
        response.raise_for_status()
        data = DetectResponse(**response.json())

    return data.logits
