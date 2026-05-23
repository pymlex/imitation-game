import time

import httpx

from config_loader import require_env
from schemas import GenerateRequest, GenerateResponse


def wait_for_generator(poll_s: float = 2.0) -> None:
    """Poll the generator health endpoint until the service is ready."""
    base_url = require_env("GENERATOR_API_URL")
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        while True:
            response = client.get("/health")
            if response.status_code == 200:
                return
            time.sleep(poll_s)


def generate_essays(
    system_prompt: str,
    topics: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Call the local generator API and return essay texts."""
    base_url = require_env("GENERATOR_API_URL")
    payload = GenerateRequest(
        system_prompt=system_prompt,
        topics=topics,
        max_new_tokens=max_new_tokens,
    )

    with httpx.Client(base_url=base_url, timeout=None) as client:
        response = client.post("/generate", json=payload.model_dump())
        response.raise_for_status()
        data = GenerateResponse(**response.json())

    return data.essays
