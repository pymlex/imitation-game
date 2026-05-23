import json
import os
from pathlib import Path

import numpy as np

from config_loader import env_int, load_env_file, require_env
from data_topics import load_topics_file
from detector_client import detect_logits, wait_for_detector
from generator_client import generate_essays, wait_for_generator


load_env_file("evolution")

EVAL_TOPICS_PATH = Path(require_env("EVAL_TOPICS_PATH"))
MAX_NEW_TOKENS = env_int("GENERATOR_MAX_NEW_TOKENS", 300)
METRICS_DIR = Path(require_env("EVOLUTION_METRICS_DIR"))


def evaluate(program: str) -> dict:
    """
    OpenEvolve fitness: generate on 250 fixed topics, minimise mean detector logit.

    Returns:
        combined_score = - mean(logits)
    """
    topics = load_topics_file(EVAL_TOPICS_PATH)

    wait_for_generator()
    wait_for_detector()

    essays = generate_essays(program, topics, MAX_NEW_TOKENS)
    logits = detect_logits(essays)
    logits_arr = np.asarray(logits, dtype=np.float64)

    mean_logit = float(logits_arr.mean())
    combined_score = -mean_logit

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    step_path = METRICS_DIR / "eval_steps.jsonl"
    record = {
        "mean_logit": mean_logit,
        "combined_score": combined_score,
        "n_topics": len(topics),
        "prompt_len": len(program),
    }
    with step_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "combined_score": combined_score,
        "mean_logit": mean_logit,
        "reward": combined_score,
    }


if __name__ == "__main__":
    import sys

    load_env_file("evolution")
    prompt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("prompts/initial_prompt.txt")
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    print(evaluate(prompt_text))
