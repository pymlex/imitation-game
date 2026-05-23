import json
from pathlib import Path

import numpy as np
import pandas as pd

from config_loader import env_int, load_env_file, require_env
from data_topics import load_topics_file
from detector_client import detect_logits, wait_for_detector
from generator_client import generate_essays, wait_for_generator


def evaluate_prompt_on_topics(
    system_prompt: str,
    topics: list[str],
    label: str,
    output_dir: Path,
) -> dict[str, float]:
    """
    Generate one essay per topic, score with the detector, persist artefacts.

    Reward used in evolution:
        R = - (1/N) * sum_i logit_i
    """
    load_env_file("evolution")
    max_new_tokens = env_int("GENERATOR_MAX_NEW_TOKENS", 300)

    wait_for_generator()
    wait_for_detector()

    output_dir.mkdir(parents=True, exist_ok=True)

    essays = generate_essays(system_prompt, topics, max_new_tokens)
    logits = detect_logits(essays)

    logits_arr = np.asarray(logits, dtype=np.float64)
    mean_logit = float(logits_arr.mean())
    reward = -mean_logit

    records = [
        {"topic": topic, "essay": essay, "logit": float(logit)}
        for topic, essay, logit in zip(topics, essays, logits)
    ]

    pd.DataFrame(records).to_csv(output_dir / f"{label}_essays.csv", index=False)

    logits_payload = {
        "label": label,
        "system_prompt": system_prompt,
        "n_topics": len(topics),
        "mean_logit": mean_logit,
        "reward": reward,
        "logits": logits,
    }
    (output_dir / f"{label}_logits.json").write_text(
        json.dumps(logits_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "mean_logit": mean_logit,
        "reward": reward,
        "frac_pred_human": float((logits_arr < 0.0).mean()),
        "frac_pred_ai": float((logits_arr >= 0.0).mean()),
    }
    (output_dir / f"{label}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metrics


def run_full_dataset_eval(
    system_prompt: str,
    topics_path: Path,
    label: str,
    results_root: Path,
) -> Path:
    """Evaluate a prompt on all topics listed in topics_path."""
    topics = load_topics_file(topics_path)
    out_dir = results_root / label
    metrics = evaluate_prompt_on_topics(system_prompt, topics, label, out_dir)
    return out_dir, metrics
