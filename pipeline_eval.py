import json
from pathlib import Path

import numpy as np
import pandas as pd

from config_loader import env_int, load_env_file, require_env
from data_topics import load_topics_file
from detector_client import detect_logits, wait_for_detector
from generator_client import generate_essays, wait_for_generator


def expand_topics_with_reps(topics: list[str], reps: int) -> list[tuple[str, int]]:
    """Repeat each topic ``reps`` times for independent generations."""
    return [(topic, rep_idx) for topic in topics for rep_idx in range(reps)]


def evaluate_prompt_on_topics(
    system_prompt: str,
    topics: list[str],
    label: str,
    output_dir: Path,
) -> dict[str, float]:
    """
    Generate ``reps`` essays per topic, score each with the detector, persist artefacts.

    Sample count is ``len(topics) * reps`` with ``EVAL_REPS_PER_TOPIC`` from the environment.
    """
    load_env_file("evolution")
    max_new_tokens = env_int("GENERATOR_MAX_NEW_TOKENS", 300)
    reps = env_int("EVAL_REPS_PER_TOPIC", 5)

    wait_for_generator()
    wait_for_detector()

    output_dir.mkdir(parents=True, exist_ok=True)

    expanded = expand_topics_with_reps(topics, reps)
    topic_list = [topic for topic, _ in expanded]

    essays = generate_essays(system_prompt, topic_list, max_new_tokens)
    logits = detect_logits(essays)

    logits_arr = np.asarray(logits, dtype=np.float64)
    mean_logit = float(logits_arr.mean())
    reward = -mean_logit
    pred_human = logits_arr < 0.0

    records = [
        {
            "topic": topic,
            "rep": int(rep_idx),
            "essay": essay,
            "logit": float(logit),
            "pred_human": int(logit < 0.0),
        }
        for (topic, rep_idx), essay, logit in zip(expanded, essays, logits)
    ]

    pd.DataFrame(records).to_csv(output_dir / f"{label}_essays.csv", index=False)

    logits_payload = {
        "label": label,
        "system_prompt": system_prompt,
        "n_topics": len(topics),
        "reps_per_topic": reps,
        "n_samples": len(topic_list),
        "mean_logit": mean_logit,
        "reward": reward,
        "logits": logits,
        "pred_human": pred_human.astype(int).tolist(),
    }
    (output_dir / f"{label}_logits.json").write_text(
        json.dumps(logits_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "n_topics": len(topics),
        "reps_per_topic": reps,
        "n_samples": len(topic_list),
        "mean_logit": mean_logit,
        "reward": reward,
        "frac_pred_human": float(pred_human.mean()),
        "frac_pred_ai": float((~pred_human).mean()),
        "n_pred_human": int(pred_human.sum()),
        "n_pred_ai": int((~pred_human).sum()),
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
) -> tuple[Path, dict[str, float]]:
    """Evaluate a prompt on all topics listed in topics_path with multiple reps per topic."""
    topics = load_topics_file(topics_path)
    out_dir = results_root / label
    metrics = evaluate_prompt_on_topics(system_prompt, topics, label, out_dir)
    return out_dir, metrics
