import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from config_loader import env_int, require_env


def project_root() -> Path:
    return Path(__file__).resolve().parent


def data_dir() -> Path:
    path = project_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_all_topics() -> list[str]:
    """Load all 558 essay topics from Hugging Face."""
    dataset = load_dataset("pymlex/spanish-essay-topics")
    split_name = next(iter(dataset.keys()))
    rows = dataset[split_name]
    return [row["topic"] for row in rows]


def save_topics(path: Path, topics: list[str]) -> None:
    path.write_text(json.dumps(topics, ensure_ascii=False, indent=2), encoding="utf-8")


def load_topics_file(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_topic_splits() -> dict[str, Path]:
    """
    Persist full dataset topics and a fixed 250-topic evaluation subset.

    Returns:
        Mapping with keys all_topics and eval_topics_250.
    """
    seed = env_int("TOPICS_SEED", 42)
    eval_count = env_int("EVAL_TOPIC_COUNT", 250)

    all_path = data_dir() / "all_topics.json"
    eval_path = data_dir() / "eval_topics_250.json"

    all_topics = load_all_topics()
    save_topics(all_path, all_topics)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_topics), size=eval_count, replace=False)
    eval_topics = [all_topics[int(i)] for i in np.sort(indices)]
    save_topics(eval_path, eval_topics)

    return {"all_topics": all_path, "eval_topics_250": eval_path}


def eval_topics_path() -> Path:
    return Path(require_env("EVAL_TOPICS_PATH"))


def all_topics_path() -> Path:
    return Path(require_env("ALL_TOPICS_PATH"))
