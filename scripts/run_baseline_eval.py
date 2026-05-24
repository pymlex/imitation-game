import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import add_project_root

from config_loader import load_env_file, require_env
from data_topics import all_topics_path
from pipeline_eval import run_full_dataset_eval


if __name__ == "__main__":
    load_env_file("evolution")
    results_root = Path(require_env("RESULTS_DIR"))
    prompt_path = Path(require_env("EVAL_BASELINE_PROMPT_PATH"))
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    out_dir, metrics = run_full_dataset_eval(
        system_prompt=system_prompt,
        topics_path=all_topics_path(),
        label="baseline_full",
        results_root=results_root,
    )
    print(out_dir)
    print(metrics)
