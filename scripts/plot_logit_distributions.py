import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import add_project_root

import pandas as pd

from config_loader import load_env_file, require_env
from plotting import load_logits_json, plot_logit_histogram


if __name__ == "__main__":
    load_env_file("evolution")
    results_root = Path(require_env("RESULTS_DIR"))

    baseline_path = results_root / "baseline_full" / "baseline_full_logits.json"
    final_path = results_root / "final_full" / "final_full_logits.json"

    baseline_logits = load_logits_json(baseline_path)
    final_logits = load_logits_json(final_path)

    summary = plot_logit_histogram(
        baseline_logits=baseline_logits,
        final_logits=final_logits,
        output_path=results_root / "logit_distribution_full.png",
    )
    summary.to_csv(results_root / "logit_distribution_summary.csv", index=False)
    print(summary.to_string(index=False))
