import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import add_project_root

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from config_loader import load_env_file, require_env
from plotting import load_logits_json


def load_pred_human(path: Path) -> np.ndarray:
    """Load per-sample human predictions from a logits JSON artefact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "pred_human" in payload:
        return np.asarray(payload["pred_human"], dtype=np.int64)
    logits = np.asarray(payload["logits"], dtype=np.float64)
    return (logits < 0.0).astype(np.int64)


if __name__ == "__main__":
    load_env_file("evolution")
    results_root = Path(require_env("RESULTS_DIR"))

    baseline_logits_path = results_root / "baseline_full" / "baseline_full_logits.json"
    final_logits_path = results_root / "final_full" / "final_full_logits.json"

    baseline_logits = load_logits_json(baseline_logits_path)
    final_logits = load_logits_json(final_logits_path)

    baseline_human = load_pred_human(baseline_logits_path)
    final_human = load_pred_human(final_logits_path)

    u_stat, u_pvalue = mannwhitneyu(
        baseline_logits,
        final_logits,
        alternative="two-sided",
    )

    contingency = np.array(
        [
            [int(baseline_human.sum()), int((1 - baseline_human).sum())],
            [int(final_human.sum()), int((1 - final_human).sum())],
        ]
    )
    chi2_stat, chi2_pvalue, _, _ = chi2_contingency(contingency)

    summary = {
        "n_samples_baseline": int(baseline_logits.size),
        "n_samples_final": int(final_logits.size),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_pvalue": float(u_pvalue),
        "baseline_mean_logit": float(baseline_logits.mean()),
        "final_mean_logit": float(final_logits.mean()),
        "baseline_human_rate": float(baseline_human.mean()),
        "final_human_rate": float(final_human.mean()),
        "baseline_n_human": int(baseline_human.sum()),
        "final_n_human": int(final_human.sum()),
        "chi2_statistic": float(chi2_stat),
        "chi2_pvalue": float(chi2_pvalue),
        "contingency_human_ai": contingency.tolist(),
    }

    out_path = results_root / "significance_tests.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    table = pd.DataFrame([summary])
    print(table.to_string(index=False))
    print(f"written: {out_path}")
