import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_logits_json(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["logits"], dtype=np.float64)


def plot_logit_histogram(
    baseline_logits: np.ndarray,
    final_logits: np.ndarray,
    output_path: Path,
    bins: int = 40,
) -> pd.DataFrame:
    """
    Plot baseline and final logit distributions on shared axes.

    Returns:
        Summary statistics as a one-row DataFrame.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        baseline_logits,
        bins=bins,
        alpha=0.55,
        label="baseline",
        density=True,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.hist(
        final_logits,
        bins=bins,
        alpha=0.55,
        label="evolved",
        density=True,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("detector logit")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "baseline_mean": float(baseline_logits.mean()),
                "baseline_std": float(baseline_logits.std()),
                "final_mean": float(final_logits.mean()),
                "final_std": float(final_logits.std()),
                "delta_mean": float(final_logits.mean() - baseline_logits.mean()),
            }
        ]
    )
    return summary
