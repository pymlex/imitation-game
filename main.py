"""
Entry point: print the recommended Colab execution order.
"""
from pathlib import Path


STEPS = [
    "pip install -r requirements.txt",
    "cp env/generator.env.example env/generator.env",
    "cp env/detector.env.example env/detector.env",
    "cp env/evolution.env.example env/evolution.env",
    "python scripts/prepare_topics.py",
    "python generator_api.py",
    "python detector_api.py",
    "python scripts/run_baseline_eval.py",
    "python scripts/run_evolution.py",
    "python scripts/run_final_eval.py",
    "python scripts/plot_logit_distributions.py",
]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(f"project root: {root}\n")
    for index, step in enumerate(STEPS, start=1):
        print(f"{index}. {step}")
