import json
import os
import shutil
from pathlib import Path

import nest_asyncio
import yaml

from config_loader import load_env_file, require_env


nest_asyncio.apply()


def write_runtime_config(template_path: Path, output_path: Path) -> None:
    """Inject API credentials from the environment into the evolution YAML."""
    text = template_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)

    api_key = require_env("OPENAI_API_KEY")
    api_base = require_env("OPENAI_API_BASE")
    llm_model = require_env("LLM_MODEL")

    payload["llm"]["models"][0]["name"] = llm_model
    payload["llm"]["models"][0]["api_base"] = api_base
    payload["llm"]["models"][0]["api_key"] = api_key

    payload["operators"]["mutation"]["llm_model"] = llm_model
    payload["operators"]["crossover"]["llm_model"] = llm_model

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.dump(payload, default_flow_style=False), encoding="utf-8")


if __name__ == "__main__":
    load_env_file("evolution")

    experiment_dir = Path(require_env("EXPERIMENT_DIR"))
    openevolve_dir = Path(require_env("OPENEVOLVE_OUTPUT_DIR"))
    metrics_dir = Path(require_env("EVOLUTION_METRICS_DIR"))

    experiment_dir.mkdir(parents=True, exist_ok=True)
    openevolve_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    initial_src = Path("prompts/initial_prompt.txt")
    initial_dst = experiment_dir / "initial_prompt.txt"
    shutil.copy(initial_src, initial_dst)

    runtime_config = experiment_dir / "config_evolution_runtime.yaml"
    write_runtime_config(Path("config/config_evolution.yaml"), runtime_config)

    os.environ["OPENAI_API_KEY"] = require_env("OPENAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = require_env("OPENAI_API_BASE")
    os.environ["LLM_MODEL"] = require_env("LLM_MODEL")

    from openevolve.api import run_evolution
    from openevolve.config import load_config

    config = load_config(str(runtime_config))
    generations = config.get("evolution", {}).get("generations", config.get("max_iterations", 100))

    result = run_evolution(
        initial_program=str(initial_dst),
        evaluator="evaluator.py",
        config=config,
        iterations=generations,
        output_dir=str(openevolve_dir),
        cleanup=False,
    )

    print("best_score:", result.best_score)

    best_out = experiment_dir / "best_prompt_evolved.txt"
    if hasattr(result, "best_code") and result.best_code:
        best_out.write_text(result.best_code, encoding="utf-8")
        print("saved:", best_out)

    summary = {
        "best_score": float(result.best_score),
        "generations": int(generations),
        "best_prompt_path": str(best_out),
    }
    (experiment_dir / "evolution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
