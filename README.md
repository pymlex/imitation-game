# Imitation Game — Spanish Essay Prompt Evolution

Evolve a **system prompt** for [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) so that generated Spanish literary essays receive lower logits from [danibor/oculus-v2.0-multilingual](https://huggingface.co/danibor/oculus-v2.0-multilingual).

| Component | Role |
|-----------|------|
| Generator API | Qwen, batched essays, max 300 new tokens, greedy decoding |
| Detector API | Oculus, batched logits |
| OpenEvolve + ZvenoAI | Small prompt mutations, island evolution |
| Evaluator | Fitness on 250 fixed topics during evolution |

Dataset: [pymlex/spanish-essay-topics](https://huggingface.co/datasets/pymlex/spanish-essay-topics) — 558 topics.

## Reward

\[
R(p) = -\frac{1}{N}\sum_{i=1}^{N} f(x_i), \quad x_i = G(p, t_i)
\]

Higher `combined_score` in OpenEvolve corresponds to lower mean detector logit.

## Project layout

```
├── generator_api.py      # FastAPI — Qwen
├── detector_api.py       # FastAPI — Oculus
├── evaluator.py          # OpenEvolve fitness (250 topics)
├── pipeline_eval.py      # Full-dataset evaluation
├── config/config_evolution.yaml
├── env/*.env.example
├── scripts/
│   ├── prepare_topics.py
│   ├── run_baseline_eval.py
│   ├── run_evolution.py
│   ├── run_final_eval.py
│   └── plot_logit_distributions.py
├── prompts/initial_prompt.txt
└── results/              # metrics, plots, evolved prompt
```

See [COLAB.md](COLAB.md) for the step-by-step Colab workflow.

## Quick start

```bash
pip install -r requirements.txt
cp env/*.env.example env/
# edit env/evolution.env — set OPENAI_API_KEY

python main.py
```

Run steps in the order printed by `main.py`. Start `generator_api.py` and `detector_api.py` before evaluation or evolution.

## Results (fill after runs)

| Stage | Mean logit | Reward \(-\bar{\ell}\) | Fraction predicted human |
|-------|------------|------------------------|---------------------------|
| Baseline (558 topics) | — | — | — |
| Evolved (558 topics) | — | — | — |

Plot: `results/logit_distribution_full.png`

Summary table: `results/logit_distribution_summary.csv`

Evolution trace: `results/evolution_metrics/eval_steps.jsonl`

## References

- Essay topics: https://huggingface.co/datasets/pymlex/spanish-essay-topics
- Generator: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Detector: https://huggingface.co/danibor/oculus-v2.0-multilingual
- OpenEvolve: https://github.com/algorithmicsuperintelligence/openevolve
