# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Research project (CS546 — Parallel Computing, IIT Spring 2026) evaluating whether LLMs can diagnose HPC workflow I/O bottlenecks from structured execution snapshots, and how much prompting strategy affects accuracy. The core experiment compares five strategies across 11 bottleneck classes on a synthetic dataset of 500 labeled snapshots.

## Running the Pipeline

```bash
# Generate synthetic snapshots (only needed if regenerating the dataset)
python src/generate_snapshots.py

# Run full evaluation — OpenAI (all 5 strategies in sequence)
python src/setup_openai.py

# Run full evaluation — Anthropic / Haiku-4.5
python src/setup_anthropic.py

# Generate all plots (run after evaluation produces result JSONs)
python src/plot_confusion_matrices.py
python src/plot_accByGt_conf_costAccToken.py
python src/plot_boxplot_per_class_latency.py
```

All plots are saved to `data/plots/`. Results land in `data/results/` as JSON files named `{model}_{strategy}_{timestamp}.json`.

## Environment Setup

Create a `.env` file at the project root with **one** of the following:

```bash
# Option A — official Anthropic API
ANTHROPIC_API_KEY=sk-ant-...

# Option B — proxy URL (no API key needed)
ANTHROPIC_BASE_URL=https://proxy-url.com

# For OpenAI
OPENAI_API_KEY=sk-proj-...
```

## Architecture

### Pipeline Flow

```
data/snapshots/snap_*.json (500 files)
        ↓
src/setup_openai.py  OR  src/setup_anthropic.py
        ↓  (5 strategies × 500 snapshots)
data/results/{model}_{strategy}_{ts}.json
        ↓
src/plot_*.py  →  data/plots/
```

### Snapshot Schema

Each snapshot in `data/snapshots/` is a JSON object with five metric groups (`workflow`, `stage`, `execution`, `io_metrics`, `resource_utilization`, `storage`) plus a ground-truth `annotation` block. The annotation is stripped before sending to the LLM and used only for scoring.

### Evaluation Scripts (`setup_openai.py` / `setup_anthropic.py`)

Both files are structurally identical. Key functions:

- **`build_prompt(snapshot, strategy)`** — assembles the user prompt for one of 5 strategies: `rule_based`, `classify_only`, `zero_shot`, `few_shot`, `chain_of_thought`
- **`diagnose(snapshot, model, strategy)`** — calls the LLM (or runs the rule-based classifier) and returns a dict with `bottleneck`, `confidence`, `key_signals`, `explanation`, `duration_s`, `token_usage`
- **`evaluate(snapshots, model, strategy)`** — main loop; skips few-shot example snapshots from the `few_shot` evaluation to prevent data leakage
- **`_compute_summary(results)`** — computes accuracy, macro F1, per-class P/R/F1, and token aggregates

The `classify_only` strategy uses strict JSON schema enforcement (`output_config` on Anthropic, `response_format` with `json_schema` on OpenAI) so the model emits only a `bottleneck` field. All other LLM strategies rely on prompt instructions to return a 4-field JSON object.

### Prompting Strategies

| Strategy | Description |
|---|---|
| `rule_based` | No LLM — priority-ordered heuristic rules on metric thresholds |
| `classify_only` | Class names only, no descriptions |
| `zero_shot` | Full class descriptions in the prompt |
| `few_shot` | 5 hand-picked examples + descriptions (examples excluded from evaluation) |
| `chain_of_thought` | 11-step reasoning template before concluding |

The large accuracy gap between `classify_only` (~45%) and `zero_shot` (~95%) is the intended experimental result — it quantifies the value of domain-specific descriptions for novel HPC concepts.

### Plot Scripts

- **`plot_confusion_matrices.py`** — one PNG per strategy; reads `data/results/*.json`, outputs to `data/plots/`
- **`plot_accByGt_conf_costAccToken.py`** — accuracy-by-GT-confidence bar chart and cost-vs-accuracy Pareto scatter
- **`plot_boxplot_per_class_latency.py`** — per-class latency boxplots (LLM strategies only; rule-based excluded)

All plot scripts auto-discover result JSONs via `glob` — adding a new result file is enough for it to be included.

## Key Constants

The 11 bottleneck classes are defined in `BOTTLENECK_CLASSES` in both setup scripts and as `CLASSES` in the plot scripts — they must stay in sync. The 5 few-shot examples are defined in `FEW_SHOT_EXAMPLES` and their IDs are excluded from few-shot evaluation via `FEW_SHOT_IDS`.
