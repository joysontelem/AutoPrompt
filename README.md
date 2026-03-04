# AutoPrompt — Benchmark Optimization

Iteratively optimizes a prompt using a fixed annotated dataset.

Given a CSV with `text` and `annotation` columns, this tool runs a loop of
**predict → evaluate → refine** to find the best-performing prompt.

## Setup

```bash
pip install -r requirements.txt
```

Configure your LLM API keys in `config/llm_env.yml`.

## Usage

```bash
python run_benchmark_optimization.py \
    --dataset examples/benchmark_dataset.csv \
    --prompt "Is this movie review positive? Answer Yes or No." \
    --task_description "Classify movie reviews as positive or negative." \
    --labels Yes No \
    --num_steps 10 \
    --output results.json
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--dataset` | Yes | — | Path to CSV with `text` and `annotation` columns |
| `--prompt` | No | (interactive) | Initial prompt to optimize |
| `--task_description` | No | (interactive) | Description of the classification task |
| `--labels` | No | Yes No | Label schema |
| `--num_steps` | No | 10 | Number of optimization iterations |
| `--output` | No | benchmark_results.json | Output JSON file |
| `--config` | No | config/config_benchmark.yml | Configuration file |

### Dataset Format

```csv
text,annotation
"The movie was absolutely fantastic!",Yes
"Waste of time and money.",No
```

See `examples/benchmark_dataset.csv` for a full example.

## How It Works

1. **Load** the annotated dataset from CSV
2. **Predict** — run the current prompt against all samples using the predictor LLM
3. **Evaluate** — compare predictions to ground-truth annotations (accuracy)
4. **Refine** — use a meta-prompt LLM to generate an improved prompt based on error analysis
5. **Repeat** until budget, patience, or iteration limit is reached

## Project Structure

```
├── run_benchmark_optimization.py   # Entry point
├── benchmark_optimizer.py          # Optimization loop
├── utils/
│   ├── config.py                   # YAML + LLM configuration
│   └── llm_chain.py                # LangChain wrappers
├── dataset/
│   └── base_dataset.py             # Dataset storage
├── eval/
│   ├── evaluator.py                # Scoring and error extraction
│   └── eval_utils.py               # Eval helper functions
├── estimator/
│   ├── __init__.py                 # Estimator factory
│   └── estimator_llm.py            # LLM-based predictor
├── config/
│   ├── config_benchmark.yml        # Benchmark optimization config
│   └── llm_env.yml                 # LLM API keys
├── prompts/
│   ├── meta_prompts_classification/ # Meta-prompts for refinement
│   └── predictor_completion/        # Prediction prompt templates
└── examples/
    └── benchmark_dataset.csv        # Sample dataset
```

## License

[Apache License 2.0](LICENSE)

