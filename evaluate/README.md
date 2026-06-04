# Evaluation Pipelines

This directory contains the evaluation entry points used by MGSD.

## Directory Layout

- `perception_eval/`: perception checks for the cold-start SFT model. It asks map/object recognition questions over VSP-Tasks images.
- `ckpt_eval/`: VisualPlanning action-plan evaluation for local checkpoints served by vLLM.
- `api_eval/`: the same action-plan evaluation as `ckpt_eval`, but for OpenAI-compatible closed-source API endpoints.
- `modality_gap_eval/`: optional text-vs-image modality-gap analysis using the same task simulator.

All scripts accept environment-variable overrides such as `MODEL_PATH`,
`DATASET_ROOT`, `FROZENLAKE_BENCH_ROOT`, `API_CONFIG`, and `BASE_URLS`.

## Perception SFT Eval

Start local vLLM replicas:

```bash
bash evaluate/perception_eval/vlm.sh
```

Run the unified FrozenLake/Maze/MiniBehaviour perception eval:

```bash
bash evaluate/perception_eval/run_vsp_perception_eval.sh
```

## Local Checkpoint Bench Eval

Start local vLLM replicas:

```bash
MODEL_PATH=models/ckpts/<your-merged-checkpoint> \
  bash evaluate/ckpt_eval/serve_ckpt_8x.sh
```

Run VisualPlanning evaluation:

```bash
bash evaluate/ckpt_eval/run_ckpt_eval.sh
```

Set `SAMPLES_PER_TASK=0` for all selected samples, or a positive value to limit runtime.

## Closed-Source API Eval

Configure either `api_config_files/api_config_openai.json` or environment variables:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
```

Then run:

```bash
bash evaluate/api_eval/run_api_eval.sh
```

Use `SAMPLES_PER_TASK=0` for the full benchmark.

## Outputs

Generated logs, cached rendered images, JSONL results, and summaries are ignored by Git and can be regenerated.
