# Closed-Source API Evaluation

Runs the same VisualPlanning action-plan benchmark as `evaluate/ckpt_eval`, but sends requests to an OpenAI-compatible API endpoint.

Configure credentials with either environment variables:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://your-endpoint/v1
```

or edit:

```text
api_config_files/api_config_openai.json
```

Smoke eval:

```bash
bash evaluate/api_eval/run_api_eval.sh
```

Full eval:

```bash
SAMPLES_PER_TASK=0 RESUME=1 bash evaluate/api_eval/run_api_eval.sh
```

Evaluate specific models:

```bash
MODELS="gpt-4o gemini-2.5-pro" SAMPLES_PER_TASK=0 \
  bash evaluate/api_eval/run_api_eval.sh
```

The default prompt files live in `evaluate/api_eval/prompts/`. Outputs are written to `evaluate/api_eval/results/` and ignored by Git.
