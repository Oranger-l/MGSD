# Checkpoint Evaluation

Evaluates a merged local VLM checkpoint on VisualPlanning FrozenLake, Maze, and MiniBehaviour action-plan tasks.

## Expected Bench Data

Place benchmark data under:

```text
data/DiffThinker/FrozenLake/VSP/maps/
data/VisualPlanning/dataset/
```

or override:

```bash
FROZENLAKE_BENCH_ROOT=/path/to/frozenlake/maps \
DATASET_ROOT=/path/to/VisualPlanning/dataset \
  bash evaluate/ckpt_eval/run_ckpt_eval.sh
```

## Run

Start vLLM replicas:

```bash
MODEL_PATH=models/ckpts/<your-merged-checkpoint> \
  bash evaluate/ckpt_eval/serve_ckpt_8x.sh
```

Run evaluation:

```bash
bash evaluate/ckpt_eval/run_ckpt_eval.sh
```

Use a single server:

```bash
BASE_URLS="http://localhost:8000/v1" WORKERS=8 \
  bash evaluate/ckpt_eval/run_ckpt_eval.sh
```

## Main Metrics

The summary JSON reports:

- `total`: micro average over all samples.
- `by_task`: per-task statistics.
- `by_task_level`: per-task, per-level statistics.
- `task_macro_avg_by_level`: level-balanced score per task.
- `all_tasks_macro_avg`: equal-weight task average.

`accuracy` is the task success rate. `optimal_accuracy` additionally requires the predicted plan length to match the shortest available plan.
