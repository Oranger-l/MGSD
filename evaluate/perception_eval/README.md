# Perception Evaluation

Evaluates the cold-start perception SFT model on VSP-Tasks image questions.

Default dataset:

```text
LlamaFactory/data/vsp_tasks_perception_sft.json
```

Default task levels:

- FrozenLake: 3-8
- Maze: 3-6
- MiniBehaviour: 5-6

Start vLLM:

```bash
MODEL_PATH=models/ckpts/Qwen3-VL-8B-VSP-Tasks-Perception-SFT-Final \
  bash evaluate/perception_eval/vlm.sh
```

Run a smoke eval:

```bash
SAMPLES_PER_LEVEL=2 bash evaluate/perception_eval/run_vsp_perception_eval.sh
```

Run the default eval:

```bash
bash evaluate/perception_eval/run_vsp_perception_eval.sh
```

Useful overrides:

```bash
TASKS="maze minibehaviour" bash evaluate/perception_eval/run_vsp_perception_eval.sh
BASE_URLS="http://localhost:8000/v1" REPLICAS=1 WORKERS=8 \
  bash evaluate/perception_eval/run_vsp_perception_eval.sh
```

Default vLLM image processor settings are `min_pixels=65536` and `max_pixels=262144`.
