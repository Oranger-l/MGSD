# Modality Gap Evaluation

Optional analysis for comparing image-input and text-input performance on VSP tasks. It reuses the data loading and reward logic from `evaluate/ckpt_eval`.

Start a base model server:

```bash
MODEL_KIND=base MODEL_PATH=models/Qwen3-VL-8B-Instruct \
  bash evaluate/modality_gap_eval/serve_model_8x.sh
```

Run image/text conditions:

```bash
MODEL_KIND=base bash evaluate/modality_gap_eval/run_two_modalities.sh
```

Start an OPCD model server:

```bash
MODEL_KIND=opcd MODEL_PATH=models/ckpts/<your-opcd-checkpoint> \
  bash evaluate/modality_gap_eval/serve_model_8x.sh
```

Run a single condition dry run:

```bash
INPUT_MODALITY=text SAMPLES_PER_TASK=2 \
  bash evaluate/modality_gap_eval/run_condition.sh --dry-run
```

Results and rendered image caches are ignored by Git.
