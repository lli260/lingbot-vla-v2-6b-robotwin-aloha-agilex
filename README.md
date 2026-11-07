---
license: apache-2.0
base_model: robbyant/lingbot-vla-v2-6b
pipeline_tag: robotics
datasets:
  - TianxingChen/RoboTwin2.0
tags:
  - vision-language-action
  - vla
  - robotwin
  - aloha-agilex
  - lingbot-vla
---

# LingBot-VLA 2.0 (6B) — RoboTwin 2.0 / Aloha-AgileX fine-tune

Full-parameter supervised fine-tune of [`robbyant/lingbot-vla-v2-6b`](https://huggingface.co/robbyant/lingbot-vla-v2-6b)
on the public **RoboTwin 2.0 Aloha-AgileX clean** split (50 tasks × 50 demonstrations).
The checkpoint is exported in the standard `hf_ckpt` layout with 1,708 tensors across 3
safetensors shards (BF16, 12.77 GB) and is intended to be used with the
[LingBot-VLA-v2](https://github.com/Robbyant/lingbot-vla-v2) inference stack.

> **Weights (11.89 GB, not in this repo):** [https://huggingface.co/liwy26/lingbot-vla-v2-6b-robotwin-aloha-agilex](https://huggingface.co/liwy26/lingbot-vla-v2-6b-robotwin-aloha-agilex)
> **Code:** this GitHub repository (`lli260/lingbot-vla-v2-6b-robotwin-aloha-agilex`)

This GitHub repo is **code + configs only**. The three `model-*.safetensors` shards are replaced by placeholder files under [`weights/`](weights/). Download them with:

```bash
# option 1: helper script
bash scripts/download_weights.sh          # Linux / macOS
# powershell -File scripts/download_weights.ps1   # Windows

# option 2: hf CLI (recommended)
hf download liwy26/lingbot-vla-v2-6b-robotwin-aloha-agilex \
  --include "model-*.safetensors" --local-dir .
```

Place the downloaded shards at the **repository root** (next to `config.json`), then verify:

```bash
python examples/checkpoint_check.py .
```

## Model at a glance

| Item | Value |
|---|---|
| Base model | `robbyant/lingbot-vla-v2-6b` (Apache-2.0) |
| Parameters | 6B (Qwen3-VL-4B backbone + action expert) |
| Inputs | 3 RGB views (`camera_top`, `camera_wrist_left`, `camera_wrist_right`; simulator keys `cam_high`, `cam_left_wrist`, `cam_right_wrist`) + 14-D robot state + language instruction |
| Output | 50-step action chunk (14-D Aloha-AgileX joint/gripper actions) |
| Denoising steps | 10 |
| Precision | BF16 |
| Tensors / shards | 1,708 / 3 |
| Repository size | ≈ 12.77 GB |

## Results

Self-reported success rate on the 50 RoboTwin 2.0 Aloha-AgileX tasks.

| Setting | Success | Episodes |
|---|---:|---:|
| `clean` | **50.50%** | 505 / 1000 |
| `randomized` | **27.40%** | 274 / 1000 |
| Overall | **38.95%** | 779 / 2000 |

**Protocol and caveats.** 50 tasks × 2 settings × 20 episodes = 2,000 episodes, no
truncation; one contiguous seed block (100000–100019) per task/setting; BF16 inference.
This is a **20-episode-per-task self-test**, not the 100-episode-per-task protocol used
in RoboTwin publications, so the numbers above are **not directly comparable** to
published 100-episode results. The five-episode subset of the 20-episode run scored 45.80%
overall, while a separate five-episode run scored 41.60%, and 69/100 task-setting pairs
differed by 1–2 episodes — expect a few points of run-to-run variance at this sample size.

The model was trained on **clean demonstrations only**. It handles light scene variation,
but a larger drop under strong visual randomization is expected.

## Repository layout

```text
.
├── README.md                          # this model card
├── LICENSE                            # Apache-2.0
├── .gitignore                         # excludes real *.safetensors from git
├── SHA256SUMS                         # checksums for weights and configs
├── config.json                        # model config (training-time fields preserved)
├── lingbotvla_cli.yaml                # ★ authoritative inference config
├── model.safetensors.index.json       # tensor -> shard index
├── preprocessor_config.json
├── tokenizer files                    # tokenizer.json / vocab.json / merges.txt / ...
├── weights/                           # ★ placeholders only (no real weights in git)
│   ├── model-00001-of-00003.safetensors.placeholder
│   ├── model-00002-of-00003.safetensors.placeholder
│   └── model-00003-of-00003.safetensors.placeholder
├── scripts/
│   ├── download_weights.sh            # fetch shards from Hugging Face
│   └── download_weights.ps1
├── assets/
│   └── norm_stats/robotwin.json       # action/state normalization statistics
├── configs/
│   └── robot_configs/robotwin.yaml    # state/action/image key mapping
├── deploy/                            # inference loader + WebSocket policy server
│   ├── lingbot_vla_v2_policy.py
│   ├── websocket_policy_server.py
│   ├── websocket_client_policy.py
│   ├── image_tools.py
│   ├── msgpack_numpy.py
│   └── utils.py
├── examples/
│   ├── README.md
│   └── checkpoint_check.py            # offline weight integrity check
└── docs/
    └── EVALUATION.md                  # evaluation protocol details

# After download (local only, git-ignored):
├── model-00001-of-00003.safetensors   # ~4.65 GB  ← from HF
├── model-00002-of-00003.safetensors   # ~4.65 GB  ← from HF
└── model-00003-of-00003.safetensors   # ~2.57 GB  ← from HF
```

**Hosts:** Hugging Face holds the weights; GitHub holds this code-only tree.
`.gitignore` keeps `*.safetensors` out of git. Uploads to HF via `hf upload` ignore
`.gitignore`, so the full checkpoint (including shards) lives on Hugging Face.

## Quick start

### 1. Install the inference stack

```bash
git clone https://github.com/Robbyant/lingbot-vla-v2
cd lingbot-vla-v2
pip install -e .
# plus the inference-only extras listed in the upstream README
```

### 2. Fetch the base tokenizer/processor

```bash
export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct
# e.g. huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir $QWEN3VL_PATH
```

### 3. Launch the policy server from the checkpoint root

`lingbotvla_cli.yaml`, `assets/`, and `configs/` use **relative paths resolved against the
current working directory**, so run the server from this directory:

```bash
cd /path/to/lingbot-vla-v2-6b-robotwin-aloha-agilex
export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct
export PYTHONPATH=/path/to/lingbot-vla-v2:$PYTHONPATH

python -m deploy.lingbot_vla_v2_policy \
  --model_path . \
  --use_length 50 \
  --use_bf16 True --use_fp32 False \
  --port 8006
```

For FP32 parity checks (upstream reports that BF16 and FP32 can differ materially):

```bash
python -m deploy.lingbot_vla_v2_policy \
  --model_path . --use_length 50 \
  --use_bf16 False --use_fp32 True --port 8006
```

`deploy/websocket_client_policy.py` contains a reference client. The RoboTwin simulator
launcher is documented in the upstream repository
(`experiment/robotwin/start_robotwin_infer_and_eval.sh`); point its `--model_path` at this
directory.

### 4. Verify the checkpoint offline

```bash
python examples/checkpoint_check.py .
sha256sum -c SHA256SUMS          # Linux / WSL
certutil -hashfile <file> SHA256 # Windows
```

## Inference notes

- **`lingbotvla_cli.yaml` is authoritative, not `config.json`.** The loader builds the
  model config from the YAML with `post_training: false` and `align_params: {}`.
  `config.json` intentionally keeps training-time fields (`post_training: true` and the
  teacher/alignment settings, with their checkpoint paths cleared). Do **not** load this
  checkpoint with a plain `AutoModel`; use the provided `deploy/` loader.
- **Teacher weights are not included.** The training run used depth / future-depth /
  future-video distillation. The exported file contains the student plus training-only
  alignment tensors; the loader detects and drops those keys, which cannot be used for
  inference.
- **Not a LoRA adapter.** All parameters are merged into the 3 shards (`use_lora: false`
  in the YAML). There is no `adapter_model.safetensors`.
- **Normalization statistics are included** at `assets/norm_stats/robotwin.json` and are
  referenced from the YAML.
- **Three cameras are required**, with the exact keys in
  `configs/robot_configs/robotwin.yaml`.
- **Image size** 256×256; **action chunk** 50; **state/action dim** 14 for Aloha-AgileX.

## Training recipe

| Stage | Details |
|---|---|
| Initialization | `robbyant/lingbot-vla-v2-6b` → full-parameter SFT, 3,500 steps on clean data |
| Continuation | +1,200 steps, same data, color augmentation applied to the **current-view images only** (brightness 0.7–1.3, contrast 0.6–1.4, saturation 0.5–1.5); future-frame distillation targets stay un-augmented |
| Data | RoboTwin 2.0 Aloha-AgileX `clean`, 50 tasks × 50 episodes = 2,500 episodes; no randomized data, no external data |
| Batch | global 224 (micro 16 × 7 GPUs × grad-accum 2) |
| Precision / parallelism | BF16, FSDP2 |
| Optimizer | `adamw`, lr 2e-5 → 2e-6, cosine schedule, 5% warmup, weight decay 0 |
| Seed | 42 |
| Final loss | 0.0318 at step 1,200 |
| Wall clock | ≈ 10 h 49 min on 7× AMD Radeon W7900D (ROCm 7.2.1) |
| Export | DCP → `hf_ckpt`, 1,708 tensors / 3 shards / 12.77 GB |

The released checkpoint is the step-1,200 export of the continuation stage.

## Evaluation setup

Success is measured by the RoboTwin 2.0 task success predicates, 20 episodes per task per
setting, with episode seeds and initial states generated by RoboTwin itself. Aggregate
numbers only are reported here; see [`docs/EVALUATION.md`](docs/EVALUATION.md) for the
protocol, hardware, and variance notes.

## Limitations

- Trained on clean episodes only; performance under aggressive visual randomization is
  substantially lower (see Results).
- Single robot embodiment (Aloha-AgileX) and a single action space; other robots require
  re-training or a different robot config and norm stats.
- Language instructions follow the RoboTwin task descriptions; free-form instructions were
  not part of training.
- Self-reported numbers were produced with one seed block and a reduced episode budget;
  treat them as indicative rather than as a benchmark claim.

## License and attribution

Released under **Apache-2.0**, following the base model license. This repository is a
fine-tune of [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2) by Robbyant and
contains no teacher or third-party backbone weights (Qwen3-VL, MoGe, and the depth/video
teachers are training-time dependencies only). Training and evaluation used
[RoboTwin 2.0](https://robotwin-platform.github.io) and its public
[dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0); those are covered by
their own licenses. `deploy/` is derived from the upstream Apache-2.0 inference code.

```bibtex
@misc{lingbotvla2_robotwin_aloha_agilex,
  title  = {LingBot-VLA 2.0 (6B) fine-tuned on RoboTwin 2.0 Aloha-AgileX},
  year   = {2026},
  note   = {Full-parameter SFT of robbyant/lingbot-vla-v2-6b}
}
```
