# Evaluation details

Benchmark: **RoboTwin 2.0**, Aloha-AgileX embodiment, the 50 public manipulation tasks.

## Protocol

| Item | Value |
|---|---|
| Tasks | 50 |
| Settings | `clean` (`demo_clean`), `randomized` (`demo_randomized`) |
| Episodes per task per setting | 20 |
| Total episodes | 2,000 |
| Episode seeds | 100000–100019 per task/setting |
| Success criterion | RoboTwin task success predicate (no manual review, no retries) |
| Truncation | none; all 2,000 episodes ran to completion |
| Inference precision | BF16 |
| Action chunk / denoising steps | 50 / 10 |
| Cameras | `camera_top`, `camera_wrist_left`, `camera_wrist_right` |

Aggregate results:

| Setting | Successes | Episodes | Success rate |
|---|---:|---:|---:|
| `clean` | 505 | 1000 | 50.50% |
| `randomized` | 274 | 1000 | 27.40% |
| Overall | 779 | 2000 | 38.95% |

## Hardware and software

- 8× AMD Radeon W7900D (48 GB), one evaluation server per GPU
- PyTorch 2.9.1 + ROCm 7.2.1
- RoboTwin 2.0 simulator, pinned revision compatible with the upstream LingBot-VLA-v2
  `experiment/robotwin` guide

## Run-to-run variance

A second evaluation of the same checkpoint reused the same episode seeds for the first
five episodes of every task. Overall success was 45.80% versus 41.60% for the shortened
run, and 69 of 100 task-setting pairs differed by 1–2 episodes. At 20 episodes per task the
per-task estimate has a standard error of roughly 11 percentage points, and the aggregate
of 1,000 episodes per setting has a standard error of roughly 1.5 points, before
accounting for simulator nondeterminism. Treat differences below ~3 points as noise.

## Comparison guidance

These numbers come from a 20-episode-per-task budget. Published RoboTwin results use 100
episodes per task, so the two are **not** directly comparable; re-evaluate with the
100-episode protocol before quoting this model against published baselines.

## Reproducing with the upstream stack

```bash
git clone https://github.com/Robbyant/lingbot-vla-v2
# install RoboTwin 2.0 following experiment/robotwin/README.md

QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct \
EVAL_WORKDIR=/path/to/RoboTwin \
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --model_path /path/to/lingbot-vla-v2-6b-robotwin-aloha-agilex \
  --output_base /path/to/eval_output \
  --task_config demo_clean \
  --num_gpus 8 \
  --num_per_gpu 1
```

Use `--task_config demo_randomized` for the randomized setting. The version of the loader
shipped in [`deploy/`](../deploy) is the one used for these numbers; it reads
`lingbotvla_cli.yaml` from the checkpoint directory and drops training-only alignment
tensors at load time.
