# Examples

## Checkpoint integrity

```bash
python examples/checkpoint_check.py .
```

Verifies the tensor index, shard coverage, declared sizes, and every entry of
`SHA256SUMS` (12.77 GB hashed; takes about a minute on an SSD).

## Policy server

Run from the repository root so that the relative paths in `lingbotvla_cli.yaml`
(`assets/norm_stats/robotwin.json`, `configs/robot_configs/`) resolve correctly.

```bash
export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct
export PYTHONPATH=/path/to/lingbot-vla-v2:$PYTHONPATH

python -m deploy.lingbot_vla_v2_policy \
  --model_path . \
  --use_length 50 \
  --use_bf16 True --use_fp32 False \
  --port 8006
```

The server speaks the OpenPI WebSocket protocol. `deploy/websocket_client_policy.py`
is a reference client.

The first request of every episode must be a reset that selects the robot config:

```python
server.infer({"reset": True, "robo_name": "robotwin"})
```

Subsequent observations use the **raw simulator keys**; the mapping to policy features is
defined in `configs/robot_configs/robotwin.yaml`:

| Key | Shape / type |
|---|---|
| `observation.images.cam_high` | `uint8` H×W×3 |
| `observation.images.cam_left_wrist` | `uint8` H×W×3 |
| `observation.images.cam_right_wrist` | `uint8` H×W×3 |
| `observation.state` | 14-D joint + gripper vector |
| `task` | task instruction string |

The server returns a dict of action-chunk arrays keyed by the action features in the robot
config (`action.arm.position`, `action.effector.position`); `--use_length` caps how many
steps are returned per call.

## RoboTwin simulation

See the upstream guide: `experiment/robotwin/README.md` in
[lingbot-vla-v2](https://github.com/Robbyant/lingbot-vla-v2), and
[`docs/EVALUATION.md`](../docs/EVALUATION.md) here for the protocol used to produce the
reported numbers.
