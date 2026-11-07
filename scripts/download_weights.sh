#!/usr/bin/env bash
# Download model weights from Hugging Face into the repository root.
set -euo pipefail
REPO_ID="liwy26/lingbot-vla-v2-6b-robotwin-aloha-agilex"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if command -v hf >/dev/null 2>&1; then
  hf download "$REPO_ID" --include "model-*.safetensors" --local-dir "$ROOT"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$REPO_ID" --include "model-*.safetensors" --local-dir "$ROOT"
else
  for s in model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
    curl -L -o "$ROOT/$s" "https://huggingface.co/$REPO_ID/resolve/main/$s"
  done
fi

echo "Done. Verify with: python examples/checkpoint_check.py \"$ROOT\""
