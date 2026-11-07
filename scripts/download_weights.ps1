# Download model weights from Hugging Face into this repository root.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts/download_weights.ps1
$ErrorActionPreference = "Stop"
$RepoId = "liwy26/lingbot-vla-v2-6b-robotwin-aloha-agilex"
$Root = Split-Path -Parent $PSScriptRoot
$Shards = @(
  "model-00001-of-00003.safetensors",
  "model-00002-of-00003.safetensors",
  "model-00003-of-00003.safetensors"
)

Write-Host "Downloading weights from Hugging Face: $RepoId"
if (Get-Command hf -ErrorAction SilentlyContinue) {
  hf download $RepoId --include "model-*.safetensors" --local-dir $Root
} elseif (Get-Command huggingface-cli -ErrorAction SilentlyContinue) {
  huggingface-cli download $RepoId --include "model-*.safetensors" --local-dir $Root
} else {
  foreach ($s in $Shards) {
    $url = "https://huggingface.co/$RepoId/resolve/main/$s"
    $out = Join-Path $Root $s
    Write-Host "GET $url -> $out"
    Invoke-WebRequest -Uri $url -OutFile $out
  }
}

Write-Host "Done. Verify with: python examples/checkpoint_check.py `"$Root`""
