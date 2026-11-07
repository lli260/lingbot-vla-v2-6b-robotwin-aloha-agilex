"""Offline integrity check for a sharded LingBot-VLA hf_ckpt directory.

Usage:
    python examples/checkpoint_check.py [checkpoint_dir]

Checks the tensor index, shard coverage, shard sizes, and (if present)
the SHA256SUMS file. Exits non-zero on failure.
"""

import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        fail(f"missing {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    if not weight_map:
        fail("empty weight_map in index")

    shard_names = sorted(set(weight_map.values()))
    shard_sizes = {}
    for name in shard_names:
        shard = root / name
        if not shard.is_file():
            fail(f"missing shard {name}")
        shard_sizes[name] = shard.stat().st_size

    for name in shard_names:
        count = sum(1 for value in weight_map.values() if value == name)
        print(f"  {name}: {count} tensors, {shard_sizes[name]:,} bytes")

    total_declared = index.get("metadata", {}).get("total_size")
    total_actual = sum(shard_sizes.values())
    print(f"  tensors: {len(weight_map)}")
    print(f"  declared data bytes: {total_declared:,}")
    print(f"  files on disk:       {total_actual:,}")

    if total_declared is not None and total_actual < total_declared:
        fail("on-disk size is smaller than the index total_size")

    json.loads((root / "config.json").read_text(encoding="utf-8"))
    if not (root / "lingbotvla_cli.yaml").is_file():
        fail("missing lingbotvla_cli.yaml (required by the inference loader)")

    sums_path = root / "SHA256SUMS"
    if sums_path.is_file():
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            expected, _, relative = line.partition("  ")
            target = root / relative
            if not target.is_file():
                fail(f"missing file listed in SHA256SUMS: {relative}")
            actual = sha256_file(target)
            if actual != expected:
                fail(f"checksum mismatch: {relative}")
            print(f"  sha256 ok: {relative}")
    else:
        print("  SHA256SUMS not found, skipped checksum verification")

    print(f"OK: {len(weight_map)} tensors across {len(shard_names)} shards in {root}")


if __name__ == "__main__":
    main()
