#!/usr/bin/env python3
"""
Export Gemma-style imprint ROM images for profile 2'b10.

This script builds the compact ROM images consumed by
`rtl/memory/imprinted_embedding_rom.v`:
  - gemma3_270m_token_emb_q88.hex
  - gemma3_270m_pos_emb_q88.hex
  - gemma3_270m_token_map.json
  - gemma3_270m_manifest.json

It supports `.safetensors` and `.npz` sources. If no source is provided,
it can generate a deterministic fixture for offline verification.
"""

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np


TOKEN_EMBED_KEYS = [
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "tok_embeddings.weight",
    "wte.weight",
]

POSITION_EMBED_KEYS = [
    "model.embed_positions.weight",
    "embed_positions.weight",
    "wpe.weight",
]


def float_to_q88(value: float) -> int:
    q = int(round(float(value) * 256.0))
    return max(-32768, min(32767, q))


def q88_hex(value: int) -> str:
    return f"{value & 0xFFFF:04x}"


def parse_safetensors(filepath: Path) -> dict:
    weights = {}
    with filepath.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len

        dtype_map = {
            "F32": np.float32,
            "F16": np.float16,
            "BF16": np.float16,
        }

        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = dtype_map.get(info["dtype"])
            if dtype is None:
                continue
            offsets = info["data_offsets"]
            shape = info["shape"]
            f.seek(data_start + offsets[0])
            raw = f.read(offsets[1] - offsets[0])
            arr = np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.float32)
            weights[name] = arr
    return weights


def load_weights(source: Path) -> tuple[dict, str]:
    suffix = source.suffix.lower()
    if suffix == ".npz":
        raw = np.load(source, allow_pickle=False)
        return {k: raw[k].astype(np.float32) for k in raw.files}, "npz"
    if suffix == ".safetensors":
        return parse_safetensors(source), "safetensors"
    raise ValueError(f"Unsupported source format: {source}")


def pick_2d_tensor(weights: dict, keys: list[str]) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        tensor = weights.get(key)
        if tensor is not None and tensor.ndim == 2:
            return tensor.astype(np.float32), key

    for key, tensor in weights.items():
        if tensor.ndim != 2:
            continue
        lower = key.lower()
        if "embed" in lower or "wte" in lower or "wpe" in lower:
            return tensor.astype(np.float32), key
    return None, None


def fit_matrix(matrix: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=np.float32)
    r = min(rows, matrix.shape[0])
    c = min(cols, matrix.shape[1])
    out[:r, :c] = matrix[:r, :c]
    return out


def build_synthetic_fixture(vocab_size: int, max_position: int, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    token = rng.normal(loc=0.0, scale=0.02, size=(vocab_size, dim)).astype(np.float32)
    pos = rng.normal(loc=0.0, scale=0.01, size=(max_position, dim)).astype(np.float32)
    return token, pos


def write_hex(path: Path, values: np.ndarray) -> str:
    flat = values.reshape(-1)
    with path.open("w", encoding="utf-8") as f:
        for v in flat:
            f.write(q88_hex(int(v)) + "\n")
    return sha256_file(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tokenizer_map(tokenizer_json: Path, vocab_size: int) -> list[dict]:
    with tokenizer_json.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Handles GPT-style vocab.json {token: id}
    inverse = {int(v): str(k) for k, v in vocab.items() if isinstance(v, int)}
    token_map = []
    for i in range(vocab_size):
        token_map.append({"token_id": i, "token": inverse.get(i, f"<tok_{i}>")})
    return token_map


def main():
    parser = argparse.ArgumentParser(description="Export Gemma imprint ROM images")
    parser.add_argument("--source", type=str, default="", help="Path to .safetensors or .npz weights")
    parser.add_argument("--tokenizer-json", type=str, default="", help="Optional tokenizer vocab JSON")
    parser.add_argument("--output-dir", type=str, default="weights/imprint", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=256, help="Exported token rows")
    parser.add_argument("--max-position", type=int, default=64, help="Exported position rows")
    parser.add_argument("--dim", type=int, default=8, help="Exported embedding dimension")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow deterministic fixture generation if source is missing")
    parser.add_argument("--synthetic-seed", type=int, default=270, help="Seed for deterministic synthetic fixture")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(args.source).resolve() if args.source else None

    used_source = "synthetic-fixture"
    source_format = "synthetic"
    token_key = "synthetic"
    pos_key = "synthetic"

    if source_path and source_path.exists():
        weights, source_format = load_weights(source_path)
        token_tensor, token_key = pick_2d_tensor(weights, TOKEN_EMBED_KEYS)
        if token_tensor is None:
            raise RuntimeError("Could not locate token embedding matrix in provided source.")

        pos_tensor, pos_key = pick_2d_tensor(weights, POSITION_EMBED_KEYS)
        token_fp = fit_matrix(token_tensor, args.vocab_size, args.dim)
        if pos_tensor is None:
            pos_fp = np.zeros((args.max_position, args.dim), dtype=np.float32)
            pos_key = "zero-filled"
        else:
            pos_fp = fit_matrix(pos_tensor, args.max_position, args.dim)

        used_source = str(source_path)
    else:
        if not args.allow_synthetic:
            raise FileNotFoundError(
                "No source provided/found. Re-run with --source <path> or --allow-synthetic."
            )
        token_fp, pos_fp = build_synthetic_fixture(
            vocab_size=args.vocab_size,
            max_position=args.max_position,
            dim=args.dim,
            seed=args.synthetic_seed,
        )

    token_q88 = np.vectorize(float_to_q88)(token_fp).astype(np.int32)
    pos_q88 = np.vectorize(float_to_q88)(pos_fp).astype(np.int32)

    token_hex = output_dir / "gemma3_270m_token_emb_q88.hex"
    pos_hex = output_dir / "gemma3_270m_pos_emb_q88.hex"
    token_sha = write_hex(token_hex, token_q88)
    pos_sha = write_hex(pos_hex, pos_q88)

    if args.tokenizer_json:
        token_map = load_tokenizer_map(Path(args.tokenizer_json).resolve(), args.vocab_size)
    else:
        token_map = [{"token_id": i, "token": f"<tok_{i}>"} for i in range(args.vocab_size)]

    token_map_path = output_dir / "gemma3_270m_token_map.json"
    with token_map_path.open("w", encoding="utf-8") as f:
        json.dump(token_map, f, indent=2)

    sample_token = min(12, args.vocab_size - 1)
    sample_pos = min(5, args.max_position - 1)
    sample_vec = (token_q88[sample_token] + pos_q88[sample_pos]).tolist()

    manifest = {
        "profile": "gemma3-270m-imprint-v1",
        "source": used_source,
        "source_format": source_format,
        "token_tensor_key": token_key,
        "position_tensor_key": pos_key,
        "vocab_size": args.vocab_size,
        "max_position": args.max_position,
        "dim": args.dim,
        "quantization": "Q8.8 signed 16-bit",
        "token_hex": token_hex.name,
        "token_hex_sha256": token_sha,
        "pos_hex": pos_hex.name,
        "pos_hex_sha256": pos_sha,
        "sample_token_id": sample_token,
        "sample_position": sample_pos,
        "sample_q88_vector": sample_vec,
        "synthetic_fixture": (source_format == "synthetic"),
    }

    manifest_path = output_dir / "gemma3_270m_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Gemma imprint export complete:")
    print(f"  token hex : {token_hex}")
    print(f"  pos hex   : {pos_hex}")
    print(f"  token map : {token_map_path}")
    print(f"  manifest  : {manifest_path}")
    print(f"  sample(token={sample_token}, pos={sample_pos}) -> {sample_vec[:4]} ...")


if __name__ == "__main__":
    main()

