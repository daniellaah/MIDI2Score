from __future__ import annotations

import argparse
from pathlib import Path

import torch


NORM_KEY_REWRITES = {
    ".norm1.": ".self_attn_norm.",
    ".norm2.": ".ffn_norm.",
    ".norm3.": ".cross_attn_norm.",
}


def _convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        converted_key = key
        for old, new in NORM_KEY_REWRITES.items():
            if old in converted_key:
                converted_key = converted_key.replace(old, new)
                break
        converted[converted_key] = value
    return converted


def convert_checkpoint(input_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(input_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dict payload or a plain state_dict.")

    if "model_state" in checkpoint:
        checkpoint = dict(checkpoint)
        checkpoint["model_state"] = _convert_state_dict(checkpoint["model_state"])
        checkpoint["checkpoint_conversion"] = "decoder_norm_layout_v2"
    else:
        checkpoint = _convert_state_dict(checkpoint)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert old decoder checkpoints to the current norm layout.",
    )
    parser.add_argument("input", type=Path, help="Path to the source checkpoint.")
    parser.add_argument("output", type=Path, help="Path to write the converted checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_checkpoint(args.input, args.output)
    print(f"converted checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
