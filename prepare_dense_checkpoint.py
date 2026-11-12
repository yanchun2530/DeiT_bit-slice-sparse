#!/usr/bin/env python3
"""Download the timm DeiT-Base ImageNet-1k weights as a plain state dict."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import timm
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", default="deit_base_patch16_224.fb_in1k",
        help="timm architecture and pretrained tag",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent
        / "checkpoints/deit_base_patch16_224.fb_in1k_fp32_state_dict.pth",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error(f"{args.output} already exists; pass --force to replace it")

    model = timm.create_model(args.arch, pretrained=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"saved={args.output}")
    print(f"sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
