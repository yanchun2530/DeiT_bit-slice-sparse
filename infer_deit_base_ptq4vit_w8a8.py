#!/usr/bin/env python3
"""Reload and evaluate a DeiT-Base PTQ4ViT W8A8 artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import timm
import torch

from configs import PTQ4ViT as quant_cfg
from utils.datasets import ViTImageNetLoaderGenerator
from run_deit_base_ptq4vit_w8a8 import (
    DEFAULT_DATA,
    DEFAULT_OUTPUT,
    evaluate,
    patch_attention_matmuls,
    set_attention_quantization,
    wrap_transformer_block_linears,
)


DEFAULT_ARTIFACT = DEFAULT_OUTPUT / (
    "deit_base_ptq4vit_transformer_w8a8_q8k4p8v4_patchfp32.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=128,
                        help="0 evaluates the complete validation set")
    return parser.parse_args()


def load_artifact(path: Path, device: torch.device):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    model = timm.create_model(artifact["model_arch"], pretrained=False)
    patch_attention_matmuls(model)
    set_attention_quantization(model, True)
    model.to(device)
    quant_cfg.matmul_name_list = []
    wrapped = wrap_transformer_block_linears(model)

    result = model.load_state_dict(artifact["float_state"], strict=False)
    expected_missing = {f"{name}.weight" for name in artifact["int8_weights"]}
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(
            "Artifact/model mismatch while restoring floating state: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )

    for name, module in wrapped.items():
        for attr, value in artifact["quant_params"][name].items():
            setattr(module, attr, value.to(device) if torch.is_tensor(value) else value)
        module.mode = "quant_forward"
        if hasattr(module, "calibrated"):
            module.calibrated = True

    for name, codes in artifact["int8_weights"].items():
        module = wrapped[name]
        integer = codes.to(device=device, dtype=torch.float32)
        integer = integer.view(module.n_V, module.crb_rows, module.n_H, module.crb_cols)
        dequantized = (integer * module.w_interval).reshape_as(module.weight)
        module.weight.data.copy_(dequantized)

    model.eval()
    return model, artifact["metrics"]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PTQ4ViT artifact inference")
    device = torch.device("cuda:0")
    model, stored_metrics = load_artifact(args.artifact, device)
    loaders = ViTImageNetLoaderGenerator(
        str(args.data), "imagenet", args.batch_size, args.batch_size,
        args.workers, kwargs={"model": model},
    )
    result = evaluate(
        model, loaders.test_loader(batch_size=args.batch_size), device,
        args.max_images,
    )
    print({
        "artifact": str(args.artifact),
        "stored_validation": stored_metrics["validation"]["ptq4vit_w8a8_q8k4p8v4"],
        "reloaded_smoke_validation": result,
    }, flush=True)


if __name__ == "__main__":
    main()
