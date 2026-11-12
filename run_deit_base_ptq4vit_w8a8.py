#!/usr/bin/env python3
"""Evaluate DeiT-Base with PTQ4ViT W8A8 in transformer blocks.

The patch embedding and classifier head stay FP32. Attention Q/K/P/V use the
same Q8/K4/P8/V4 fake-quant path as the related ViT-Small experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

ROOT = Path(__file__).resolve().parent
PTQ4VIT_ROOT = ROOT / "PTQ4ViT"
if str(PTQ4VIT_ROOT) not in sys.path:
    sys.path.insert(0, str(PTQ4VIT_ROOT))

import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, maybe_add_mask
from tqdm import tqdm

from configs import PTQ4ViT as quant_cfg
from utils.datasets import ViTImageNetLoaderGenerator
from utils.net_wrap import wrap_certain_modules_in_net
from utils.quant_calib import HessianQuantCalibrator


DEFAULT_CHECKPOINT = ROOT / "checkpoints/deit_base_patch16_224.fb_in1k_fp32_state_dict.pth"
DEFAULT_DATA = Path(os.environ.get("IMAGENET_ROOT", ROOT / "data/imagenet"))
DEFAULT_OUTPUT = ROOT / "results/deit_base_ptq4vit_w8a8_q8k4p8v4_patchfp32"
TARGET_MODULE_SUFFIXES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
QUANT_PARAM_NAMES = (
    "w_interval", "a_interval", "a_pos_interval", "a_neg_interval",
    "A_interval", "B_interval", "split", "n_G_A", "n_G_B", "n_V_A",
    "n_V_B", "n_H_A", "n_H_B", "crb_groups_A", "crb_groups_B",
    "crb_rows_A", "crb_rows_B", "crb_cols_A", "crb_cols_B",
    "pad_groups_A", "pad_groups_B", "pad_rows_A", "pad_rows_B",
    "pad_cols_A", "pad_cols_B", "w_bit", "a_bit", "A_bit", "B_bit",
    "w_qmax", "a_qmax", "A_qmax", "B_qmax", "calibrated", "mode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arch", default="deit_base_patch16_224.fb_in1k")
    parser.add_argument("--calib-size", type=int, default=32)
    parser.add_argument("--calib-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-eval-images", type=int, default=0,
                        help="0 evaluates the entire ImageNet validation split")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_attention_matmuls(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, Attention):
            module._mix_quant_enabled = False
            module.forward = MethodType(mixed_precision_attention_forward, module)


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"Hadamard dimension must be a power of two, got {n}")
    y = x.float()
    step = 1
    while step < n:
        y = y.reshape(*y.shape[:-1], -1, 2, step)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.cat((a + b, a - b), dim=-1).reshape(*x.shape)
        step *= 2
    return (y / math.sqrt(n)).to(x.dtype)


def symmetric_quant_dequant(x: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    max_abs = torch.amax(torch.abs(x), dim=-1, keepdim=True)
    scale = max_abs.clamp(min=1e-5) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int8)
    return (q.float() * scale.float()).to(x.dtype)


def set_attention_quantization(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, Attention):
            module._mix_quant_enabled = enabled


def mixed_precision_attention_forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
    batch, tokens, channels = x.shape
    qkv = self.qkv(x).reshape(
        batch, tokens, 3, self.num_heads, self.head_dim
    ).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    if self._mix_quant_enabled:
        q = symmetric_quant_dequant(hadamard_transform(q), 8)
        k = symmetric_quant_dequant(hadamard_transform(k), 4)
        v = symmetric_quant_dequant(hadamard_transform(v), 4)
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    if attn_mask is not None:
        attn = maybe_add_mask(attn, attn_mask)
    attn = self.attn_drop(attn.softmax(dim=-1))
    if self._mix_quant_enabled:
        attn = symmetric_quant_dequant(attn, 8)
    x = (attn @ v).transpose(1, 2)
    if self._mix_quant_enabled:
        x = hadamard_transform(x)
    x = x.reshape(batch, tokens, self.attn_dim)
    x = self.norm(x)
    return self.proj_drop(self.proj(x))


@torch.inference_mode()
def evaluate(model: nn.Module, loader, device: torch.device, limit: int = 0) -> dict[str, Any]:
    model.eval()
    total = top1 = top5 = 0
    progress = tqdm(loader, desc="ImageNet validation", mininterval=5)
    for images, targets in progress:
        if limit and total >= limit:
            break
        if limit:
            take = min(images.shape[0], limit - total)
            images, targets = images[:take], targets[:take]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits; refusing to report accuracy")
        predictions = logits.topk(5, dim=1).indices
        top1 += int((predictions[:, 0] == targets).sum().item())
        top5 += int((predictions == targets[:, None]).any(dim=1).sum().item())
        total += int(targets.numel())
        progress.set_postfix(top1=f"{100 * top1 / total:.3f}%")
    if total == 0:
        raise RuntimeError("ImageNet validation loader yielded no images")
    return {"images": total, "top1": 100 * top1 / total, "top5": 100 * top5 / total}


def is_target_name(name: str) -> bool:
    return name.startswith("blocks.") and name.rsplit(".", 2)[-2] + "." + name.rsplit(".", 1)[-1] in TARGET_MODULE_SUFFIXES


def source_weight_zero_stats(model: nn.Module) -> dict[str, Any]:
    per_layer = {}
    total_zeros = total_values = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not is_target_name(name):
            continue
        weight = module.weight.detach()
        zeros = int((weight == 0).sum().item())
        count = weight.numel()
        per_layer[name] = {"exact_zero_count": zeros, "values": count,
                           "exact_zero_fraction": zeros / count}
        total_zeros += zeros
        total_values += count
    if not total_values:
        raise RuntimeError("No transformer-block linear weights matched the target scope")
    return {
        "scope": "48 transformer-block Linear weights (QKV, attention projection, MLP fc1/fc2)",
        "exact_zero_count": total_zeros,
        "values": total_values,
        "exact_zero_fraction": total_zeros / total_values,
        "per_layer": per_layer,
    }


def wrap_transformer_block_linears(model: nn.Module):
    layers = list(range(len(model.blocks)))
    wrapped = wrap_certain_modules_in_net(
        model,
        quant_cfg,
        layers=layers,
        modules_to_wrap=["qkv", "proj", "fc1", "fc2"],
        wrap_embedding=False,
    )
    expected = len(layers) * len(TARGET_MODULE_SUFFIXES)
    if len(wrapped) != expected or any(not is_target_name(name) for name in wrapped):
        raise RuntimeError(
            f"Expected {expected} transformer linears, got {len(wrapped)}: {list(wrapped)}"
        )
    return wrapped


def export_int8_weights(wrapped_modules):
    result = {}
    for name, module in wrapped_modules.items():
        if not hasattr(module, "w_interval") or module.w_interval is None:
            continue
        weight = module.weight.detach().view(
            module.n_V, module.crb_rows, module.n_H, module.crb_cols
        )
        codes = (weight / module.w_interval).round().clamp(
            -module.w_qmax, module.w_qmax - 1
        )
        result[name] = codes.reshape_as(module.weight).cpu().to(torch.int8)
    return result


def cpu_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [cpu_value(item) for item in value]
    return None


def quantized_weight_zero_stats(source_stats, int8_weights) -> dict[str, Any]:
    per_layer = {}
    zero_count = value_count = 0
    for name, codes in int8_weights.items():
        zeros = int((codes == 0).sum().item())
        count = codes.numel()
        source = source_stats["per_layer"][name]
        per_layer[name] = {
            "int8_zero_code_count": zeros,
            "values": count,
            "int8_zero_code_fraction": zeros / count,
            "source_fp32_exact_zero_count": source["exact_zero_count"],
            "source_fp32_exact_zero_fraction": source["exact_zero_fraction"],
        }
        zero_count += zeros
        value_count += count
    if not value_count:
        raise RuntimeError("No quantized int8 weight codes were exported")
    return {
        "scope": source_stats["scope"],
        "int8_zero_code_count": zero_count,
        "values": value_count,
        "int8_zero_code_fraction": zero_count / value_count,
        "interpretation": (
            "Zeros created by W8 rounding in the quantized representation; no weight pruning mask was applied, "
            "so this is not structural sparsity or evidence of sparse-kernel speedup."
        ),
        "per_layer": per_layer,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PTQ4ViT Hessian calibration")
    if not (args.data / "train").is_dir() or not (args.data / "val").is_dir():
        raise FileNotFoundError(f"Expected ImageNet train/val folders under {args.data}")

    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)

    model = timm.create_model(args.arch, pretrained=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    source_hash = sha256_file(args.checkpoint)
    patch_attention_matmuls(model)
    model.to(device).eval()
    print(f"Loaded {args.arch}; checkpoint SHA-256={source_hash}", flush=True)

    source_zero_stats = source_weight_zero_stats(model)
    loaders = ViTImageNetLoaderGenerator(
        str(args.data), "imagenet", args.calib_size, args.eval_batch_size,
        args.workers, kwargs={"model": model},
    )
    val_loader = loaders.test_loader(batch_size=args.eval_batch_size)
    calib_loader = loaders.calib_loader(num=args.calib_size, seed=17)

    set_attention_quantization(model, False)
    dense_metrics = evaluate(model, val_loader, device, args.max_eval_images)
    print(f"Dense FP32 reference: {dense_metrics}", flush=True)

    # PTQ4ViT's config includes attention matmuls by default. The user-selected
    # Q8/K4/P8/V4 path quantizes them explicitly, so avoid a second quantizer.
    quant_cfg.matmul_name_list = []
    set_attention_quantization(model, True)
    wrapped_modules = wrap_transformer_block_linears(model)
    model.eval()

    calibration_start = time.time()
    calibrator = HessianQuantCalibrator(
        model, wrapped_modules, calib_loader, sequential=False,
        batch_size=args.calib_batch_size,
    )
    calibrator.batching_quant_calib()
    calibration_seconds = time.time() - calibration_start
    del calibrator
    torch.cuda.empty_cache()
    print(f"PTQ4ViT calibration finished in {calibration_seconds / 60:.2f} min", flush=True)

    quant_metrics = evaluate(model, val_loader, device, args.max_eval_images)
    delta_top1 = quant_metrics["top1"] - dense_metrics["top1"]
    print(
        "PTQ4ViT transformer W8A8 + attention Q8/K4/P8/V4: "
        f"{quant_metrics}; Top-1 delta={delta_top1:+.4f} pp",
        flush=True,
    )

    int8_weights = export_int8_weights(wrapped_modules)
    quant_zero_stats = quantized_weight_zero_stats(source_zero_stats, int8_weights)
    quantized_weight_keys = {f"{name}.weight" for name in int8_weights}
    float_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name not in quantized_weight_keys
    }
    quant_params = {}
    for name, module in wrapped_modules.items():
        quant_params[name] = {
            attr: cpu_value(getattr(module, attr))
            for attr in QUANT_PARAM_NAMES
            if hasattr(module, attr) and cpu_value(getattr(module, attr)) is not None
        }

    metrics = {
        "model": args.arch,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": source_hash,
        "quantization": {
            "linear_weights_and_inputs": "PTQ4ViT Hessian PTQ W8A8",
            "w8a8_linear_scope": source_zero_stats["scope"],
            "patch_embedding": "FP32 weights and activations",
            "classifier_head": "FP32 weights and activations",
            "attention": "symmetric per-vector Q8/K4/P8/V4 fake QDQ with Hadamard rotation",
            "attention_integer_kernel": False,
            "packed_int8_or_int4_weights": False,
        },
        "calibration": {
            "dataset": "ImageNet train",
            "images": args.calib_size,
            "batch_size": args.calib_batch_size,
            "seed": 17,
            "seconds": calibration_seconds,
        },
        "validation": {
            "dataset": "ImageNet val",
            "dense_fp32": dense_metrics,
            "ptq4vit_w8a8_q8k4p8v4": quant_metrics,
            "top1_delta_pp": delta_top1,
            "top5_delta_pp": quant_metrics["top5"] - dense_metrics["top5"],
        },
        "weight_zero_rates": {
            "source_fp32_exact_zeros": source_zero_stats,
            "quantized_int8_zero_codes": quant_zero_stats,
        },
        "quantized_linear_modules": len(int8_weights),
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "ptq4vit_git_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "reference_note": (
            "This is fake quantize/dequantize evaluated with floating-point PyTorch operators. "
            "The reported INT8 zero-code rate reflects quantization rounding, not pruning."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    artifact_path = args.output / "deit_base_ptq4vit_transformer_w8a8_q8k4p8v4_patchfp32.pt"
    torch.save({
        "format": "PTQ4ViT transformer-linear W8A8; FP32 patch embedding and classifier; Q8/K4/P8/V4 attention QDQ",
        "model_arch": args.arch,
        "int8_weights": int8_weights,
        "float_state": float_state,
        "quant_params": quant_params,
        "metrics": metrics,
    }, artifact_path)
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved artifact: {artifact_path}", flush=True)
    print(f"Saved metrics: {metrics_path}", flush=True)
    print(
        "Weight zero rates: "
        f"source FP32 exact={source_zero_stats['exact_zero_fraction']:.6%}, "
        f"quantized INT8 codes={quant_zero_stats['int8_zero_code_fraction']:.6%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
