#!/usr/bin/env python3
"""Fine-tune dense DeiT-Base with a fixed W8 S+M bit-slice block mask.

Mask preparation reuses the saved dense PTQ4ViT calibration artifact and the
activation-weighted mask ranking from optimize_deit_granet_weight_bitslice.py.
The training forward uses fake-quantized S+M W8 weights in the 48 transformer
Linear layers; activations, patch embedding, and classifier remain FP32/AMP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parent
PTQ_ROOT = ROOT / "PTQ4ViT"
if str(PTQ_ROOT) not in sys.path:
    sys.path.insert(0, str(PTQ_ROOT))

import optimize_deit_granet_weight_bitslice as bitslice_search  # noqa: E402
from infer_deit_base_ptq4vit_w8a8 import DEFAULT_ARTIFACT, load_artifact  # noqa: E402


DEFAULT_DATA = Path(os.environ.get("IMAGENET_ROOT", ROOT / "data/imagenet"))
DEFAULT_DENSE_CHECKPOINT = ROOT / "checkpoints/deit_base_patch16_224.fb_in1k_fp32_state_dict.pth"
DEFAULT_OUTPUT = ROOT / "results/qat_75_globalcoverage_35ep"
TARGET_SUFFIXES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_DENSE_CHECKPOINT)
    parser.add_argument("--ptq-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mask-artifact", type=Path, default=None)
    parser.add_argument("--arch", default="deit_base_patch16_224.fb_in1k")
    parser.add_argument("--target-rate", type=float, default=75.0,
                        help="Target aggregate S+M zero plane-block rate, in percent")
    parser.add_argument(
        "--mask-selection",
        choices=("plane-order", "global-cost", "global-coverage"),
        default="global-coverage",
        help=(
            "plane-order fills b0 through b6 in sequence; global-cost ranks all "
            "eligible blocks together; global-coverage also reserves one block "
            "per group to avoid masking every magnitude plane in a group"
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--ptq-calib-images", type=int, default=32)
    parser.add_argument("--ptq-calib-seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="resume model, optimizer, and AMP scaler state from a training checkpoint",
    )
    parser.add_argument(
        "--resume-best-top1", type=float, default=None,
        help="best validation Top-1 already achieved before the resume checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Per-GPU batch size; 256 on four GPUs gives global batch 1,024")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--eval-images", type=int, default=0,
        help="deterministic random validation subset size; 0 evaluates the full split",
    )
    parser.add_argument(
        "--max-train-steps", type=int, default=0,
        help="stop after this many optimizer steps total; 0 trains all requested epochs",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--benchmark-only", action="store_true",
                        help="run a short real-data throughput/memory probe, then exit")
    parser.add_argument("--benchmark-steps", type=int, default=80)
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def make_zero_plane_masks(
    target: float,
    records: dict[str, dict[str, Any]],
    ranges: dict[str, tuple[int, int]],
    plan: dict[float, dict[int, int]],
    rankings: dict[int, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Freeze naturally-zero magnitude planes and the newly selected blocks."""
    masks: dict[str, torch.Tensor] = {}
    for name, record in records.items():
        codes = record["codes"].to(device=device, dtype=torch.int16)
        rows, width = codes.shape
        groups_per_row = width // 8
        magnitude = codes.abs().clamp(max=127).reshape(rows, groups_per_row, 8)
        zero_mask = torch.stack(
            [~magnitude.bitwise_and(1 << bit).ne(0).any(dim=-1) for bit in range(7)],
            dim=-1,
        )
        masks[name] = zero_mask

    for bit in range(7):
        selected_count = int(plan[target][bit])
        if selected_count == 0:
            continue
        selected_ids = rankings[bit][:selected_count].to(device=device)
        for name, record in records.items():
            start, end = ranges[name]
            local_ids = selected_ids[(selected_ids >= start) & (selected_ids < end)] - start
            if local_ids.numel():
                masks[name].reshape(-1, 7)[local_ids, bit] = True
    return masks


def rank_global_plane_blocks(
    records: dict[str, dict[str, Any]],
    ranges: dict[str, tuple[int, int]],
    total_groups: int,
    device: torch.device,
    preserve_one_per_group: bool = False,
) -> tuple[torch.Tensor, dict[int, int]]:
    """Rank every eligible (8-weight group, magnitude bit) block together."""
    scores = torch.full(
        (total_groups, 7), float("inf"), dtype=torch.float32, device=device
    )
    eligible_by_bit: dict[int, int] = {}
    for bit in range(7):
        bit_mask = 1 << bit
        eligible_total = 0
        for name, record in records.items():
            rows, input_width = record["codes"].shape
            groups_per_row = input_width // 8
            magnitude = record["codes"].abs().clamp(max=127)
            grouped_magnitude = magnitude.reshape(rows, groups_per_row, 8)
            bit_set = grouped_magnitude.bitwise_and(bit_mask).ne(0)
            eligible = bit_set.any(dim=-1)

            scale_grouped = record["scale"].reshape(rows, groups_per_row, 8)
            x2_grouped = record["x2"].reshape(1, groups_per_row, 8)
            perturbation = bit_set.to(torch.float32) * float(bit_mask) * scale_grouped
            score = (perturbation.square() * x2_grouped).sum(dim=-1)
            start, end = ranges[name]
            scores[start:end, bit] = torch.where(
                eligible, score, torch.full_like(score, float("inf"))
            ).reshape(-1)
            eligible_total += int(eligible.sum().item())
        eligible_by_bit[bit] = eligible_total

    candidate_count = sum(eligible_by_bit.values())
    if preserve_one_per_group:
        has_candidate = torch.isfinite(scores).any(dim=1)
        valid_scores = scores.masked_fill(~torch.isfinite(scores), float("-inf"))
        reserved_bit = valid_scores.argmax(dim=1)
        group_ids = torch.arange(total_groups, device=device)[has_candidate]
        scores[group_ids, reserved_bit[has_candidate]] = float("inf")
        candidate_count -= int(has_candidate.sum().item())

    flat_scores = scores.reshape(-1)
    order = torch.argsort(flat_scores, stable=True)
    return order[:candidate_count].cpu(), eligible_by_bit


def split_global_prefix(
    target: float,
    global_order: torch.Tensor,
    selected_count: int,
) -> tuple[dict[float, dict[int, int]], dict[int, torch.Tensor]]:
    """Convert a global candidate prefix into the per-plane format used below."""
    prefix = global_order[:selected_count]
    bit_ids = prefix.remainder(7)
    counts = torch.bincount(bit_ids, minlength=7)
    plan = {target: {bit: int(counts[bit]) for bit in range(7)}}
    rankings = {
        bit: (prefix[bit_ids.eq(bit)] // 7).contiguous()
        for bit in range(7)
    }
    return plan, rankings


def prepare_mask_artifact(args: argparse.Namespace, output_path: Path) -> None:
    if not args.ptq_artifact.is_file():
        raise FileNotFoundError(f"Dense PTQ4ViT artifact not found: {args.ptq_artifact}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to prepare the calibrated bit-slice mask")
    if not 0.0 < args.target_rate < 100.0:
        raise ValueError("--target-rate must be between 0 and 100 percent")

    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    print(f"Loading dense PTQ4ViT artifact: {args.ptq_artifact}", flush=True)
    model, ptq_metrics = load_artifact(args.ptq_artifact, device)
    source_sha = sha256_file(args.source_checkpoint)
    calibrated_source_sha = ptq_metrics.get("source_checkpoint_sha256")
    if calibrated_source_sha and source_sha != calibrated_source_sha:
        raise RuntimeError(
            "The dense checkpoint does not match the checkpoint used to calibrate "
            f"the PTQ artifact: {source_sha} != {calibrated_source_sha}"
        )
    # load_artifact already wraps and calibrates the Linear layers; collect the
    # existing modules instead of wrapping them a second time (which would
    # reset their calibrated intervals).
    wrapped = {
        name: module
        for name, module in model.named_modules()
        if bitslice_search.ptq.is_target_name(name)
        and isinstance(module, nn.Linear)
        and getattr(module, "w_interval", None) is not None
    }
    records, ranges = bitslice_search.collect_base_records(wrapped, device)
    codes = {name: record["codes"] for name, record in records.items()}
    baseline = bitslice_search.summarize_smag(codes)
    if len(wrapped) != 48:
        raise RuntimeError(f"Expected 48 transformer Linear layers, got {len(wrapped)}")

    loader_generator = bitslice_search.ptq.ViTImageNetLoaderGenerator(
        str(args.data), "imagenet", args.ptq_calib_images, 64, args.workers,
        kwargs={"model": model},
    )
    calib_loader = loader_generator.calib_loader(
        num=args.ptq_calib_images, seed=args.ptq_calib_seed
    )
    print(
        f"Dense W8 baseline S+M rate={100 * baseline['all_plane_zero_block_rate']:.4f}%; "
        f"target={args.target_rate:.4f}%",
        flush=True,
    )
    bitslice_search.collect_quantized_input_second_moments(model, records, calib_loader, device)

    target_key = float(args.target_rate)
    target_zero_blocks = math.ceil(
        args.target_rate / 100.0 * baseline["total_plane_blocks"]
    )

    if args.mask_selection in ("global-cost", "global-coverage"):
        print("Ranking all eligible magnitude-plane blocks by activation-weighted cost...", flush=True)
        global_order, eligible_by_bit = rank_global_plane_blocks(
            records,
            ranges,
            baseline["groups"],
            device,
            preserve_one_per_group=(args.mask_selection == "global-coverage"),
        )
        baseline_magnitude_zeros = sum(baseline["plane_zero_block_counts"][:7])
        if target_zero_blocks < baseline["zero_block_count"]:
            raise RuntimeError(
                "Requested target is below the natural dense W8 S+M zero-block rate"
            )
        max_candidates = min(
            int(global_order.numel()), target_zero_blocks - baseline_magnitude_zeros
        )
        print(
            f"Global candidates={global_order.numel():,}; selecting up to "
            f"{max_candidates:,} while accounting for induced sign-plane zeros",
            flush=True,
        )

        def summarize_global_prefix(selected_count: int):
            trial_plan, trial_rankings = split_global_prefix(
                target_key, global_order, selected_count
            )
            trial_codes = bitslice_search.make_candidate_codes(
                target_key, records, ranges, trial_plan, eligible_by_bit,
                trial_rankings, device,
            )
            return bitslice_search.summarize_smag(trial_codes)

        # The aggregate rate also counts sign-plane blocks. Removing a final
        # negative magnitude can therefore add an extra zero block, so search
        # for the largest global-cost prefix that does not exceed the target.
        low, high = 0, max_candidates
        best_count = 0
        while low <= high:
            middle = (low + high) // 2
            trial_summary = summarize_global_prefix(middle)
            if trial_summary["zero_block_count"] <= target_zero_blocks:
                best_count = middle
                low = middle + 1
            else:
                high = middle - 1
        plan, rankings = split_global_prefix(target_key, global_order, best_count)
        candidate_codes = bitslice_search.make_candidate_codes(
            target_key, records, ranges, plan, eligible_by_bit, rankings, device
        )
        candidate_summary = bitslice_search.summarize_smag(candidate_codes)
        print(
            f"Selected {best_count:,} globally lowest-cost blocks; "
            f"measured rate={100 * candidate_summary['all_plane_zero_block_rate']:.6f}%",
            flush=True,
        )
    else:
        plan, eligible_by_bit = bitslice_search.build_target_plan([target_key], baseline)
        rankings: dict[int, torch.Tensor] = {}
        for bit in range(7):
            count = int(plan[target_key][bit])
            if count <= 0:
                rankings[bit] = torch.empty(0, dtype=torch.int64)
                continue
            print(f"Ranking b{bit} bit-plane blocks (selecting {count:,})...", flush=True)
            ranking, eligible = bitslice_search.rank_plane_blocks(records, ranges, bit, device)
            if eligible != eligible_by_bit[bit]:
                raise RuntimeError(f"Eligible b{bit} group count changed during ranking")
            rankings[bit] = ranking

        def current_candidate():
            codes_now = bitslice_search.make_candidate_codes(
                target_key, records, ranges, plan, eligible_by_bit, rankings, device
            )
            return codes_now, bitslice_search.summarize_smag(codes_now)

        candidate_codes, candidate_summary = current_candidate()
        if candidate_summary["zero_block_count"] > target_zero_blocks:
            for bit in range(6, -1, -1):
                original_count = int(plan[target_key][bit])
                if original_count == 0:
                    continue
                low, high = 0, original_count
                best_count = 0
                best_candidate = None
                best_summary = None
                while low <= high:
                    middle = (low + high) // 2
                    plan[target_key][bit] = middle
                    trial_codes, trial_summary = current_candidate()
                    if trial_summary["zero_block_count"] <= target_zero_blocks:
                        best_count = middle
                        best_candidate = trial_codes
                        best_summary = trial_summary
                        low = middle + 1
                    else:
                        high = middle - 1
                plan[target_key][bit] = best_count
                if best_candidate is not None:
                    candidate_codes, candidate_summary = best_candidate, best_summary
                if candidate_summary["zero_block_count"] <= target_zero_blocks:
                    break

    plane_masks = make_zero_plane_masks(
        target_key, records, ranges, plan, rankings, device
    )
    achieved = 100.0 * candidate_summary["all_plane_zero_block_rate"]
    if achieved > args.target_rate + 100.0 / baseline["total_plane_blocks"]:
        raise RuntimeError(
            f"Mask selection achieved {achieved:.6f}% instead of {args.target_rate:.6f}%"
        )

    intervals = {}
    for name, module in wrapped.items():
        intervals[name] = {
            "w_interval": module.w_interval.detach().cpu(),
            "n_V": int(module.n_V),
            "n_H": int(module.n_H),
            "crb_rows": int(module.crb_rows),
            "crb_cols": int(module.crb_cols),
        }
    payload = {
        "format": "deit_dense_w8_smag_bitplane_block_mask_v1",
        "model_arch": args.arch,
        "target_rate_percent": float(args.target_rate),
        "mask_selection_strategy": args.mask_selection,
        "achieved_initial_rate_percent": achieved,
        "source_checkpoint": str(args.source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "ptq_artifact": str(args.ptq_artifact),
        "ptq_artifact_sha256": sha256_file(args.ptq_artifact),
        "ptq_stored_metrics": ptq_metrics,
        "quantization": (
            "saved dense PTQ4ViT W8 Hessian intervals; symmetric signed int8 codes; "
            "S+M magnitude bits b0-b6 are masked; -128 is saturated to -127"
        ),
        "scope": "48 transformer-block Linear layers: qkv, attn.proj, mlp.fc1, mlp.fc2",
        "grouping": "each output row, consecutive groups of 8 input features",
        "mask_method": (
            "fixed natural S+M magnitude-plane zeros plus a global one-shot ranking "
            "of all eligible (group,bit) blocks by activation-weighted output-MSE "
            "proxy; the lowest-cost prefix is selected across bits b0-b6, with a "
            "binary search to account for induced sign-plane zeros"
            + (
                "; the highest-cost eligible magnitude plane in each group is reserved "
                "so each group keeps at least one nonzero magnitude plane"
                if args.mask_selection == "global-coverage"
                else ""
            )
            if args.mask_selection in ("global-cost", "global-coverage")
            else "fixed zeros include initially-zero S+M magnitude planes plus additional "
                 "activation-weighted low-cost blocks selected to hit the aggregate target"
        ),
        "calibration_images": args.ptq_calib_images,
        "calibration_seed": args.ptq_calib_seed,
        "baseline_rate_percent": 100.0 * baseline["all_plane_zero_block_rate"],
        "baseline_plane_zero_block_counts": baseline["plane_zero_block_counts"],
        "selected_additional_blocks_by_magnitude_bit": {
            f"b{bit}": int(plan[target_key][bit]) for bit in range(7)
        },
        "candidate_summary": candidate_summary,
        "intervals": intervals,
        "zero_plane_masks": {
            name: mask.detach().cpu().to(torch.uint8)
            for name, mask in plane_masks.items()
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_save(output_path, payload)
    print(
        f"Saved mask artifact: {output_path}; initial S+M rate={achieved:.6f}%",
        flush=True,
    )


class BitsliceQATLinear(nn.Linear):
    """Linear layer with STE W8 quantization and fixed S+M plane-block zeros."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        interval: torch.Tensor,
        zero_plane_mask: torch.Tensor,
        n_v: int,
        n_h: int,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.n_v = int(n_v)
        self.n_h = int(n_h)
        if out_features % self.n_v or in_features % self.n_h:
            raise ValueError(
                f"Incompatible PTQ grouping n_V={n_v}, n_H={n_h} for "
                f"Linear({in_features},{out_features})"
            )
        self.crb_rows = out_features // self.n_v
        self.crb_cols = in_features // self.n_h
        if tuple(interval.shape) != (self.n_v, 1, self.n_h, 1):
            raise ValueError(
                f"Expected interval {(self.n_v, 1, self.n_h, 1)}, got {tuple(interval.shape)}"
            )
        expected_mask = (out_features, in_features // 8, 7)
        if tuple(zero_plane_mask.shape) != expected_mask:
            raise ValueError(f"Expected bit-plane mask {expected_mask}, got {tuple(zero_plane_mask.shape)}")
        self.register_buffer("w_interval", interval.detach().float().clone())
        self.register_buffer("zero_plane_mask", zero_plane_mask.detach().bool().clone())
        keep_bits = torch.zeros(
            (out_features, in_features // 8), dtype=torch.int16
        )
        for bit in range(7):
            keep_bits.bitwise_or_(
                (~zero_plane_mask[:, :, bit].detach().cpu().bool()).to(torch.int16) << bit
            )
        self.register_buffer("keep_magnitude_bits", keep_bits)

    def _quantized_smag_codes(self) -> torch.Tensor:
        weight = self.weight.float()
        view = weight.reshape(self.n_v, self.crb_rows, self.n_h, self.crb_cols)
        codes = torch.round(view / self.w_interval).clamp(-128, 127)
        codes = codes.reshape(self.out_features, self.in_features).to(torch.int16)
        # Signed int8 -128 is saturated to -127 in the historical S+M metric.
        codes = codes.clamp(min=-127, max=127)
        magnitude = codes.abs().reshape(self.out_features, self.in_features // 8, 8)
        kept_magnitude = magnitude.bitwise_and(self.keep_magnitude_bits.unsqueeze(-1))
        sign = codes.lt(0)
        signed = torch.where(sign, -kept_magnitude.reshape_as(codes), kept_magnitude.reshape_as(codes))
        return signed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        codes = self._quantized_smag_codes()
        view_shape = (self.n_v, self.crb_rows, self.n_h, self.crb_cols)
        scale = self.w_interval.expand(view_shape).reshape_as(self.weight)
        dequantized = codes.to(dtype=scale.dtype) * scale
        ste_weight = self.weight + (dequantized.to(self.weight.dtype) - self.weight).detach()
        return F.linear(x, ste_weight, self.bias)


def is_target_layer(name: str) -> bool:
    return name.startswith("blocks.") and any(name.endswith(suffix) for suffix in TARGET_SUFFIXES)


def install_qat_layers(model: nn.Module, payload: dict[str, Any]) -> dict[str, BitsliceQATLinear]:
    intervals = payload["intervals"]
    masks = payload["zero_plane_masks"]
    linears: dict[str, BitsliceQATLinear] = {}
    for name, module in list(model.named_modules()):
        if not is_target_layer(name):
            continue
        if name not in intervals or name not in masks:
            raise KeyError(f"Missing calibrated bit-slice parameters for {name}")
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        interval_info = intervals[name]
        replacement = BitsliceQATLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            interval_info["w_interval"],
            masks[name],
            interval_info["n_V"],
            interval_info["n_H"],
        )
        with torch.no_grad():
            replacement.weight.copy_(module.weight.detach())
            if module.bias is not None:
                replacement.bias.copy_(module.bias.detach())
        setattr(parent, child_name, replacement)
        linears[name] = replacement
    if len(linears) != 48:
        raise RuntimeError(f"Expected 48 quantized transformer linears, installed {len(linears)}")
    return linears


@torch.no_grad()
def measure_smag_rate(linears: dict[str, BitsliceQATLinear]) -> dict[str, Any]:
    total_groups = 0
    zero_by_plane = [0] * 8
    for module in linears.values():
        codes = module._quantized_smag_codes().to(torch.int16)
        rows, width = codes.shape
        grouped = codes.reshape(rows, width // 8, 8)
        magnitude = codes.abs().clamp(max=127).reshape_as(grouped)
        for bit in range(7):
            zero_by_plane[bit] += int(
                magnitude.bitwise_and(1 << bit).eq(0).all(dim=-1).sum().item()
            )
        zero_by_plane[7] += int(codes.ge(0).reshape_as(grouped).all(dim=-1).sum().item())
        total_groups += rows * (width // 8)
    total_blocks = total_groups * 8
    zero_blocks = sum(zero_by_plane)
    return {
        "groups": total_groups,
        "total_plane_blocks": total_blocks,
        "zero_block_count": zero_blocks,
        "rate_percent": 100.0 * zero_blocks / total_blocks,
        "plane_zero_block_counts": zero_by_plane,
        "plane_zero_block_rates_percent": [100.0 * value / total_groups for value in zero_by_plane],
    }


def build_transforms(model: nn.Module):
    cfg = getattr(model, "pretrained_cfg", {})
    input_size = int(cfg.get("input_size", (3, 224, 224))[-1])
    mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))
    crop_pct = float(cfg.get("crop_pct", 0.9))
    resize_size = int(input_size / crop_pct)
    interpolation = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }.get(str(cfg.get("interpolation", "bicubic")).lower(), InterpolationMode.BICUBIC)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, val_transform


def make_loaders(args, model, rank: int, world_size: int, distributed: bool):
    train_transform, val_transform = build_transforms(model)
    train_set = datasets.ImageFolder(args.data / "train", transform=train_transform)
    val_set = datasets.ImageFolder(args.data / "val", transform=val_transform)
    if train_set.classes != val_set.classes:
        raise ValueError("ImageNet train and validation class folder order differs")
    if args.eval_images > 0:
        if args.eval_images > len(val_set):
            raise ValueError(
                f"--eval-images={args.eval_images} exceeds validation size {len(val_set)}"
            )
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(val_set), generator=generator)[:args.eval_images].tolist()
        val_set = Subset(val_set, indices)
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    ) if distributed else None
    val_sampler = DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    ) if distributed else None
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **loader_options,
    )
    return train_loader, val_loader, train_sampler


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeiT W8 S+M QAT")
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
    return torch.device("cuda", local_rank), rank, world_size, distributed


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def build_optimizer(model: nn.Module, args: argparse.Namespace):
    decay, no_decay = [], []
    for name, parameter in unwrap(model).named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
    )


def build_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def set_lr(optimizer, args, step: int, total_steps: int, steps_per_epoch: int) -> None:
    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch))
    if step < warmup_steps:
        factor = (step + 1) / warmup_steps
    else:
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = args.lr * factor
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, distributed: bool) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits; refusing to report validation accuracy")
        top5 = logits.topk(5, dim=1).indices
        totals[0] += (top5[:, 0] == targets).sum().item()
        totals[1] += (top5 == targets[:, None]).any(dim=1).sum().item()
        totals[2] += targets.numel()
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return {
        "top1": 100.0 * (totals[0] / totals[2]).item(),
        "top5": 100.0 * (totals[1] / totals[2]).item(),
        "images": int(totals[2].item()),
    }


def append_metric(path: Path, row: dict[str, Any], rank: int) -> None:
    if rank != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def save_checkpoint(path, model, optimizer, scaler, epoch, global_step, args, mask_payload, rank):
    if rank != 0:
        return
    payload = {
        "model": unwrap(model).state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "arch": args.arch,
        "source_checkpoint": str(args.source_checkpoint),
        "source_checkpoint_sha256": mask_payload["source_checkpoint_sha256"],
        "mask_artifact": str(args.mask_artifact),
        "target_smag_block_zero_rate_percent": args.target_rate,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "batch_size_per_gpu": args.batch_size,
    }
    atomic_save(path, payload)


def run_benchmark(model, loader, sampler, optimizer, scaler, args, device, rank, world_size):
    if args.benchmark_steps <= 0 or args.benchmark_warmup < 0:
        raise ValueError("benchmark-steps must be positive and benchmark-warmup non-negative")
    if sampler is not None:
        sampler.set_epoch(0)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(loader)
    start = time.time() if args.benchmark_warmup == 0 else None
    end = None
    images_per_step = args.batch_size * world_size
    total_steps = args.benchmark_warmup + args.benchmark_steps
    for step_index in range(total_steps):
        images, targets = next(iterator)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=not args.no_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if args.benchmark_warmup > 0 and step_index + 1 == args.benchmark_warmup:
            torch.cuda.synchronize(device)
            start = time.time()
        if step_index + 1 == total_steps:
            torch.cuda.synchronize(device)
            end = time.time()
    if start is None or end is None:
        raise RuntimeError("Benchmark timer failed to record the measured interval")
    elapsed = end - start
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        samples = args.benchmark_steps * images_per_step
        print(
            f"BENCH batch/gpu={args.batch_size} global_batch={images_per_step} "
            f"steps={args.benchmark_steps} elapsed={elapsed:.2f}s "
            f"steps/s={args.benchmark_steps / elapsed:.3f} "
            f"images/s={samples / elapsed:.1f} "
            f"peak_mem_gib={torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mask_artifact is None:
        args.mask_artifact = args.output_dir / "mask_init.pt"
    if args.prepare_only:
        prepare_mask_artifact(args, args.mask_artifact)
        return
    if not args.mask_artifact.is_file():
        raise FileNotFoundError(
            f"Mask artifact missing: {args.mask_artifact}. Run with --prepare-only first."
        )

    device, rank, world_size, distributed = init_distributed()
    seed_everything(args.seed, rank)
    torch.backends.cudnn.benchmark = True
    if rank == 0:
        print(
            f"Dense-initialized S+M bit-slice QAT: target={args.target_rate:.4f}%, "
            f"global batch={args.batch_size * world_size}, GPUs={world_size}, epochs={args.epochs}",
            flush=True,
        )

    mask_payload = torch.load(args.mask_artifact, map_location="cpu", weights_only=False)
    if mask_payload.get("format") != "deit_dense_w8_smag_bitplane_block_mask_v1":
        raise RuntimeError("Unrecognized bit-slice mask artifact format")
    if abs(float(mask_payload["target_rate_percent"]) - args.target_rate) > 1e-6:
        raise RuntimeError("CLI target rate does not match mask artifact target")
    source_state = torch.load(args.source_checkpoint, map_location="cpu", weights_only=True)
    model = timm.create_model(args.arch, pretrained=False)
    incompatible = model.load_state_dict(source_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Dense checkpoint did not load strictly")
    linears = install_qat_layers(model, mask_payload)
    resume_payload = None
    if args.resume_from is not None:
        resume_payload = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if resume_payload.get("arch") != args.arch:
            raise RuntimeError("Resume checkpoint architecture does not match --arch")
        if resume_payload.get("source_checkpoint_sha256") != mask_payload["source_checkpoint_sha256"]:
            raise RuntimeError("Resume checkpoint was created from a different dense source checkpoint")
        if abs(float(resume_payload.get("target_smag_block_zero_rate_percent", -1.0)) - args.target_rate) > 1e-6:
            raise RuntimeError("Resume checkpoint target sparsity does not match --target-rate")
        if int(resume_payload.get("batch_size_per_gpu", -1)) != args.batch_size:
            raise RuntimeError("Resume checkpoint per-GPU batch size does not match --batch-size")
        model.load_state_dict(resume_payload["model"], strict=True)
    model.to(device)
    if distributed:
        # Fixed quantization/mask buffers are independently loaded from the
        # same artifact on every rank; skip NCCL buffer broadcast because it
        # does not support the int16 bit-mask buffer dtype.
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
        )
    train_loader, val_loader, train_sampler = make_loaders(
        args, unwrap(model), rank, world_size, distributed
    )
    optimizer = build_optimizer(model, args)
    scaler = build_grad_scaler(enabled=not args.no_amp)
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        scaler.load_state_dict(resume_payload["scaler"])
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)
    global_step = int(resume_payload["global_step"]) if resume_payload is not None else 0
    resume_start_step = global_step
    if global_step > total_steps:
        raise RuntimeError(
            f"Resume checkpoint is already at step {global_step}, beyond requested total {total_steps}"
        )
    start_epoch, resume_batches = divmod(global_step, steps_per_epoch)
    best_top1 = args.resume_best_top1 if resume_payload is not None and args.resume_best_top1 is not None else 0.0
    if resume_payload is not None and rank == 0:
        if not args.resume_from.is_file():
            raise FileNotFoundError(args.resume_from)
        best_path = args.output_dir / "best.pt"
        if not best_path.exists():
            shutil.copy2(args.resume_from, best_path)
        print(
            f"Resuming at global_step={global_step}; epoch={start_epoch + 1}/{args.epochs}, "
            f"skip_batches={resume_batches}; total_steps={total_steps}; "
            f"best Top-1 so far={best_top1:.3f}%",
            flush=True,
        )
    if resume_payload is not None:
        del resume_payload

    if args.benchmark_only:
        run_benchmark(
            model, train_loader, train_sampler, optimizer, scaler,
            args, device, rank, world_size,
        )
        if distributed:
            dist.destroy_process_group()
        return

    if rank == 0:
        print(
            f"Train images={len(train_loader.dataset):,}; steps/epoch={steps_per_epoch:,}; "
            f"dense baseline Top-1=81.982% (ImageNet val 50k)",
            flush=True,
        )
        initial_sparsity = measure_smag_rate(linears)
        print(
            f"Initial masked W8 validation starting; measured S+M rate="
            f"{initial_sparsity['rate_percent']:.6f}%",
            flush=True,
        )
    initial_metrics = evaluate(model, val_loader, device, distributed)
    if rank == 0:
        initial_sparsity = measure_smag_rate(linears)
        initial_row = {
            "phase": "resume_start" if args.resume_from is not None else "masked_w8_initial",
            "epoch": global_step / steps_per_epoch,
            "global_step": global_step,
            **initial_metrics,
            "smag": initial_sparsity,
            "delta_top1_vs_dense_pp": initial_metrics["top1"] - 81.982,
        }
        append_metric(args.output_dir / "metrics.jsonl", initial_row, rank)
        print(
            f"{'RESUME_START' if args.resume_from is not None else 'INITIAL'} "
            f"Top-1={initial_metrics['top1']:.3f}% "
            f"({initial_row['delta_top1_vs_dense_pp']:+.3f} pp vs dense), "
            f"Top-5={initial_metrics['top5']:.3f}%, "
            f"S+M={initial_sparsity['rate_percent']:.6f}%",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = torch.zeros(1, dtype=torch.float64, device=device)
        seen = torch.zeros(1, dtype=torch.float64, device=device)
        epoch_start = time.time()
        for batch_index, (images, targets) in enumerate(train_loader):
            if epoch == start_epoch and batch_index < resume_batches:
                continue
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            set_lr(optimizer, args, global_step, total_steps, steps_per_epoch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=not args.no_amp):
                logits = model(images)
                loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            running_loss += loss.detach().double() * targets.numel()
            seen += targets.numel()
            if rank == 0 and global_step % 100 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} "
                    f"loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.3g}",
                    flush=True,
                )
        if distributed:
            dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(seen, op=dist.ReduceOp.SUM)
        mean_loss = (running_loss / seen.clamp_min(1)).item()
        if rank == 0:
            print(
                f"EPOCH {epoch + 1} loss={mean_loss:.5f} "
                f"elapsed={time.time() - epoch_start:.1f}s",
                flush=True,
            )
        append_metric(
            args.output_dir / "metrics.jsonl",
            {"phase": "train", "epoch": epoch + 1, "global_step": global_step, "loss": mean_loss},
            rank,
        )

        should_eval = ((epoch + 1) % args.eval_every == 0) or (epoch + 1 == args.epochs)
        should_checkpoint = ((epoch + 1) % args.checkpoint_every == 0) or (epoch + 1 == args.epochs)
        if should_eval:
            validation = evaluate(model, val_loader, device, distributed)
            if rank == 0:
                sparsity = measure_smag_rate(linears)
                row = {
                    "phase": "qat_val",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    **validation,
                    "smag": sparsity,
                    "delta_top1_vs_dense_pp": validation["top1"] - 81.982,
                }
                append_metric(args.output_dir / "metrics.jsonl", row, rank)
                print(
                    f"VAL epoch={epoch + 1} Top-1={validation['top1']:.3f}% "
                    f"({row['delta_top1_vs_dense_pp']:+.3f} pp vs dense), "
                    f"Top-5={validation['top5']:.3f}%, "
                    f"S+M={sparsity['rate_percent']:.6f}%",
                    flush=True,
                )
                if validation["top1"] > best_top1:
                    best_top1 = validation["top1"]
                    save_checkpoint(
                        args.output_dir / "best.pt", model, optimizer, scaler,
                        epoch, global_step, args, mask_payload, rank,
                    )
        if should_checkpoint:
            save_checkpoint(
                args.output_dir / "last.pt", model, optimizer, scaler,
                epoch, global_step, args, mask_payload, rank,
            )
        if distributed:
            dist.barrier()
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    if rank == 0:
        summary = {
            "model_arch": args.arch,
            "source_checkpoint": str(args.source_checkpoint),
            "source_checkpoint_sha256": mask_payload["source_checkpoint_sha256"],
            "initial_target_smag_rate_percent": args.target_rate,
            "best_validation_top1": best_top1,
            "epochs": args.epochs,
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "max_train_steps": args.max_train_steps,
            "completed_global_steps": global_step,
            "resumed_from": str(args.resume_from) if args.resume_from is not None else None,
            "start_global_step": resume_start_step,
            "eval_images": args.eval_images or len(val_loader.dataset),
            "mask_artifact": str(args.mask_artifact),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Training finished; summary={args.output_dir / 'summary.json'}", flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
