#!/usr/bin/env python3
"""Search S+M weight bit-slice block masks for a sparse DeiT-Base checkpoint.

The search raises the S+M plane-block zero rate by clearing magnitude planes
in 8-weight groups. For each plane, groups are ranked by a diagonal
activation-weighted squared perturbation proxy collected from the PTQ
calibration inputs. This is a post-training W8A8 experiment; it does not
change or overwrite the source GraNet checkpoint.
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
from typing import Any

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent
PTQ4VIT_DIR = PROJECT_DIR / "PTQ4ViT"
if PTQ4VIT_DIR.is_dir():
    sys.path.insert(0, str(PTQ4VIT_DIR))

import run_deit_base_ptq4vit_w8a8 as ptq


DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints/deit_base_granet75.pt"
DEFAULT_DATA = Path(os.environ.get("IMAGENET_ROOT", PROJECT_DIR / "data/imagenet"))
DEFAULT_OUTPUT = PROJECT_DIR / "results/granet_weight_smag_bitslice_search"
TARGET_TYPES = ("qkv", "proj", "fc1", "fc2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arch", default="deit_base_patch16_224.fb_in1k")
    parser.add_argument("--targets", default="60,65,70,75,80,85")
    parser.add_argument("--images", type=int, default=1000)
    parser.add_argument("--canonical-images", type=int, default=10_000)
    parser.add_argument("--ptq-calib-images", type=int, default=32)
    parser.add_argument("--ptq-calib-seed", type=int, default=17)
    parser.add_argument("--ptq-calib-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-target", type=float, default=75.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def parse_targets(raw: str) -> list[float]:
    try:
        values = sorted({float(value.strip()) for value in raw.split(",")})
    except ValueError as exc:
        raise ValueError(f"Invalid --targets value: {raw}") from exc
    if not values or any(not 0.0 < value < 100.0 for value in values):
        raise ValueError("Target block sparsities must be percentages between 0 and 100")
    return values


def summarize_smag(codes_by_name: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Count zero blocks per S+M plane; each group is 8 values within one row."""
    plane_zeros = [0] * 8
    group_count = value_count = exact_zero_count = minus128_count = 0
    by_type: dict[str, dict[str, Any]] = {}
    per_layer: dict[str, dict[str, Any]] = {}

    for name, codes in sorted(codes_by_name.items()):
        codes = codes.to(dtype=torch.int16)
        if codes.ndim != 2 or codes.shape[1] % 8:
            raise ValueError(f"Expected [out,in] with in divisible by 8: {name} {tuple(codes.shape)}")
        rows, input_width = codes.shape
        groups = rows * (input_width // 8)
        grouped = codes.reshape(rows, input_width // 8, 8)
        magnitude = codes.abs().clamp(max=127).reshape_as(grouped)
        sign = codes.lt(0).reshape_as(grouped)
        layer_zeros = []
        for bit in range(7):
            zero_count = int((magnitude.bitwise_and(1 << bit).eq(0).all(dim=-1)).sum().item())
            plane_zeros[bit] += zero_count
            layer_zeros.append(zero_count)
        sign_zero = int((~sign).all(dim=-1).sum().item())
        plane_zeros[7] += sign_zero
        layer_zeros.append(sign_zero)
        exact_zeros = int(codes.eq(0).sum().item())
        negative_128 = int(codes.eq(-128).sum().item())
        row = {
            "groups": groups,
            "values": codes.numel(),
            "exact_zero_code_count": exact_zeros,
            "exact_zero_code_rate": exact_zeros / codes.numel(),
            "plane_zero_block_counts": layer_zeros,
            "plane_zero_block_rates": [value / groups for value in layer_zeros],
            "all_plane_zero_block_rate": sum(layer_zeros) / (groups * 8),
        }
        per_layer[name] = row
        kind = name.rsplit(".", 1)[-1]
        bucket = by_type.setdefault(kind, {
            "groups": 0, "values": 0, "exact_zero_code_count": 0,
            "plane_zero_block_counts": [0] * 8,
        })
        bucket["groups"] += groups
        bucket["values"] += codes.numel()
        bucket["exact_zero_code_count"] += exact_zeros
        for bit, count in enumerate(layer_zeros):
            bucket["plane_zero_block_counts"][bit] += count
        group_count += groups
        value_count += codes.numel()
        exact_zero_count += exact_zeros
        minus128_count += negative_128

    if group_count == 0:
        raise ValueError("No target Linear weight codes found")
    return {
        "groups": group_count,
        "values": value_count,
        "total_plane_blocks": group_count * 8,
        "plane_zero_block_counts": plane_zeros,
        "plane_zero_block_rates": [value / group_count for value in plane_zeros],
        "zero_block_count": sum(plane_zeros),
        "all_plane_zero_block_rate": sum(plane_zeros) / (group_count * 8),
        "exact_zero_code_count": exact_zero_count,
        "exact_zero_code_rate": exact_zero_count / value_count,
        "minus128_code_count": minus128_count,
        "by_linear_type": {
            kind: {
                **bucket,
                "plane_zero_block_rates": [
                    value / bucket["groups"] for value in bucket["plane_zero_block_counts"]
                ],
                "all_plane_zero_block_rate": (
                    sum(bucket["plane_zero_block_counts"]) / (bucket["groups"] * 8)
                ),
            }
            for kind, bucket in sorted(by_type.items())
        },
        "per_layer": per_layer,
    }


def collect_base_records(
    wrapped: dict[str, torch.nn.Module],
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[int, int]]]:
    records: dict[str, dict[str, Any]] = {}
    ranges: dict[str, tuple[int, int]] = {}
    group_offset = 0
    for name, module in wrapped.items():
        view_shape = (module.n_V, module.crb_rows, module.n_H, module.crb_cols)
        weight_view = module.weight.detach().reshape(view_shape)
        interval_view = module.w_interval.detach().expand_as(weight_view)
        code = (weight_view / module.w_interval).round().clamp(
            -module.w_qmax, module.w_qmax - 1
        ).reshape_as(module.weight).to(torch.int16)
        rows, input_width = code.shape
        if input_width % 8:
            raise ValueError(f"Input width is not divisible by 8 for {name}: {input_width}")
        groups = rows * (input_width // 8)
        scale = interval_view.reshape_as(module.weight).detach().to(device=device, dtype=torch.float32)
        # S+M has no -128 code. Match the historical measurement by saturating it to -127.
        code = code.to(device=device)
        code = torch.where(code.eq(-128), torch.full_like(code, -127), code)
        records[name] = {
            "module": module,
            "codes": code,
            "scale": scale,
            "groups": groups,
            "rows": rows,
            "input_width": input_width,
            "x2": None,
        }
        ranges[name] = (group_offset, group_offset + groups)
        group_offset += groups
    if len(records) != 48:
        raise ValueError(f"Expected 48 transformer linears, got {len(records)}")
    return records, ranges


def collect_quantized_input_second_moments(
    model: torch.nn.Module,
    records: dict[str, dict[str, Any]],
    calib_loader,
    device: torch.device,
) -> None:
    sums = {
        name: torch.zeros(record["input_width"], dtype=torch.float64, device=device)
        for name, record in records.items()
    }
    counts = {name: 0 for name in records}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"Unexpected calibration input for {name}")
            # Use the actual quantized Linear input, including PostGelu handling for fc2.
            qx = module.quant_input(inputs[0].detach())
            flat = qx.reshape(-1, qx.shape[-1]).to(torch.float64)
            sums[name].add_(flat.square().sum(dim=0))
            counts[name] += int(flat.shape[0])
        return hook

    for name, record in records.items():
        handles.append(record["module"].register_forward_pre_hook(make_hook(name)))
    try:
        model.eval()
        with torch.inference_mode():
            for images, _ in calib_loader:
                model(images.to(device, non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()

    for name, record in records.items():
        if counts[name] == 0:
            raise RuntimeError(f"No calibration inputs captured for {name}")
        record["x2"] = (sums[name] / counts[name]).to(torch.float32)


def rank_plane_blocks(
    records: dict[str, dict[str, Any]],
    ranges: dict[str, tuple[int, int]],
    bit: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    score_parts = []
    id_parts = []
    eligible_total = 0
    bit_mask = 1 << bit
    for name, record in records.items():
        rows, input_width = record["codes"].shape
        groups_per_row = input_width // 8
        magnitude = record["codes"].abs().clamp(max=127)
        grouped_magnitude = magnitude.reshape(rows, groups_per_row, 8)
        bit_set = grouped_magnitude.bitwise_and(bit_mask).ne(0)
        eligible = bit_set.any(dim=-1)
        scale_grouped = record["scale"].reshape(rows, groups_per_row, 8)
        x2_grouped = record["x2"].reshape(1, groups_per_row, 8)
        # E[(delta-W * quantized-input)^2], using diagonal input covariance.
        perturbation = bit_set.to(torch.float32) * float(bit_mask) * scale_grouped
        score = (perturbation.square() * x2_grouped).sum(dim=-1)
        start, end = ranges[name]
        local_ids = torch.arange(start, end, dtype=torch.int64, device=device).reshape(
            rows, groups_per_row
        )
        score_parts.append(score[eligible])
        id_parts.append(local_ids[eligible])
        eligible_total += int(eligible.sum().item())
    if eligible_total == 0:
        return torch.empty(0, dtype=torch.int64), 0
    scores = torch.cat(score_parts)
    group_ids = torch.cat(id_parts)
    if scores.numel() != eligible_total or group_ids.numel() != eligible_total:
        raise RuntimeError(f"Candidate count mismatch for magnitude bit {bit}")
    order = torch.argsort(scores, stable=True)
    return group_ids.index_select(0, order).cpu(), eligible_total


def build_target_plan(
    targets: list[float],
    baseline: dict[str, Any],
) -> tuple[dict[float, dict[int, int]], dict[int, int]]:
    groups = int(baseline["groups"])
    total_blocks = int(baseline["total_plane_blocks"])
    zero_counts = baseline["plane_zero_block_counts"]
    eligible_by_bit = {bit: groups - int(zero_counts[bit]) for bit in range(7)}
    plan = {target: {} for target in targets}
    cumulative = int(baseline["zero_block_count"])
    target_counts = {target: math.ceil(target / 100.0 * total_blocks) for target in targets}

    for bit in range(7):
        eligible = eligible_by_bit[bit]
        for target in targets:
            need = max(0, target_counts[target] - cumulative)
            plan[target][bit] = min(need, eligible)
        cumulative += eligible
    for target in targets:
        achieved_blocks = int(baseline["zero_block_count"]) + sum(plan[target].values())
        if achieved_blocks < target_counts[target]:
            raise RuntimeError(
                f"Cannot reach requested target {target}% using magnitude planes; "
                f"maximum is {100 * achieved_blocks / total_blocks:.4f}%"
            )
    return plan, eligible_by_bit


def make_candidate_codes(
    target: float,
    records: dict[str, dict[str, Any]],
    ranges: dict[str, tuple[int, int]],
    plan: dict[float, dict[int, int]],
    eligible_by_bit: dict[int, int],
    rankings: dict[int, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    codes_by_name = {}
    for name, record in records.items():
        original = record["codes"]
        negative = original.lt(0)
        magnitude = original.abs().clamp(max=127).clone()
        rows, input_width = magnitude.shape
        grouped = magnitude.reshape(rows, input_width // 8, 8)
        start, end = ranges[name]
        local_group_count = end - start

        for bit in range(7):
            selected_count = plan[target][bit]
            if selected_count <= 0:
                continue
            mask_value = 1 << bit
            if selected_count >= eligible_by_bit[bit]:
                grouped.bitwise_and_(~mask_value)
                continue
            selected_ids = rankings[bit][:selected_count]
            local_ids = selected_ids[(selected_ids >= start) & (selected_ids < end)] - start
            if local_ids.numel() == 0:
                continue
            if int(local_ids.max().item()) >= local_group_count:
                raise RuntimeError(f"Bad global group index for {name}")
            selected = torch.zeros(local_group_count, dtype=torch.bool, device=device)
            selected[local_ids.to(device)] = True
            flat_groups = grouped.reshape(local_group_count, 8)
            flat_groups[selected] = flat_groups[selected].bitwise_and(~mask_value)

        signed = torch.where(negative, -magnitude, magnitude)
        codes_by_name[name] = signed
    return codes_by_name


def install_candidate(
    codes_by_name: dict[str, torch.Tensor],
    records: dict[str, dict[str, Any]],
) -> None:
    with torch.no_grad():
        for name, codes in codes_by_name.items():
            record = records[name]
            dequantized = codes.to(torch.float32) * record["scale"]
            record["module"].weight.copy_(dequantized.to(record["module"].weight.dtype))


def save_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    targets = parse_targets(args.targets)
    if args.images <= 0 or args.canonical_images < args.images:
        raise ValueError("--images must be positive and <= --canonical-images")
    if args.ptq_calib_images <= 0 or args.ptq_calib_batch_size <= 0 or args.batch_size <= 0:
        raise ValueError("Calibration and evaluation batch sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PTQ4ViT calibration")
    if not (args.data / "train").is_dir() or not (args.data / "val").is_dir():
        raise FileNotFoundError(f"Expected ImageNet train/val folders under {args.data}")

    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This experiment requires a CUDA device")
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model = timm.create_model(args.arch, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    ptq.patch_attention_matmuls(model)
    model.to(device).eval()

    loaders = ptq.ViTImageNetLoaderGenerator(
        str(args.data), "imagenet", args.ptq_calib_images, args.batch_size,
        args.workers, kwargs={"model": model},
    )
    dataset = loaders.test_set
    if args.canonical_images > len(dataset):
        raise ValueError(f"Canonical sample exceeds validation size: {len(dataset)}")
    canonical_indices = np.random.Generator(np.random.PCG64(args.seed)).choice(
        len(dataset), size=args.canonical_images, replace=False
    )
    indices = canonical_indices[:args.images]
    sample_hash = sha256_indices(indices)
    val_loader = DataLoader(
        Subset(dataset, indices.tolist()), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    calib_loader = loaders.calib_loader(num=args.ptq_calib_images, seed=args.ptq_calib_seed)

    result: dict[str, Any] = {
        "experiment": "GraNet DeiT-Base S+M magnitude-plane block pruning search",
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "source_completed_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "source_global_step": int(checkpoint.get("global_step", -1)),
        "source_mask_sparsity": checkpoint.get("sparsity_target"),
        "model_arch": args.arch,
        "scope": "48 transformer-block Linear weights; patch embedding and classifier excluded",
        "grouping": "each output row, consecutive groups of 8 along the input dimension",
        "block_definition": "one S+M plane is zero for a group when all 8 encoded bits are zero",
        "method": (
            "For each target, clear magnitude planes b0..b6 in ascending order. "
            "Within the active plane, choose the requested number of nonzero blocks by "
            "ascending sum(delta_weight^2 * E[quantized_input^2]) using diagonal input covariance."
        ),
        "sign_plane": "preserved; only S+M magnitude planes b0-b6 are pruned",
        "s_m_minus128": "saturated to -127 to match the historical S+M metric",
        "quantization": {
            "method": "PTQ4ViT Hessian W8A8 with Q8/K4/P8/V4 attention QDQ",
            "calibration_images": args.ptq_calib_images,
            "calibration_seed": args.ptq_calib_seed,
            "calibration_batch_size": args.ptq_calib_batch_size,
        },
        "validation": {
            "dataset": "ImageNet val",
            "images": args.images,
            "canonical_images": args.canonical_images,
            "seed": args.seed,
            "sample_index_sha256": sample_hash,
        },
        "targets_percent": targets,
        "candidates": [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    json_path = args.output / "search.json"
    save_json(json_path, result)

    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"Loaded epoch {result['source_completed_epoch']} checkpoint "
        f"(step {result['source_global_step']}); evaluating {args.images} images "
        f"(sample SHA-256 {sample_hash})",
        flush=True,
    )
    print(
        f"Checkpoint SHA-256={result['source_checkpoint_sha256']}; "
        f"mask target={result['source_mask_sparsity']}",
        flush=True,
    )

    ptq.set_attention_quantization(model, False)
    source_fp32_metrics = ptq.evaluate(model, val_loader, device)
    print(f"GraNet source FP32 metrics: {source_fp32_metrics}", flush=True)

    ptq.quant_cfg.matmul_name_list = []
    ptq.set_attention_quantization(model, True)
    wrapped = ptq.wrap_transformer_block_linears(model)
    calibrator = ptq.HessianQuantCalibrator(
        model, wrapped, calib_loader, sequential=False,
        batch_size=args.ptq_calib_batch_size,
    )
    calibration_started = time.time()
    calibrator.batching_quant_calib()
    calibration_seconds = time.time() - calibration_started
    del calibrator
    torch.cuda.empty_cache()
    for name, module in wrapped.items():
        if module.mode != "quant_forward" or not bool(module.calibrated):
            raise RuntimeError(f"PTQ4ViT did not calibrate {name}")

    ptq_baseline = ptq.evaluate(model, val_loader, device)
    print(
        f"PTQ4ViT W8A8 baseline: {ptq_baseline}; "
        f"calibration={calibration_seconds / 60:.2f} min",
        flush=True,
    )

    records, ranges = collect_base_records(wrapped, device)
    baseline_codes = {name: record["codes"] for name, record in records.items()}
    baseline_summary = summarize_smag(baseline_codes)
    print(
        f"Baseline S+M plane-block zero rate="
        f"{100 * baseline_summary['all_plane_zero_block_rate']:.4f}%; "
        f"plane rates={['%.3f%%' % (100 * x) for x in baseline_summary['plane_zero_block_rates']]}",
        flush=True,
    )

    print("Collecting calibrated Linear input second moments...", flush=True)
    collect_quantized_input_second_moments(model, records, calib_loader, device)
    plan, eligible_by_bit = build_target_plan(targets, baseline_summary)
    rankings: dict[int, torch.Tensor] = {}
    for bit in range(7):
        max_k = max(plan[target][bit] for target in targets)
        if max_k <= 0:
            rankings[bit] = torch.empty(0, dtype=torch.int64)
            continue
        print(f"Ranking magnitude b{bit} blocks (up to {max_k:,})...", flush=True)
        ranking, eligible = rank_plane_blocks(records, ranges, bit, device)
        if eligible != eligible_by_bit[bit]:
            raise RuntimeError(
                f"Bit b{bit} eligible blocks changed: {eligible} != {eligible_by_bit[bit]}"
            )
        rankings[bit] = ranking[:max_k]

    result["source_fp32_metrics"] = source_fp32_metrics
    result["ptq4vit_baseline_metrics"] = ptq_baseline
    result["ptq_calibration_seconds"] = calibration_seconds
    result["baseline_smag"] = baseline_summary
    result["plane_selection_counts"] = {
        str(target): {f"b{bit}": count for bit, count in plan[target].items()}
        for target in targets
    }
    save_json(json_path, result)

    for target in targets:
        started = time.time()
        candidate_codes = make_candidate_codes(
            target, records, ranges, plan, eligible_by_bit, rankings, device
        )
        candidate_summary = summarize_smag(candidate_codes)
        install_candidate(candidate_codes, records)
        metrics = ptq.evaluate(model, val_loader, device)
        row = {
            "target_block_zero_rate_percent": target,
            "achieved_smag_block_zero_rate_percent": (
                100 * candidate_summary["all_plane_zero_block_rate"]
            ),
            "delta_smag_block_zero_rate_pp": (
                100 * (
                    candidate_summary["all_plane_zero_block_rate"]
                    - baseline_summary["all_plane_zero_block_rate"]
                )
            ),
            "magnitude_plane_blocks_cleared": {
                f"b{bit}": int(plan[target][bit]) for bit in range(7)
            },
            "exact_zero_weight_code_rate_percent": (
                100 * candidate_summary["exact_zero_code_rate"]
            ),
            "top1_percent": metrics["top1"],
            "top5_percent": metrics["top5"],
            "delta_top1_vs_ptq_baseline_pp": metrics["top1"] - ptq_baseline["top1"],
            "delta_top5_vs_ptq_baseline_pp": metrics["top5"] - ptq_baseline["top5"],
            "elapsed_seconds": time.time() - started,
            "by_linear_type": candidate_summary["by_linear_type"],
        }
        result["candidates"].append(row)
        print(
            f"target={target:.2f}% achieved={row['achieved_smag_block_zero_rate_percent']:.4f}% "
            f"Top-1={metrics['top1']:.3f}% ({row['delta_top1_vs_ptq_baseline_pp']:+.3f} pp) "
            f"Top-5={metrics['top5']:.3f}% ({row['delta_top5_vs_ptq_baseline_pp']:+.3f} pp)",
            flush=True,
        )
        save_json(json_path, result)

        if args.save_target is not None and abs(target - args.save_target) < 1e-6:
            artifact_path = args.output / f"candidate_smag_block{target:g}.pt"
            torch.save(
                {
                    "format": (
                        "PTQ4ViT W8 code weights after S+M bit-slice block pruning; "
                        "signed decoded codes with per-module weight intervals"
                    ),
                    "source_checkpoint": str(args.checkpoint),
                    "source_checkpoint_sha256": result["source_checkpoint_sha256"],
                    "model_arch": args.arch,
                    "target_block_zero_rate_percent": target,
                    "achieved_block_zero_rate_percent": (
                        100 * candidate_summary["all_plane_zero_block_rate"]
                    ),
                    "codes": {
                        name: codes.detach().cpu().to(torch.int8)
                        for name, codes in candidate_codes.items()
                    },
                    "weight_intervals": {
                        name: record["module"].w_interval.detach().cpu()
                        for name, record in records.items()
                    },
                    "metrics": metrics,
                },
                artifact_path,
            )
            row["artifact"] = str(artifact_path)
            save_json(json_path, result)

    print(f"Saved search results: {json_path}", flush=True)


if __name__ == "__main__":
    main()
