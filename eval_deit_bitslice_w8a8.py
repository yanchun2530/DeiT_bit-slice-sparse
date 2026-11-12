#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, time
import os
from pathlib import Path

import timm
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'PTQ4ViT'))
import run_deit_base_ptq4vit_w8a8 as ptq
from configs import PTQ4ViT as quant_cfg

DEFAULT_CKPT = ROOT / 'results/qat_75_globalcoverage_35ep/last.pt'
DEFAULT_MASK = ROOT / 'results/mask_75_globalcoverage/mask_init.pt'
DEFAULT_DATA = Path(os.environ.get('IMAGENET_ROOT', ROOT / 'data/imagenet'))
DEFAULT_OUT = ROOT / 'results/deit_base_bitslice_w8a8_eval'


def apply_smag_mask(weight_module, interval, mask):
    weight = weight_module.weight
    n_v = int(getattr(weight_module, 'n_V', interval.shape[0]))
    n_h = int(getattr(weight_module, 'n_H', interval.shape[2]))
    w_qmax = int(getattr(weight_module, 'w_qmax', 128))
    rows, cols = weight_module.out_features // n_v, weight_module.in_features // n_h
    view = weight.detach().float().reshape(n_v, rows, n_h, cols)
    codes = torch.round(view / interval.to(view.device)).clamp(
        -w_qmax, w_qmax - 1
    ).reshape(weight_module.out_features, weight_module.in_features).to(torch.int16)
    codes = codes.clamp(-127, 127)
    grouped_mag = codes.abs().reshape(weight_module.out_features, weight_module.in_features // 8, 8)
    mask = mask.to(device=weight.device, dtype=torch.bool)
    keep_bits = torch.zeros(mask.shape[:2], device=weight.device, dtype=torch.int16)
    for bit in range(7):
        keep_bits |= ((~mask[:, :, bit]).to(torch.int16) << bit)
    masked_mag = grouped_mag.bitwise_and(keep_bits.unsqueeze(-1)).reshape_as(codes)
    masked_codes = torch.where(codes < 0, -masked_mag, masked_mag)
    scale = interval.to(weight.device).expand(n_v, rows, n_h, cols).reshape_as(weight)
    weight.data.copy_(masked_codes.reshape_as(weight).to(weight.dtype) * scale.to(weight.dtype))


@torch.no_grad()
def smag_rate(wrapped):
    total_groups = 0
    zeros = [0] * 8
    for module in wrapped.values():
        view = module.weight.detach().float().reshape(
            module.n_V, module.crb_rows, module.n_H, module.crb_cols
        )
        codes = torch.round(view / module.w_interval).clamp(
            -module.w_qmax, module.w_qmax - 1
        ).reshape(module.out_features, module.in_features).to(torch.int16).clamp(-127, 127)
        groups = codes.shape[0] * (codes.shape[1] // 8)
        mag = codes.abs().reshape(codes.shape[0], codes.shape[1] // 8, 8)
        for bit in range(7):
            zeros[bit] += int(mag.bitwise_and(1 << bit).eq(0).all(-1).sum().item())
        zeros[7] += int(codes.ge(0).reshape(codes.shape[0], codes.shape[1] // 8, 8).all(-1).sum().item())
        total_groups += groups
    return {'groups': total_groups, 'zero_blocks': sum(zeros),
            'total_plane_blocks': total_groups * 8,
            'rate_percent': 100.0 * sum(zeros) / (total_groups * 8),
            'zero_blocks_by_plane': zeros}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--mask-artifact', type=Path, default=DEFAULT_MASK)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--calib-size', type=int, default=32)
    parser.add_argument('--calib-batch-size', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-eval-images', type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading QAT checkpoint: {args.checkpoint}', flush=True)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    qat_state = payload['model']
    mask_payload = torch.load(args.mask_artifact, map_location='cpu', weights_only=False)
    model = timm.create_model(payload['arch'], pretrained=False)
    base_state = {
        k: v for k, v in qat_state.items()
        if not k.endswith(('.w_interval', '.zero_plane_mask', '.keep_magnitude_bits'))
    }
    result = model.load_state_dict(base_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f'Checkpoint mismatch: {result}')
    linears = {name: module for name, module in model.named_modules()
               if ptq.is_target_name(name)}
    if len(linears) != 48:
        raise RuntimeError(f'Expected 48 target linears, got {len(linears)}')
    for name, module in linears.items():
        interval = mask_payload['intervals'][name]['w_interval']
        ckpt_interval = qat_state[name + '.w_interval'].cpu()
        if not torch.allclose(interval.float(), ckpt_interval.float(), rtol=1e-6, atol=1e-8):
            raise RuntimeError(f'W8 interval mismatch for {name}')
        apply_smag_mask(module, interval, mask_payload['zero_plane_masks'][name])
    ptq.patch_attention_matmuls(model)
    model.to(device).eval()

    loaders = ptq.ViTImageNetLoaderGenerator(
        str(args.data), 'imagenet', args.calib_size, args.eval_batch_size,
        args.workers, kwargs={'model': model},
    )
    calib_loader = loaders.calib_loader(num=args.calib_size, seed=17)
    val_loader = loaders.test_loader(batch_size=args.eval_batch_size)
    quant_cfg.matmul_name_list = []
    ptq.set_attention_quantization(model, True)
    wrapped = ptq.wrap_transformer_block_linears(model)

    print(f'Calibrating PTQ4ViT W8A8 on {args.calib_size} ImageNet-train images', flush=True)
    start = time.time()
    calibrator = ptq.HessianQuantCalibrator(
        model, wrapped, calib_loader, sequential=False, batch_size=args.calib_batch_size
    )
    calibrator.batching_quant_calib()
    del calibrator
    torch.cuda.empty_cache()

    # Reapply the fixed learned S+M plane mask to calibrated W8 codes.
    for name, module in wrapped.items():
        interval = mask_payload['intervals'][name]['w_interval']
        module.w_interval.copy_(interval.to(module.w_interval.device))
        apply_smag_mask(module, interval, mask_payload['zero_plane_masks'][name])
    rate = smag_rate(wrapped)
    print('Post-calibration masked W8 S+M block rate=%.6f%%' % rate['rate_percent'], flush=True)

    print('Running full ImageNet validation', flush=True)
    metrics = ptq.evaluate(model, val_loader, device, args.max_eval_images)
    result = {
        'checkpoint': str(args.checkpoint),
        'mask_artifact': str(args.mask_artifact),
        'quantization': 'PTQ4ViT transformer Linear W8A8; attention Q8/K4/P8/V4 QDQ; FP32 patch embedding/classifier',
        'calibration': {'dataset': 'ImageNet train', 'images': args.calib_size,
                        'batch_size': args.calib_batch_size, 'seed': 17,
                        'seconds': time.time() - start},
        'validation': metrics,
        'weight_smag_block_rate_after_w8a8_calibration': rate,
        'note': 'Learned S+M magnitude-plane masks were preserved and reapplied to calibrated W8 codes. Activations use PTQ4ViT A8 quantize-dequantize.',
    }
    out = args.output_dir / 'metrics.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
