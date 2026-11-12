#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${IMAGENET_ROOT:?Set IMAGENET_ROOT to the ImageNet root containing train/ and val/}"
NPROC="${NPROC_PER_NODE:-4}"
ARCH="deit_base_patch16_224.fb_in1k"
SOURCE="$ROOT/checkpoints/deit_base_patch16_224.fb_in1k_fp32_state_dict.pth"
PTQ_DIR="$ROOT/results/deit_base_ptq4vit_w8a8_q8k4p8v4_patchfp32"
PTQ_ARTIFACT="$PTQ_DIR/deit_base_ptq4vit_transformer_w8a8_q8k4p8v4_patchfp32.pt"
MASK="$ROOT/results/mask_75_globalcoverage/mask_init.pt"
PILOT="$ROOT/results/qat_75_globalcoverage_pilot_500step"
OUT="$ROOT/results/qat_75_globalcoverage_35ep"
EVAL="$ROOT/results/deit_base_bitslice_w8a8_eval"

mkdir -p "$ROOT/checkpoints" "$ROOT/results"
if [[ ! -f "$SOURCE" ]]; then
  python "$ROOT/prepare_dense_checkpoint.py" --output "$SOURCE"
fi

python "$ROOT/run_deit_base_ptq4vit_w8a8.py" \
  --checkpoint "$SOURCE" --data "$DATA" --output "$PTQ_DIR" \
  --calib-size 32 --calib-batch-size 4 --eval-batch-size 64 \
  --workers 8 --max-eval-images 128

python "$ROOT/train_deit_dense_smag_bitslice_qat.py" \
  --source-checkpoint "$SOURCE" --ptq-artifact "$PTQ_ARTIFACT" \
  --data "$DATA" --output-dir "$OUT" --mask-artifact "$MASK" \
  --arch "$ARCH" --target-rate 75 --mask-selection global-coverage \
  --ptq-calib-images 32 --ptq-calib-seed 17 \
  --epochs 35 --batch-size 256 --workers 8 \
  --eval-every 1 --checkpoint-every 1 \
  --lr 1e-4 --weight-decay 0.05 --warmup-epochs 5 --seed 17 \
  --prepare-only

torchrun --standalone --nproc_per_node="$NPROC" \
  "$ROOT/train_deit_dense_smag_bitslice_qat.py" \
  --source-checkpoint "$SOURCE" --ptq-artifact "$PTQ_ARTIFACT" \
  --data "$DATA" --output-dir "$PILOT" --mask-artifact "$MASK" \
  --arch "$ARCH" --target-rate 75 --mask-selection global-coverage \
  --ptq-calib-images 32 --ptq-calib-seed 17 \
  --epochs 35 --batch-size 256 --workers 8 \
  --eval-every 1 --checkpoint-every 1 \
  --lr 1e-4 --weight-decay 0.05 --warmup-epochs 5 --seed 17 \
  --max-train-steps 500

torchrun --standalone --nproc_per_node="$NPROC" \
  "$ROOT/train_deit_dense_smag_bitslice_qat.py" \
  --source-checkpoint "$SOURCE" --ptq-artifact "$PTQ_ARTIFACT" \
  --data "$DATA" --output-dir "$OUT" --mask-artifact "$MASK" \
  --arch "$ARCH" --target-rate 75 --mask-selection global-coverage \
  --ptq-calib-images 32 --ptq-calib-seed 17 \
  --epochs 35 --batch-size 256 --workers 8 \
  --eval-every 1 --checkpoint-every 1 \
  --lr 1e-4 --weight-decay 0.05 --warmup-epochs 5 --seed 17 \
  --resume-from "$PILOT/last.pt" --resume-best-top1 67.950

python "$ROOT/eval_deit_bitslice_w8a8.py" \
  --checkpoint "$OUT/last.pt" --mask-artifact "$MASK" \
  --data "$DATA" --output-dir "$EVAL" \
  --calib-size 32 --calib-batch-size 4 --eval-batch-size 64 \
  --workers 8 --max-eval-images 0
