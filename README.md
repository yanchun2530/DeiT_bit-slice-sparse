# DeiT Bit-Slice Sparse QAT

复现 DeiT-Base 上的 **75% S+M bit-slice 块 mask + QAT** 实验。训练从 dense DeiT 初始化，不使用 GraNet 的 75% 元素剪枝 checkpoint。

## 实验口径

- 模型：`deit_base_patch16_224.fb_in1k`，48 个 Transformer Linear 层。
- Mask：W8 S+M 权重的全局 `global-coverage` 选择；按每个输出行的连续 8 个输入权重分组，每组每个 bit plane 形成一个块；mask 选幅值 plane b0–b6，符号 plane b7 不由目标 mask 清除；目标零块率 75%，mask 在 QAT 中固定。
- QAT：35 轮，4×RTX 3090，每卡 batch 256、global batch 1,024；AdamW，学习率 `1e-4`，weight decay `0.05`，5 轮 warmup，seed 17，FP16 AMP。
- QAT 验证（W8 权重 fake-quant、FP16 AMP 激活）：Top-1 **78.022%** / Top-5 **93.504%**；训练后 W8 S+M 块零率 **76.3207%**。
- W8A8 评测：PTQ4ViT 对 Transformer Linear 做 W8A8 fake-QDQ，attention 使用 Q8/K4/P8/V4 QDQ，patch embedding 与分类 head 为 FP32。ImageNet 验证集 50k：Top-1 **77.814%** / Top-5 **93.416%**，比 dense W8A8 Top-1 低 **3.878 个百分点**。
- 这是 fake-quant/dequant 软件评测；没有使用 packed INT8/INT4 内核。

实验配置和结果摘要见 [`experiment.json`](experiment.json)。PTQ4ViT 依赖固定在上游 commit `b42c790c2811e7835d435a9f9a6c64a079a7483a`。

## 环境与数据

需要 Linux、CUDA 12.1、Python 3.10、4 张可用 GPU，以及 ImageNet-1k 数据目录：

```text
IMAGENET_ROOT/
├── train/<class>/*.JPEG
└── val/<class>/*.JPEG
```

克隆时取 PTQ4ViT 子模块：

```bash
git clone --recurse-submodules https://github.com/yanchun2530/DeiT_bit-slice-sparse.git
cd DeiT_bit-slice-sparse
conda create -n deit-bitslice python=3.10 pip -y
conda activate deit-bitslice
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.2 torchvision==0.17.2
pip install -r requirements.txt
export IMAGENET_ROOT=/path/to/ILSVRC2012
```

`prepare_dense_checkpoint.py` 使用 timm 下载官方 DeiT-Base ImageNet-1k 权重并保存为 state dict。若要用已有权重，可跳过该步，并在后续命令传入 `--source-checkpoint`。

## 从头复现

完整流水线会生成 W8A8 校准 artifact、构造固定 75% mask、训练 35 轮并做完整 50k W8A8 验证。默认启动 4 个进程，可用 `NPROC_PER_NODE` 改写：

```bash
IMAGENET_ROOT=/path/to/ILSVRC2012 NPROC_PER_NODE=4 bash scripts/reproduce.sh
```

中间步骤和参数也可以单独运行。先生成 PTQ artifact 和 mask 后，运行 500-step pilot，再从它恢复完整 35 轮 schedule；这样不依赖仓库外的 pilot checkpoint：

```bash
python prepare_dense_checkpoint.py

python run_deit_base_ptq4vit_w8a8.py \
  --checkpoint checkpoints/deit_base_patch16_224.fb_in1k_fp32_state_dict.pth \
  --data "$IMAGENET_ROOT" \
  --output results/deit_base_ptq4vit_w8a8_q8k4p8v4_patchfp32 \
  --calib-size 32 --calib-batch-size 4 --max-eval-images 128

python train_deit_dense_smag_bitslice_qat.py \
  --source-checkpoint checkpoints/deit_base_patch16_224.fb_in1k_fp32_state_dict.pth \
  --ptq-artifact results/deit_base_ptq4vit_w8a8_q8k4p8v4_patchfp32/deit_base_ptq4vit_transformer_w8a8_q8k4p8v4_patchfp32.pt \
  --data "$IMAGENET_ROOT" --output-dir results/qat_75_globalcoverage_35ep \
  --mask-artifact results/mask_75_globalcoverage/mask_init.pt \
  --target-rate 75 --mask-selection global-coverage --prepare-only

torchrun --standalone --nproc_per_node=4 train_deit_dense_smag_bitslice_qat.py \
  --source-checkpoint checkpoints/deit_base_patch16_224.fb_in1k_fp32_state_dict.pth \
  --ptq-artifact results/deit_base_ptq4vit_w8a8_q8k4p8v4_patchfp32/deit_base_ptq4vit_transformer_w8a8_q8k4p8v4_patchfp32.pt \
  --data "$IMAGENET_ROOT" --output-dir results/qat_75_globalcoverage_pilot_500step \
  --mask-artifact results/mask_75_globalcoverage/mask_init.pt \
  --target-rate 75 --mask-selection global-coverage \
  --epochs 35 --batch-size 256 --eval-every 1 --checkpoint-every 1 \
  --lr 1e-4 --weight-decay 0.05 --warmup-epochs 5 --seed 17 \
  --max-train-steps 500

torchrun --standalone --nproc_per_node=4 train_deit_dense_smag_bitslice_qat.py \
  --source-checkpoint checkpoints/deit_base_patch16_224.fb_in1k_fp32_state_dict.pth \
  --ptq-artifact results/deit_base_ptq4vit_w8a8_q8k4p8v4_patchfp32/deit_base_ptq4vit_transformer_w8a8_q8k4p8v4_patchfp32.pt \
  --data "$IMAGENET_ROOT" --output-dir results/qat_75_globalcoverage_35ep \
  --mask-artifact results/mask_75_globalcoverage/mask_init.pt \
  --target-rate 75 --mask-selection global-coverage \
  --epochs 35 --batch-size 256 --eval-every 1 --checkpoint-every 1 \
  --lr 1e-4 --weight-decay 0.05 --warmup-epochs 5 --seed 17 \
  --resume-from results/qat_75_globalcoverage_pilot_500step/last.pt \
  --resume-best-top1 67.950

python eval_deit_bitslice_w8a8.py \
  --checkpoint results/qat_75_globalcoverage_35ep/last.pt \
  --mask-artifact results/mask_75_globalcoverage/mask_init.pt \
  --data "$IMAGENET_ROOT" --output-dir results/deit_base_bitslice_w8a8_eval \
  --max-eval-images 0
```

校准、训练、评测会在 `results/` 下生成大型 artifacts 和 checkpoint；它们不会提交到 Git。ImageNet 数据和预训练权重也不会放进仓库。

## 复现边界

训练脚本支持从头运行和 `--resume-from` 续训。原实验从 500-step pilot checkpoint 续跑，并在中断后恢复 optimizer/scaler；`scripts/reproduce.sh` 保留这段启动流程。不同 CUDA、数据加载和分布式调度下指标可能略有差异。原始 dense state-dict 文件 SHA256：`27d67f16d9928ff608a84ad7d2f60d680afd19e63ce4d63cbafdc962b030ff99`。`prepare_dense_checkpoint.py` 会从同一 timm pretrained tag 重新序列化 state dict，因此文件字节哈希可能不同；后续 PTQ 校准和 mask 生成会共享该文件并校验来源一致。
