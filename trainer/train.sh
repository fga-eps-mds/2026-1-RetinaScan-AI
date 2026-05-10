#!/bin/bash

export MODEL="RETFound_dinov2"
export MODEL_ARCH="retfound_dinov2"
export FINETUNE="./checkpoints/RETFound_dinov2_meh.pth"
export DATASET="ODIR_binary"
export NUM_CLASS=2
export DATA_PATH="./odir_binary"
export TASK="${MODEL_ARCH}_${DATASET}_finetune"

#⚠️ mude: só uma linha, sem quebras no meio
torchrun --nproc_per_node=1 \
  --master_port=48766 \
  main_finetune.py \
  --model "${MODEL}" \
  --model_arch "${MODEL_ARCH}" \
  --finetune "${FINETUNE}" \
  --savemodel \
  --global_pool \
  --batch_size 16 \
  --world_size 1 \
  --epochs 50 \
  --nb_classes "${NUM_CLASS}" \
  --data_path "${DATA_PATH}" \
  --input_size 224 \
  --task "${TASK}" \
  --adaptation "finetune"