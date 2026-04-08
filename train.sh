#!/bin/bash

export MODEL="RETFound_mae"
export MODEL_ARCH="retfound_mae"
export FINETUNE="./checkpoints/RETFound_mae_natureCFP.pth"
export DATASET="RFMiD_binary"
export NUM_CLASS=2
export DATA_PATH="./rfmid_binary"
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
  --batch_size 24 \
  --world_size 1 \
  --epochs 50 \
  --nb_classes "${NUM_CLASS}" \
  --data_path "${DATA_PATH}" \
  --input_size 224 \
  --task "${TASK}" \
  --adaptation "finetune"