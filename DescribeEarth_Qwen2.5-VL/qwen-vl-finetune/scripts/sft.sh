#!/bin/bash

# export PATH=/usr/local/cuda-11.8/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

GPUS="1"  

NPROC_PER_NODE=$(echo $GPUS | awk -F',' '{print NF}')
if [ -z "$NPROC_PER_NODE" ] || [ "$NPROC_PER_NODE" -eq 0 ]; then
  echo "[ERROR] Invalid GPU selection: $GPUS"
  exit 1
fi

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=../../weights/Qwen2.5-VL-3B-RC  # Using HuggingFace model ID

# Training hyperparameters
lr=2e-7
batch_size=2
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=dota_train,dior_train

# Output configuration
run_name="describe-earth_qwen2vl"
output_dir=../../outputs

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --output_dir ${output_dir} \
    --bf16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name}"

echo "[INFO] Using GPUs: $GPUS (Total processes per node: $NPROC_PER_NODE)"
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=${NPROC_PER_NODE} \
                                    --master_addr=${MASTER_ADDR} \
                                    --master_port=${MASTER_PORT} \
                                    ${entry_file} ${args}

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}