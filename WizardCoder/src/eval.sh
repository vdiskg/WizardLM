#!/bin/bash

source /workspace/storage/coder/WizardLM/WizardCoder/src/env.sh

mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model

# 164 problems, 21 per GPU if GPU=8
index=0
#gpu_num=8
gpu_num=1
for ((i = 0; i < $gpu_num; i++)); do
  #start_index=$((i * 21))
  #end_index=$(((i + 1) * 21))
  start_index=0
  end_index=164

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python humaneval_gen.py --model ${model} --prompt_template ${prompt_template} \
      --start_index ${start_index} --end_index ${end_index} --temperature ${temp} \
      --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} --max_len ${max_len} --output_path ${output_path} --greedy_decode
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done

