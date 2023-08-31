#!/bin/bash

source /workspace/storage/coder/WizardLM/WizardCoder/src/env.sh

mkdir -vp ${output_path}/tasks
echo 'Output path: '$output_path
echo 'Model to eval: '$model

# 164 problems, 21 per GPU if GPU=8
task_count=164
#gpu_num=8
gpu_num=1

step=$(awk "BEGIN {print int((${task_count}+${gpu_num}-1)/${gpu_num})}")
echo "step=${step}"

index=0
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * step))
  end_index=$(((i + 1) * step))
  echo "i=${i}, start_index=${start_index}, end_index=${end_index}"

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python humaneval_gen.py --model ${model} --prompt_template ${prompt_template} \
      --start_index ${start_index} --end_index ${end_index} --temperature ${temp} \
      --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} --max_len ${max_len} --output_path ${output_path}/tasks --greedy_decode
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done

