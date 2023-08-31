#!/bin/bash

source /workspace/storage/coder/WizardLM/WizardCoder/src/env.sh

mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model

gpu_num=2

start_index=0
end_index=164

torchrun --nproc_per_node=${gpu_num} humaneval_gen.py --model ${model} --prompt_template ${prompt_template} \
      --start_index ${start_index} --end_index ${end_index} --temperature ${temp} \
      --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} --max_len ${max_len} --output_path ${output_path} --greedy_decode