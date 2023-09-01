#!/bin/bash

source /workspace/storage/coder/WizardLM/WizardCoder/src/env.sh

echo 'Output path: '$output_path

python process_humaneval.py --path ${output_path}/tasks --out_path ${output_path}/gen.jsonl --add_prompt


