#!/bin/bash

source /workspace/storage/coder/WizardLM/WizardCoder/src/env.sh

echo 'Output path: '$output_path

python process_humaneval.py --path ${output_path} --out_path ${output_path}.jsonl --add_prompt


