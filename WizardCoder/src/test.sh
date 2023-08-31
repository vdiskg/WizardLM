#!/bin/bash

source /workspace/storage/coder/WizardLM/WizardCoder/src/env.sh

echo 'Output path: '$output_path
evaluate_functional_correctness ${output_path}/${output_path}.jsonl

