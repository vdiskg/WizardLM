import sys
import os
import time
from datetime import datetime

import fire
import torch
import transformers
import json
import jsonlines

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def evaluate(
        batch_data,
        tokenizer,
        model,
        input=None,
        temperature=1,
        top_p=0.9,
        top_k=40,
        num_beams=1,
        max_new_tokens=2048,
        **kwargs,
):
    prompts = generate_prompt(batch_data, input)
    inputs = tokenizer(prompts, return_tensors="pt", max_length=256, truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    return output


def generate_prompt(instruction, input=None):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


def main(
    load_8bit: bool = False,
    base_model: str = "Model_Path",
    input_data_path = "Input.jsonl",
    output_data_path = "Output.jsonl",
):
    t0 = time.time()
    print(f'start: {datetime.now()}')
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )

    model.config.pad_token_id = tokenizer.pad_token_id

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    input_data = jsonlines.open(input_data_path, mode='r')
    output_data = jsonlines.open(output_data_path, mode='w')

    t1 = time.time()
    print(f'loaded: {datetime.now()}, cost: {"{:.6f}".format(t1 - t0)}s')
    results = []
    for num, line in enumerate(input_data):
        one_data = line
        id = one_data["idx"]
        instruction = one_data["Instruction"]
        t2 = time.time()
        print(f'idx: {id}, time: {datetime.now()}, instruction: {instruction}')
        _output = evaluate(instruction, tokenizer, model)
        final_output = _output[0].split("### Response:")[1].strip()
        t3 = time.time()
        print(f'idx: {id}, time: {datetime.now()}, cost: {"{:.6f}".format(t3 - t2)}s')
        new_data = {
            "id": id,
            "instruction": instruction,
            "wizardcoder": final_output
        }
        results.append(new_data)
        output_data.write(new_data)

    for result in results:
        id = result["id"]
        instruction = result["instruction"]
        output_data = result["wizardcoder"]
        print(f'idx: {id}, instruction:\n {instruction}\n\n output:\n {output_data}')


if __name__ == "__main__":
    fire.Fire(main)
