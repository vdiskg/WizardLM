import argparse
import pprint
import sys
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from human_eval.data import write_jsonl, read_problems, stream_jsonl

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def generate_prompt(input, prompt_template):
    if prompt_template is None or prompt_template == '' or prompt_template == 'WizardCoder':
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input}

### Response:"""

    elif prompt_template == 'QA':
        return f"""Q: {input}
A: """
    else:
        raise Exception(f'Unknown prompt template: {prompt_template}')


def get_model(
    load_8bit: bool = False,
    base_model: str = "bigcode/starcoder",
):
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
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=512, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    parser.add_argument('--greedy_decode', action='store_true', help='')
    parser.add_argument('--overwrite', action='store_true', help='')
    parser.add_argument('--prompt_template', type=str, default='WizardCoder', help="")

    args = parser.parse_args()

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = read_problems()

    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    tokenizer, model = get_model(base_model=args.model)
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False if args.greedy_decode else True,
        temperature=args.temperature,
        max_length=args.max_len,
        num_return_sequences=args.num_seqs_per_iter,
        eos_token_id=tokenizer.eos_token_id,
        top_p=0.95
    )

    print(f"Loaded {args.model}.")
    prompt_template = generate_prompt(args.prompt_template)
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_prompt(prompt, prompt_template)]

        ids_batch = [task_ids[i]]

        completion_seqs = []

        encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)

        if args.decoding_style == 'sampling':
            loops = int(args.N / args.num_seqs_per_iter)
        else:
            loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                gen_tokens = model.generate(
                    **encoding,
                    generation_config=generation_config
                )

            if gen_tokens is not None:
                gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            else:
                gen_seqs = None

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {'task_id': task_id,
                         'completion': completion_seq,
                         'all_code': all_code,
                         }
                    )

        print("Saving results to {}".format(output_file))
        write_jsonl(output_file, completion_seqs)


if __name__ == '__main__':
    main()