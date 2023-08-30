modle_dir="/workspace/temp/model"
#model_name="WizardCoder-15B-V1.0"
#model_name="octocoder"
#model_name="CodeLlama-34b-Instruct-hf"
#model_name="CodeLlama-13b-Instruct-hf"
model_name="CodeLlama-7b-Instruct-hf"

#prompt_template=""
prompt_template="CodeLlama"

model="${modle_dir}/${model_name}"
temp=0.0
max_len=2048
pred_num=1
num_seqs_per_iter=1

output_path=/workspace/storage/output/T${temp}_N${pred_num}_${model_name}

