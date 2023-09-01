import os

from huggingface_hub import snapshot_download

# 模型在huggingface上的名称
repo_id = "bigcode/starcoder"
# 本地模型存储的地址
local_dir = "/workspace/temp/model/starcoder/"
local_dir_use_symlinks = False  # 本地模型使用文件保存，而非blob形式保存
# 在hugging face上生成的 access token
token = os.environ.get("HF_TOKEN", None)
# 如果需要代理的话
proxies = {
    'http': None,
    'https': None,
}
snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=local_dir_use_symlinks, token=token,
                  proxies=proxies)
