from huggingface_hub import snapshot_download
from config import config

model_id = "meta-llama/Llama-3.1-8B-Instruct"
model_id_pathsafe = model_id.replace("/","-")
local_model_path = f"./models/{model_id_pathsafe}"

snapshot_download(repo_id=model_id, use_auth_token=config.huggingface.token, local_dir=local_model_path, resume_download=True)
