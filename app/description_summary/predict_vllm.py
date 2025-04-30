from typing import Optional
from pathlib import Path
from huggingface_hub import snapshot_download
from config import config
import logging
import boto3
import os
import json
import io
import tarfile
from datasets import Dataset
from transformers import AutoTokenizer

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

from vllm.config import KVTransferConfig
import torch, gc

gc.collect()
torch.cuda.empty_cache()


class Predictor:
    def __init__(self):
        self._loras = {}
        self._token_max_len = 2048

        for adapter in config.sagemaker.adapters:
            self.lora_download_s3(
                lora_adapter=adapter.adapter_name,
                s3_key=os.path.join(
                    config.sagemaker.lora_prefix,
                    adapter.adapter_path,
                )
            )
        logging.info(f"Uploaded loras {self._loras}")

        self._engine_config = EngineArgs(
            model="meta-llama/Llama-3.1-8B-Instruct",
            max_model_len=self._token_max_len,
            max_num_seqs=1,
            max_loras=3,
            enable_lora=True,
            max_lora_rank=16,
            max_cpu_loras=3,
            gpu_memory_utilization=0.85,
            max_num_batched_tokens=None,
            enable_chunked_prefill=False,
        )
        self._sampling_params = SamplingParams(
            logprobs=1,
            prompt_logprobs=1,
            temperature=0.2,
            top_k=50,
            top_p=0.95,
            max_tokens=256,
        )
        self._engine = LLMEngine.from_engine_args(self._engine_config)

        logging.info("Successfully uploaded model from models/meta-llama-Llama-3.1-8B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "models/meta-llama-Llama-3.1-8B-Instruct", trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.tokenizer.padding_side = "left"
        self.prompt = '''Extract and summarize key information about the company. 
        Use only text provided. Dont start with: Based on the provided text, here's a summary, etc. 
        Description should be short and informative, up to 5 sentences without markup. 
        If it is impossible to extract relevant information return None. Text:'''

    def unzip_tar(self, fname: str):
        if fname.endswith("tar.gz"):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall(fname.split(".tar")[0])
            tar.close()
        os.remove(fname)

    def lora_download_s3(self, lora_adapter: str, s3_key: str) -> None:
        local_path = f"./tmp/{lora_adapter}"
        if not os.path.exists(local_path):
            os.makedirs(local_path, exist_ok=True)
            logging.info(f"Created new dir {local_path}")
        if any((True for _ in os.scandir(local_path))):
            logging.info(f"{local_path} is not empty. Assigning path to lora {lora_adapter}")
            self._loras[lora_adapter] = local_path
            return
        s3_client = boto3.client("s3")
        s3_client.download_file(
            Bucket=config.sagemaker.bucket,
            Key=s3_key,
            Filename=f"{local_path}.tar.gz",
        )
        self.unzip_tar(f"{local_path}.tar.gz")
        self._loras[lora_adapter] = local_path

    def token_len_check(self, prompt: str) -> str:
        tokens = self.tokenizer(
            prompt,
            max_length=self._token_max_len - self._sampling_params.max_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False
        )
        logging.info(f"Token length: {tokens['input_ids'].size(1)}")
        return self.tokenizer.decode(tokens.input_ids[0], skip_special_tokens=False)

    def generate_one(self, descriptions: str, lora: Optional[str] = None) -> str:
        logging.info(f"Sending description {descriptions} to model")
        request_id = 0
        if lora:
            lora_path = self._loras.get(lora)
            if not lora_path:
                raise ValueError(f"LoRA path not found for: {lora}")
            lora_request=LoRARequest(lora_name=lora, lora_int_id=1, lora_local_path=lora_path)
            logging.info(f"Using LoRA {lora} at path {lora_path}")
            base_prompt = [{"role": "user" , "content": descriptions}]
        else:
            lora_request=None
            logging.warning("No LoRARequest provided — token_lora_mapping will be minimal.")
            base_prompt = [{"role": "user", "content": f"{self.prompt} {descriptions}"}]
        
        formatted_prompt_str = self.tokenizer.apply_chat_template(base_prompt, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer(formatted_prompt_str, return_tensors="pt")
        logging.info(f"Final prompt length: {tokens['input_ids'].shape[1]}")
        # formatted_prompt_str = self.token_len_check(formatted_prompt_str)
        logging.info(f"Adding request with id {request_id} for formatted_prompt_str {formatted_prompt_str}")
        self._engine.add_request(
            request_id=str(request_id),
            prompt=formatted_prompt_str,
            params=self._sampling_params,
            lora_request=lora_request,
        )
        request_id += 1
        logging.info("Starting processing requests")
        while self._engine.has_unfinished_requests():
            logging.info(f"Engine has unfinished requests: {self._engine.has_unfinished_requests()}")
            request_outputs: list[RequestOutput] = self._engine.step()
            logging.info(f"Generated {len(request_outputs)} request_outputs")
            for request_output in request_outputs:
                if request_output.finished:
                    print(request_output.outputs[0].text)
                    print("-" * 50)
                    return request_output.outputs[0].text

def download_dataset_s3(s3_data_path: str) -> Dataset:
    s3_resource = boto3.Session().resource('s3').Bucket(config.sagemaker.bucket)
    test_response = s3_resource.Object(f"{os.path.join(s3_data_path, 'test.json')}").get()
    test_json = json.load(io.BytesIO(test_response["Body"].read()))
    return Dataset.from_list(test_json)

def download_dataset_disk(data_path: str) -> Dataset:
    logging.info(f"Downloading test.json from {data_path}")
    if os.path.exists(data_path):
        return Dataset.from_json(data_path)
    logging.error(f"Datapath {data_path + '/test.json'} doesn't exist")


def run_inference_on_json(data_path, result_path):
    gc.collect()
    torch.cuda.empty_cache()
    data_set = download_dataset_s3(data_path)
    logging.info(f"Successfully downloaded data from {data_path}")
    model = Predictor()

    generated_summary = []
    for row in data_set:
        descriptions = row["company_descriptions"]
        logging.info(f"Sending description {descriptions} to model")
        summary = model.generate_one(
            descriptions=descriptions, 
            lora="summary_adapter_2",
            )
        generated_summary.append(summary)
    df = data_set.to_pandas()
    df["generated_summary"] = generated_summary
    df.to_json(result_path, orient="records", lines=False, index=False)
