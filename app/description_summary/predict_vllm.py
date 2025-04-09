from typing import Optional
from pathlib import Path
from huggingface_hub import snapshot_download
from config import config
import boto3
import os
import json
import io
import tarfile
from datasets import Dataset

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


class Predictor:
    def __init__(self):
        self._loras = {}
        for adapter in config.sagemaker.adapters:
            self.lora_download(
                lora_adapter=adapter.adapter_name,
                s3_key=os.path.join(
                    config.sagemaker.lora_prefix,
                    adapter.adapter_path,
                )
            )
        self._engine_config = EngineArgs(
            model=config.huggingface.base_model_id,
            max_loras=5,
            enable_lora=True,
            max_lora_rank=16,
            max_cpu_loras=6,
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

    def unzip_tar(self, fname: str):
        if fname.endswith("tar.gz"):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall(fname.split(".tar")[0])
            tar.close()
        os.remove(fname)

    def lora_download(self, lora_adapter: str, s3_key: str) -> str:
        local_path = f"./tmp/{lora_adapter}"
        if not os.path.exists(local_path):
            os.makedirs(local_path, exist_ok=True)
        s3_client = boto3.client("s3")
        s3_client.download_file(
            Bucket = config.sagemaker.bucket,
            Key=s3_key,
            Filename=f"{local_path}.tar.gz",
        )
        self.unzip_tar(f"{local_path}.tar.gz")
        self._loras[lora_adapter] = local_path

    def generate_one(self, descriptions: str, lora: Optional[str]) -> str:
        request_id = 0
        prompt = [{"role": "user", "content": descriptions}]
        while self._engine.has_unfinished_requests() or descriptions:
            self._engine.add_request(
                str(request_id),
                prompt,
                self._sampling_params,
                lora_request=LoRARequest(lora, self._loras[lora])
            )
        request_outputs: list[RequestOutput] = self._engine.step()
        if len(request_outputs) == 1 and request_outputs[0].finished:
            return request_outputs[0]


def download_dataset(s3_data_path: str) -> Dataset:
    s3_resource = boto3.Session().resource('s3').Bucket(config.sagemaker.bucket)
    test_response = s3_resource.Object(f"{os.path.join(s3_data_path, "test.json")}").get()
    test_json = json.load(io.BytesIO(test_response["Body"].read()))
    return Dataset.from_dict(test_json)


def run_inference_on_json(data_path, result_path):
    ds = download_dataset(data_path)
    df = ds.to_pandas()
    model = Predictor()

    generated_summary = []
    for idx in range(len(df)):
        descriptions = df.iloc[idx]["description"]
        summary = model.generate_one(descriptions=descriptions, lora="summary_adapter")
        generated_summary.append(summary)
    df["generated_summary"] = generated_summary
    df.to_csv(result_path, index=False)
