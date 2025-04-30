import json
import logging
from pathlib import Path

import evaluate
import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from description_summary.summary_evaluate import SummaryEvaluator
from description_summary.predict_vllm import download_dataset_s3

logger = logging.getLogger()


class Predictor:
    def __init__(self, model_load_path: str):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            attn_implementation = "flash_attention_2"
            device = "cuda"
            # If bfloat16 is not supported, 'compute_dtype' is set to 'torch.float16' and 'attn_implementation' is set to 'sdpa'.
        elif torch.backends.mps.is_available():
            # MPS doesn't currently support bfloat16, so use float16
            compute_dtype = torch.float16
            attn_implementation = "sdpa"
            device = "mps"
            logging.info("Using MPS (Metal Performance Shaders) with float16.")
        else:
            compute_dtype = torch.float16
            attn_implementation = "sdpa"
            device = "cpu"
        self.device = torch.device(device)

        # new_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        new_model = AutoPeftModelForCausalLM.from_pretrained(
            model_load_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=compute_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        self.merged_model = new_model.merge_and_unload()
        self.merged_model = new_model

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_load_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.tokenizer.padding_side = "left"

        self.pipe = pipeline("text-generation", model=self.merged_model, tokenizer=self.tokenizer)

    @torch.no_grad()
    def generate_v2(self, descriptions: str) -> str:
        messages = [{"role": "user", "content": descriptions}]
        chat_msg = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_msg, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.merged_model.generate(input_ids=inputs["input_ids"].to(self.device), max_new_tokens=256)
        summarized_text = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
        return summarized_text

    @torch.no_grad()
    def generate(self, descriptions: str) -> str:
        pipe = self.pipe

        messages = [{"role": "user", "content": descriptions}]
        prompt = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            num_beams=1,
            temperature=0.5,
            top_k=50,
            top_p=0.95,
            max_time=180,
        )
        summary = outputs[0]["generated_text"][len(prompt) :].replace("\n\n", " ").strip()
        return summary


def run_inference_on_json_tgi(data_path: str, model_load_path: str, result_path: str):
    ds = download_dataset_s3(data_path)
    df = ds.to_pandas()
    model = Predictor(model_load_path=model_load_path)

    generated_summary = []
    for idx in tqdm(range(len(df))):
        description = df.iloc[idx]["company_descriptions"]
        summary = model.generate(descriptions=description)
        generated_summary.append(summary)
    df["generated_summary"] = generated_summary
    df.to_csv(result_path, index=False)


def run_evaluate_on_csv(csv_path: Path):
    df = Dataset.from_csv(str(csv_path)).to_pandas()
    predictions = df["generated_summary"].values
    references = df["summaries"].values
    summary_evaluator = SummaryEvaluator()
    summary_evaluator.evaluate_model_generated_summary(predictions, references)
