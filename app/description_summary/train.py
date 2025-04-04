from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
import logging
from pathlib import Path
import argparse
import wandb
from trl import SFTTrainer, SFTConfig

from .conf.arguments import DataTrainingArguments, ModelArguments
from .data import format_dataset_summaries
from .utils import setup_logger
import torch
from peft import LoraConfig, TaskType
import os
from huggingface_hub import login

logger = logging.getLogger(__name__)
hf_token = os.getenv("HUGGINGFACE__TOKEN")
if hf_token:
    login(token=hf_token)
else:
    raise ValueError("HF_TOKEN not found in environment variables")

wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)


def get_config(config_path: Path):
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_json_file(config_path)
    return model_args, data_args, training_args


def get_model(model_id: str):
    device_map = None
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():  # Check BF16 support
            compute_dtype = torch.bfloat16  # Use BF16 (preferred on Ampere)
        else:
            compute_dtype = torch.float16
        attn_implementation = "flash_attention_2"
        device = "cuda"
        device_map = "auto"
        # If bfloat16 is not supported, 'compute_dtype' is set to 'torch.float16' and 'attn_implementation' is set to 'sdpa'.
    elif torch.backends.mps.is_available():
        # MPS doesn't currently support bfloat16, so use float16
        torch.mps.empty_cache()
        compute_dtype = torch.float16
        attn_implementation = "sdpa"
        device = "mps"
        logging.info("Using MPS (Metal Performance Shaders) with float16.")
    else:
        compute_dtype = torch.float16
        attn_implementation = "sdpa"
        device = "cpu"
        # This line of code is used to print the value of 'attn_implementation', which indicates the chosen attention implementation.
    print(f"{attn_implementation}. Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "left"

    if not device_map:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
    device = torch.device(device)
    model = model.to(device)
    return tokenizer, model


def train(config_path: Path):
    setup_logger(logger)

    model_args, data_args, training_args = get_config(config_path=config_path)

    logger.info(f"model_args = {model_args}")
    logger.info(f"data_args = {data_args}")
    logger.info(f"training_args = {training_args}")

    target_modules = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ]

    set_seed(training_args.seed)

    dataset_chatml = format_dataset_summaries(
        model_id=model_args.model_id,
        train_file=data_args.train_file,
        test_file=data_args.test_file,
        path_to_data=data_args.path_to_data,
    )
    logger.info(dataset_chatml["train"][0])

    tokenizer, model = get_model(model_id=model_args.model_id)
    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    sft_config = SFTConfig(**training_args.to_dict())
    sft_config.dataset_text_field = "text"
    sft_config.max_seq_length = 1024

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_chatml["train"],
        eval_dataset=dataset_chatml["test"],
        peft_config=peft_config,
        args=sft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model()
    trainer.create_model_card()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)  # Expect config path as a string
    args = parser.parse_args()

    # Convert to a Path object
    config_path = Path(args.config_path)
    train(
        config_path=Path(config_path),
   )
