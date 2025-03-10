from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
import logging
from pathlib import Path

from dataclasses import dataclass
from trl import SFTTrainer, SFTConfig
from description_summary.fine_tuner.data import process_dataset_summaries
from description_summary.utils import setup_logger
import torch
from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)
torch.mps.empty_cache()


@dataclass
class DataTrainingArguments:
    train_file: str
    test_file: str


@dataclass
class ModelArguments:
    model_id: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float


def get_config(config_path: Path):
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_json_file(config_path)
    return model_args, data_args, training_args


def get_model(model_id: str, device_map):
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
        # This line of code is used to print the value of 'attn_implementation', which indicates the chosen attention implementation.
    print(f"{attn_implementation}. Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        # device_map=device_map,
        attn_implementation=attn_implementation,
    )
    device = torch.device(device)
    model = model.to(device)
    return tokenizer, model


def train(config_path: Path, subsample: float, new_data: bool):
    setup_logger(logger)

    model_args, data_args, training_args = get_config(config_path=config_path)

    logger.info(f"model_args = {model_args}")
    logger.info(f"data_args = {data_args}")
    logger.info(f"training_args = {training_args}")

    # device_map = {"": 0}
    device_map = None
    target_modules = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ]

    set_seed(training_args.seed)

    dataset_chatml = process_dataset_summaries(
        model_id=model_args.model_id,
        train_file=data_args.train_file,
        test_file=data_args.test_file,
        subsample=subsample,
        new=new_data,
    )
    logger.info(dataset_chatml["train"][0])

    tokenizer, model = get_model(model_id=model_args.model_id, device_map=device_map)
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
    )

    trainer.train()
    trainer.save_model()
    trainer.create_model_card()
