from dataclasses import dataclass


@dataclass
class DataTrainingArguments:
    train_file: str
    test_file: str
    path_to_data: str


@dataclass
class ModelArguments:
    model_id: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
