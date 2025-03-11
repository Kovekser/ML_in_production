from functools import partial
from app.labeling.argilla_description_summarization import ArgillaSummarizationClient
from app.config import config
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from os import path
from typing import Optional


class DataLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.client = ArgillaSummarizationClient(
            api_url=config.ARGILLA.HOST,
            api_key=config.ARGILLA.API_KEY,
            workspace=config.ARGILLA.WORKSPACE,
        )
        self.datasets = None

    def create_fomatted_dataset(self, dataset: Dataset):
        descriptions = [
            field["company_descriptions"].replace("KBAPI: ", "").replace(" Homepage text: ", "") for field in dataset["fields"]
        ]
        answers = [suggestion["summary"]["value"] for suggestion in dataset["suggestions"]]
        return Dataset.from_dict({"company_descriptions": descriptions, "summaries": answers})

    def download_labelled_data(self, path_to_save: Path, random_state: int = 15, subsample: Optional[float] = None) -> DatasetDict:
        if not path_to_save.exists():
            path_to_save.mkdir(parents=True, exist_ok=True)
        dataset_path = path_to_save.joinpath(f"{self.dataset_name}.json")
        self.client.download_records_dataset(self.dataset_name, dataset_path)
        print(f"Dataset was downloaded to: {path_to_save}")
        dataset = Dataset.from_json(dataset_path.as_posix())
        print(f"Dataset size: {len(dataset)}")
        if subsample:
            dataset = dataset.shuffle(seed=random_state).select(range(int(len(dataset) * subsample)))
            print(f"Subsampled dataset size: {len(dataset)}")
        new_dataset = self.create_fomatted_dataset(dataset)
        self.datasets = new_dataset.train_test_split(test_size=0.05, seed=random_state)
        return self.datasets

    def load_to_json(self, path_to_save: Path):
        if not path_to_save.exists():
            print(f"Path {path_to_save} does not exist")
            path_to_save.mkdir(parents=True, exist_ok=True)
        self.datasets["train"].to_json(path_to_save.joinpath("train.json"))
        self.datasets["test"].to_json(path_to_save.joinpath("test.json"))
        print(f"Data saved to: {path_to_save}")


class SummaryDatasetProcessor:
    def __init__(self, model_id: str, path_to_data: Path):
        self.model_id = model_id
        self._path_to_data = path_to_data
        self._tokenizer_id = model_id

    def construct_message_row(self, row):
        messages = []
        user_message = {"role": "user", "content": row["company_descriptions"]}
        messages.append(user_message)
        assistant_message = {"role": "assistant", "content": row["summaries"]}
        messages.append(assistant_message)
        print(f"Messages: {messages}")  # Debugging
        return {"messages": messages}

    def format_data_chat_template(self, row, tokenizer):
        chat_template = tokenizer.apply_chat_template(
                row["messages"], add_generation_prompt=False, tokenize=False
            )
        return {"text": chat_template}

    def process_dataset(self, train_file_name:str, test_file_name: str) -> DatasetDict:
        dataset = DatasetDict(
            {
                "train": Dataset.from_json(f"{path.join(self._path_to_data, train_file_name)}"),
                "test": Dataset.from_json(f"{path.join(self._path_to_data, test_file_name)}"),
            }
        )

        tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_id)
        tokenizer.padding_side = "left"  # For Llama 3 model
        tokenizer.pad_token = tokenizer.eos_token

        dataset_chatml = dataset.map(self.construct_message_row)
        dataset_chatml = dataset_chatml.map(
            partial(self.format_data_chat_template, tokenizer=tokenizer)
        )
        return dataset_chatml


def process_dataset_summaries(model_id: str, train_file: str, test_file: str, subsample: float, new: bool) -> DatasetDict:
    train_dir, train_file = path.split(train_file)
    test_head, test_file = path.split(test_file)
    path_to_save = Path(train_dir)
    if new:
        dataset_name = "descriptions_summaries"
        data_loader = DataLoader(dataset_name)
        data_loader.download_labelled_data(path_to_save, subsample=subsample)
        data_loader.load_to_json(path_to_save)

    processor = SummaryDatasetProcessor(model_id, path_to_save)
    return processor.process_dataset(train_file_name=train_file, test_file_name=test_file)
