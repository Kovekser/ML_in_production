from functools import partial
from app.labeling.argilla_description_summarization import ArgillaSummarizationClient
from app.config import config
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from os import path
from typing import Optional
import sagemaker
import boto3
import json
import io


class DataLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.client = ArgillaSummarizationClient(
            api_url=config.ARGILLA.HOST,
            api_key=config.ARGILLA.API_KEY,
            workspace=config.ARGILLA.WORKSPACE,
        )
        self.datasets = None

    def create_formatted_dataset(self, dataset: Dataset):
        descriptions = [
            row["company_descriptions"].replace("KBAPI: ", "").replace(" Homepage text: ", "") for row in dataset["fields"]
        ]
        answers = [suggestion["summary"]["value"] for suggestion in dataset["suggestions"]]
        return Dataset.from_dict({"company_descriptions": descriptions, "summaries": answers})

    def download_labelled_data(self, random_state: int = 15, subsample: Optional[float] = None) -> DatasetDict:
        dataset_data = self.client.download_records_dict(self.dataset_name)
        print(f"Dataset was downloaded to dict")
        dataset = Dataset.from_dict(dataset_data)
        print(f"Dataset size: {len(dataset)}")
        if subsample:
            dataset = dataset.shuffle(seed=random_state).select(range(int(len(dataset) * subsample)))
            print(f"Subsampled dataset size: {len(dataset)}")
        new_dataset = self.create_formatted_dataset(dataset)
        self.datasets = new_dataset.train_test_split(test_size=0.05, seed=random_state)
        return self.datasets

    def load_to_json_local(self, path_to_save: Path):
        self.datasets["train"].to_json(path_to_save.joinpath("train.json"))
        self.datasets["test"].to_json(path_to_save.joinpath("test.json"))
        print(f"Data saved to: {path_to_save}")

    def load_to_json_s3(self, bucket: str, prefix: str):
        s3_resource = boto3.Session().resource('s3').Bucket(bucket)
        s3_resource.put_object(
            Key=path.join(prefix, 'train.json'),
            Body=json.dumps(self.datasets["train"].to_dict())
        )
        s3_resource.put_object(
            Key=path.join(prefix, 'test.json'),
            Body=json.dumps(self.datasets["test"].to_dict())
        )


class SummaryDatasetProcessor:
    def __init__(self, model_id: str, path_to_data: str):
        self.model_id = model_id
        self._tokenizer_id = model_id
        self._s3_bucket = None
        self._path_to_data = None
        if path_to_data.startswith("s3"):
            self._s3_bucket = path_to_data.split("//")[-1].split("/")[0]
            self._path_to_data = path_to_data.split("//")[-1].split("/", 1)[1]
        else:
            self._path_to_data = path_to_data

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

    def process_dataset(self, train_file_name: str, test_file_name: str) -> DatasetDict:
        if self._s3_bucket:
            s3_resource = boto3.Session().resource('s3').Bucket(self._s3_bucket)
            train_response = s3_resource.Object(f"{path.join(self._path_to_data, train_file_name)}").get()
            test_response = s3_resource.Object(f"{path.join(self._path_to_data, test_file_name)}").get()
            train_json = json.load(io.BytesIO(train_response["Body"].read()))
            test_json = json.load(io.BytesIO(test_response["Body"].read()))
            dataset = DatasetDict(
                {
                    "train": Dataset.from_dict(train_json),
                    "test": Dataset.from_dict(test_json),
                }
            )
        else:
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
        print(dataset_chatml["train"][0])  # Debugging
        print(dataset_chatml["test"][0])  # Debugging
        return dataset_chatml


def upload_dataset(dataset_name: str, subsample: float, s3: str, path_to_save: Path):
    if path_to_save:
        path_to_save = Path(path_to_save)
        if not path_to_save.exists():
            print(f"Path {path_to_save} does not exist")
            path_to_save.mkdir(parents=True, exist_ok=True)
    data_loader = DataLoader(dataset_name)
    data_loader.download_labelled_data(subsample=subsample)
    if s3:
        data_loader.load_to_json_s3(s3, dataset_name)
    else:
        data_loader.load_to_json_local(Path.cwd())


def process_dataset_summaries(model_id: str, train_file: str, test_file: str, path_to_data: str) -> DatasetDict:
    processor = SummaryDatasetProcessor(model_id, path_to_data)
    return processor.process_dataset(train_file_name=train_file, test_file_name=test_file)
