import json
from os import path
from pathlib import Path
from typing import Optional

import boto3
from datasets import Dataset, DatasetDict

from config import config
from labeling.argilla_description_summarization import ArgillaSummarizationClient


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
            Key=path.join(f"{prefix}/datasets", 'train.json'),
            Body=json.dumps(self.datasets["train"].to_dict())
        )
        s3_resource.put_object(
            Key=path.join(f"{prefix}/datasets", 'test.json'),
            Body=json.dumps(self.datasets["test"].to_dict())
        )


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
