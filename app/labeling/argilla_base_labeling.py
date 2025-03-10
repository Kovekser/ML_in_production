import argilla as rg

from app.clients import OpenAIClient, get_ai_client
from typing import List, Optional, Any
import re
from argilla.records._io import HFDataset
from pathlib import Path


class ArgillaBaseLabelClient:
    labels: List[str] = []

    def __init__(self, api_url: str, api_key: str, workspace: str):

        self.client = rg.Argilla(
            api_url=api_url,
            api_key=api_key,
        )

        self._workspace_name = workspace
        self.workspace = self._get_or_create_workspace()
        self.openai_client = get_ai_client()

    def _get_or_create_workspace(self):
        my_workspace = self.client.workspaces(self._workspace_name)

        if not my_workspace:
            workspace = rg.Workspace(name="admin")
            my_workspace = workspace.create()
            my_workspace.add_user("argilla")
        return my_workspace

    @property
    def dataset_settings(self):
        raise NotImplementedError("You should implement this method in your class")

    def get_or_create_dataset(self, dataset_name: str):
        dataset = self.get_dataset(dataset_name)
        if not dataset:
            dataset = self.create_dataset(dataset_name)
        return dataset

    def create_dataset(self, dataset_name: str, ):
        dataset = rg.Dataset(
            name=dataset_name, settings=self.dataset_settings, client=self.client, workspace=self.workspace,
        )
        dataset.create()
        return dataset

    def get_dataset(self, dataset_name: str):
        return self.client.datasets(dataset_name)

    def upload_records(self, dataset_name, records):
        dataset = self.get_or_create_dataset(dataset_name)
        dataset.records.log(records)
        print(f"Records were uploaded to the dataset {dataset_name}")

    def create_records(self, filename: str):
        raise NotImplementedError("You should implement this method in your class")

    @staticmethod
    def create_suggestion(openai_suggestions: Any) -> rg.Suggestion:
        raise NotImplementedError("You should implement this method in your class")

    def create_records_with_suggestions(self, filename: str, dataset_name, num_of_records: Optional[int] = None):
        raise NotImplementedError("You should implement this method in your class")

    def download_records_json(self, dataset: str, filename: str):
        dataset = self.get_or_create_dataset(dataset)
        dataset.records.to_json(filename)
        print(f"Records were downloaded to the file {filename}")

    def download_records_dataset(self, dataset_name: str, path_: Path) -> None:
        dataset = self.get_or_create_dataset(dataset_name)
        print(f"Records were downloaded to the dataset")
        dataset.records.to_json(path_)
        return


    @staticmethod
    def pre_process_text(text: str) -> str:
        text = text.encode("ascii", errors="ignore").decode()
        text = re.sub(' +', ' ', text)
        return " ".join(text.split())
