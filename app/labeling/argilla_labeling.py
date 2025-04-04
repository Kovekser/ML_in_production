import argilla as rg
from csv import DictReader
from app.utils import ArgillaLabelParams, parse_command_line_argilla
from app.clients import OpenAIClient, LabelResponseOne
from typing import List, Optional
import re

args: ArgillaLabelParams = parse_command_line_argilla()


class ArgillaLabelClient:
    def __init__(self, api_url: str, api_key: str, workspace: str):

        self._labels = ["company", "location", "description"]
        self.client = rg.Argilla(
            api_url=api_url,
            api_key=api_key,
        )

        self._workspace_name = workspace
        self.workspace = self._get_or_create_workspace()
        self.openai_client = OpenAIClient()

    def _get_or_create_workspace(self):
        my_workspace = self.client.workspaces(self._workspace_name)

        if not my_workspace:
            workspace = rg.Workspace(name="admin")
            my_workspace = workspace.create()
            my_workspace.add_user("argilla")
        return my_workspace

    @property
    def summary_dataset_settings(self):
        return rg.Settings(
            guidelines="Summarize the text provided.",
            fields=[
                rg.TextField(
                    name="company_descriptions",
                    title="Text of entites descriptions",
                    use_markdown=False,
                )
            ],
            questions=[
                rg.TextQuestion(
                    name="summary",
                    title="Summarize the text",
                    required=True,
                    description="Summarize multiple descriptions of the company's main economic activity",
                )
            ]
        )

    @property
    def location_dataset_settings(self):
        return rg.Settings(
            guidelines="Label companies and locations in text if available.",
            fields=[
                rg.TextField(
                    name="funding_event",
                    title="Text of the funding event",
                    use_markdown=False,
                )
            ],
            questions=[
                rg.SpanQuestion(
                    name="entity",
                    field="funding_event",
                    labels=self._labels,
                    title="Label companies, their descriptions and locations in the text",
                    required=True,
                    description="Highlight the names of companies, their descriptions and locations in the text.",
                )
            ]
        )

    def get_or_create_dataset(self, dataset_name: str):
        dataset = self.get_dataset(dataset_name)
        if not dataset:
            dataset = self.create_dataset(dataset_name)
        return dataset

    def create_dataset(self, dataset_name: str, ):
        dataset = rg.Dataset(
            name=dataset_name, settings=self.location_dataset_settings, client=self.client, workspace=self.workspace,
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
        records = []
        with open(filename) as f:
            reader = DictReader(f)
            for row in reader:
                x = rg.Record(
                    fields={
                        "funding_event": row["event_text"],
                    },
                )
                records.append(x)
        return records

    @staticmethod
    def create_suggestion(openai_suggestions: List[LabelResponseOne]) -> rg.Suggestion:
        print(f"Creating suggestions {openai_suggestions}")
        suggestion_vaues = []
        for suggestion in openai_suggestions:
            suggestion_vaues.append(suggestion.model_dump(mode="json"))
        return rg.Suggestion(
                    question_name="entity",
                    value=suggestion_vaues,
                    agent="openai",
                )

    def create_records_with_suggestions(self, filename: str, dataset_name, num_of_records: Optional[int] = None):
        records = []
        with open(filename) as f:
            reader = DictReader(f)
            for row in reader:
                event_text = self.pre_process_text(row["event_text"])
                suggestions = self.openai_client.generate_labels(event_text)
                r = rg.Record(
                    fields={
                        "funding_event": event_text,
                    },
                    suggestions=[self.create_suggestion(suggestions)],
                )
                try:
                    self.upload_records(dataset_name, [r])
                except Exception as e:
                    print(f"Error uploading record: {r}")
                    print(f"Error: {e}")
                    continue
                print(f"Processed {len(records)} records")
                if num_of_records and len(records) == num_of_records:
                    break
        return records

    def download_records(self, dataset: str, filename: str):
        dataset = self.get_or_create_dataset(dataset)
        dataset.records.to_json(filename)
        print(f"Records were downloaded to the file {filename}")

    @staticmethod
    def pre_process_text(text: str) -> str:
        text = text.encode("ascii", errors="ignore").decode()
        text = re.sub(' +', ' ', text)
        return " ".join(text.split())


if __name__ == "__main__":
    if not args.dataset_name:
        raise ValueError("Dataset name is required")

    client = ArgillaLabelClient(
        api_url="http://localhost:6900",
        api_key="argilla.apikey",
        workspace="admin",
    )
    print(f"Client was created")

    if args.upload_file and not args.suggestion:
        records = client.create_records(args.upload_file)
        client.upload_records(args.dataset_name, records)
    elif args.upload_file and args.suggestion:
        records = client.create_records_with_suggestions(args.upload_file, args.dataset_name)
    elif args.download_file_json:
        client.download_records(args.dataset_name, args.download_file_json)
