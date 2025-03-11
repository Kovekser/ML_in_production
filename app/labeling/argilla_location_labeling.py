import argilla as rg
from csv import DictReader
from app.clients import LabelResponseOne
from app.utils import parse_command_line_argilla, ArgillaLabelParams
from typing import List, Optional
from app.labeling.argilla_base_labeling import ArgillaBaseLabelClient

args: ArgillaLabelParams = parse_command_line_argilla()


class ArgillaFundingEventLabelClient(ArgillaBaseLabelClient):
    labels = ["company", "location", "description"]

    def __init__(self, api_url: str, api_key: str, workspace: str):
        super().__init__(api_url, api_key, workspace)

    @property
    def dataset_settings(self):
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
                    labels=self.labels,
                    title="Label companies, their descriptions and locations in the text",
                    required=True,
                    description="Highlight the names of companies, their descriptions and locations in the text.",
                )
            ]
        )

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


if __name__ == "__main__":
    if not args.dataset_name:
        raise ValueError("Dataset name is required")

    client = ArgillaFundingEventLabelClient(
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
        client.download_records_json(args.dataset_name, args.download_file_json)
    elif args.download_file_dataset:
        client.download_records_dataset(args.dataset_name)
