import argilla as rg
from csv import DictReader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--upload", "-U", type=str)
parser.add_argument("--download", "-D", type=str)

args = parser.parse_args()
upload_file = args.upload
print(f"upload_file: {upload_file}")
download_file = args.download
print(f"download_file: {download_file}")


class ArgillaLabelClient:
    dataset_name = "funding_event_analysis"

    def __init__(self, api_url: str, api_key: str, workspace: str):

        self._labels = ["company", "location", "description"]
        self.client = rg.Argilla(
            api_url=api_url,
            api_key=api_key,
        )

        self._workspace_name = workspace
        self.workspace = self._get_or_create_workspace()

    def _get_or_create_workspace(self):
        my_workspace = self.client.workspaces(self._workspace_name)

        if not my_workspace:
            workspace = rg.Workspace(name="admin")
            my_workspace = workspace.create()
            my_workspace.add_user("argilla")
        return my_workspace

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
        dataset = rg.Dataset(name=dataset_name, settings=self.dataset_settings, client=self.client)
        dataset.create()
        return dataset

    def get_dataset(self, dataset_name: str):
        return self.client.datasets(dataset_name)

    def upload_records(self, dataset_name, records):
        dataset = self.get_or_create_dataset(dataset_name)
        dataset.records.log(records)

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

    def download_records(self, dataset: str, filename: str):
        dataset = self.get_or_create_dataset(dataset)
        dataset.records.to_json(filename)


if __name__ == "__main__":
    client = ArgillaLabelClient(
        api_url="http://localhost:6900",
        api_key="argilla.apikey",
        workspace="admin",
    )
    print(f"Client was created")
    if upload_file:
        records = client.create_records(upload_file)
        client.upload_records(client.dataset_name, records)
    if download_file:
        client.download_records(client.dataset_name, download_file)
