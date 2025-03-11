import pandas as pd
from app.clients.llm_clients import get_ai_client, get_tgi_client
from app.utils import parse_command_line_argilla, ArgillaLabelParams
from ast import literal_eval
import urllib3
from bs4 import BeautifulSoup
from typing import List
from config import config

from datetime import datetime

args: ArgillaLabelParams = parse_command_line_argilla()


class DescriptionSummarizerOpenAI:
    def __init__(self, data_collection: str):
        self._data: pd.DataFrame = self._read_data(f"data/{data_collection}.csv")
        self._openai_client = get_ai_client()
        self._output_file = f"data/{data_collection}_processed_{config.MODEL}_{datetime.now()}.csv"

    def summarize_descriptions(self, num_of_records: int = 10) -> None:
        data = self._data
        if num_of_records and num_of_records < len(data):
            data = self._data.sample(n=num_of_records)
        data.insert(4, f"description_{config.model}", "")
        for index, row in data.iterrows():
            descriptions = row["descriptions"]
            if row["homepage"]:
                try:
                    descriptions.extend(self.get_homepage_text(row["homepage"]))
                except Exception as e:
                    print(f"Failed to get homepage text for {row['homepage']}: {e}")
                    continue
            if not descriptions:
                continue
            summary = self._openai_client.generate_text_summary(descriptions, row["entity_name"])
            data.at[index, "description_openai"] = summary
        data.to_csv(self._output_file, index=False)

    def _read_data(self, data_file_path: str):
        return pd.read_csv(data_file_path, converters={"descriptions": literal_eval}, delimiter=";", header=0)

    @staticmethod
    def get_homepage_text(homepage_url: str) -> str:
        response = urllib3.request("GET", homepage_url)
        if response.status != 200:
            return ""
        homepage_bs4 = BeautifulSoup(response.data, 'html.parser')
        sentences = {item.strip() for item in homepage_bs4.get_text().split("\n") if item.strip()}
        for item in homepage_bs4.find_all("meta"):
            if item.get("content"):
                sentences.add(item.get("content"))
        return "\n".join([sentence for sentence in sentences if len(sentence.split()) > 4])


if __name__ == "__main__":
    summarizer = DescriptionSummarizerOpenAI(args.dataset_name)
    summarizer.summarize_descriptions(args.num_of_records)
    print(f"Summarized descriptions saved to {summarizer._output_file}")