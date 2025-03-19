__all__ = ["OpenAIClient", "LabelsEnum", "LabelResponseOne", "LabelResponse", "get_ai_client"]
import wandb
from pydantic import BaseModel
from enum import Enum
from openai import OpenAI
from typing import List, Set
import re
import weave
from app.config import config
from .base import BaseLLMClient
import json
import pandas as pd


wandb.login(key=config.wandb_api_key)
project = ".".join(config.MODEL.split(":"))
weave.init(project)


class LabelsEnum(str, Enum):
    location = "location"
    company = "company"
    description = "description"


class LabelResponseOne(BaseModel, frozen=True):
    label: LabelsEnum
    start: int
    end: int
    text: str


class LabelResponse(BaseModel):
    entities: List[LabelResponseOne]


class OpenAIClient(BaseLLMClient):
    def __init__(self, client: OpenAI = None) -> None:
        super().__init__(client)
        self._summary_prompt_template = lambda c_name, texts: f"""Extract and summarize key information about 
        the company {c_name}. Use only text provided: {texts}.  Dont start with: Based on the provided text, 
        here's a summary, etc. Description should be short and informative, up to 5 sentences without markup. 
        If it is impossible to extract relevant information return None
        """

    def generate_summaries_batch_request_upload(self, data: pd.DataFrame):
        batch_requests = []

        for index, row in data.iterrows():
            request_item = {
                "custom_id": row["request_id"],  # Custom ID for tracking
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": self._summary_prompt_template(row["entity_name"], row["descriptions"])},
                    ],
                    "temperature": 1,
                }
            }
            batch_requests.append(request_item)

        # Convert to JSONL format (newline-delimited JSON)
        batch_jsonl = "\n".join(json.dumps(request) for request in batch_requests)

        # Save to a .jsonl file
        with open("data/batch_requests/summary.jsonl", "w") as file:
            file.write(batch_jsonl)

        # Upload batch request file
        batch_input_file = self._client.files.create(
            file=open("data/batch_requests/summary.jsonl", "rb"),
            purpose="batch"
        )

        print(batch_input_file)

        # create batch job
        batch_object = self._client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "synthetic summaries for company descriptions"
            }
        )
        with open("data/batches/batch_summary.json", "w") as file:
            file.write(json.dumps(batch_object))

    def check_batch_status_and_download(self, batch_id: str):
        batch = self._client.batches.retrieve(batch_id)
        print(batch)
        if batch["status"] == "completed":
            batch_output_file = self._client.files.content(batch["output_file_id"])
            print("Downloaded batch responses")
        return batch

    @weave.op()
    def generate_labels(self, text: str) -> List[LabelResponseOne]:
        prompt = f"""There is funding event article: {text}
Put labels to information from this article (if it is available) : company, location, description. Labels should not overlap.
Fields start and end for labels should be exact positions of symbols in the original text.
Description is the text about current main company's economic activity.
There may be multiple labels of the same type in the text.
All available locations and companies should be labeled if they are mentioned explicitly.
Description which describe current main economic activity only should be labeled excluding funding event
        """
        results = set()
        response = self._client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self._system_content},
                {"role": "user", "content": prompt},
            ],
            response_format=LabelResponse,
            temperature=1,
        )
        print(f"OpenAI response: {response.choices[0].message.parsed.entities}")
        labels = response.choices[0].message.parsed.entities
        text_labels = {label.text for label in labels}
        for label in labels:
            results.update(self.adjust_text_start_end(text, label))
        return self.remove_overlapping_labels(results, text_labels)

    @staticmethod
    def adjust_text_start_end(text: str, label: LabelResponseOne) -> List[LabelResponseOne]:
        results = []
        matches = re.finditer(label.text, text)
        for match in matches:
            results.append(LabelResponseOne(label=label.label, start=match.start(), end=match.end(), text=label.text))
        return results

    @staticmethod
    def remove_overlapping_labels(labels: Set[LabelResponseOne], text_labels: Set[str]) -> List[LabelResponseOne]:
        labels = sorted(
            list(result for result in labels if result.text in text_labels),
            key=lambda x: (x.start, len(x.text)),
        )
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i].start <= labels[j].start <= labels[i].end:
                    if len(labels[j].text) < len(labels[j].text):
                        labels.pop(j)
                    else:
                        labels.pop(i)
                    break
        return labels


def get_ai_client():
    if config.MODEL != "gpt-4o":
        client = OpenAI(base_url=config.MODEL_BASE_URL, api_key=config.MODEL_API_KEY)
    else:
        client = OpenAI()
    return OpenAIClient(client=client)
