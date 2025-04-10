from functools import partial
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from os import path
import boto3
import json
import io


class SummaryDatasetFormatter:
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
        assistant_message = {"role": "assistant", "content": row["summary"]}
        messages.append(assistant_message)
        print(f"Messages: {messages}")  # Debugging
        return {"messages": messages}

    def format_data_chat_template(self, row, tokenizer):
        chat_template = tokenizer.apply_chat_template(
                row["messages"], add_generation_prompt=False, tokenize=False
            )
        return {"text": chat_template}

    def format_dataset(self, train_file_name: str, test_file_name: str) -> DatasetDict:
        if self._s3_bucket:
            s3_resource = boto3.Session().resource('s3').Bucket(self._s3_bucket)
            train_response = s3_resource.Object(f"{path.join(self._path_to_data, train_file_name)}").get()
            test_response = s3_resource.Object(f"{path.join(self._path_to_data, test_file_name)}").get()
            train_json = json.load(io.BytesIO(train_response["Body"].read()))
            test_json = json.load(io.BytesIO(test_response["Body"].read()))
            dataset = DatasetDict(
                {
                    "train": Dataset.from_list(train_json),
                    "test": Dataset.from_list(test_json),
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


def format_dataset_summaries(model_id: str, train_file: str, test_file: str, path_to_data: str) -> DatasetDict:
    processor = SummaryDatasetFormatter(model_id, path_to_data)
    return processor.format_dataset(train_file_name=train_file, test_file_name=test_file)
