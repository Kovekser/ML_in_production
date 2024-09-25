from pydantic import BaseModel
from enum import Enum
from openai import OpenAI
from typing import List, Set
import re


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


class OpenAIClient:
    def __init__(self):
        self.client = OpenAI()

    def generate_suggestions(self, text: str) -> List[LabelResponseOne]:
        prompt = f"""There is funding event article: {text}
Put labels to information from this article (if it is available) : company, location, description. Labels should not overlap.
Fields start and end for labels should be exact positions of symbols in the original text.
Description is the text about current main company's economic activity.
There may be multiple labels of the same type in the text.
All available locations and companies should be labeled if they are mentioned explicitly.
Description which describe current main economic activity only should be labeled excluding funding event
        """
        results = set()
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You investor and funding events expert"},
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
