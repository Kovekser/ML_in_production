from pydantic import BaseModel, field_validator
from enum import Enum
from openai import OpenAI
from typing import List


class LabelsEnum(str, Enum):
    location = "location"
    company = "company"
    description = "description"


class LabelResponseOne(BaseModel):
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
        return response.choices[0].message.parsed.entities
