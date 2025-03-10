from transformers import pipeline
import wandb
import weave
from config import config
from typing import List


wandb.login(key=config.wandb_api_key)
project = ".".join(config.MODEL.split(":"))
weave.init(project)

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class T5Client:
    def __init__(self):
        pass

    @weave.op()
    def generate_text_summary(self, texts: List[str], company_name: str) -> str:
        prompt = f"""Extract and summarize key information about the company {company_name} in 5 sentences.
            If it is impossible to extract relevant information return None
            """

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_content},
                {"role": "user", "content": prompt},
            ],
            temperature=1,
        )
        content = response.choices[0].message.content
        return content