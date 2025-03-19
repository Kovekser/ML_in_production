from typing import Any, List
from app.config import config
import wandb
import weave

wandb.login(key=config.wandb_api_key)
project = ".".join(config.MODEL.split(":"))
weave.init(project)


class BaseLLMClient:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._model = config.MODEL
        self._system_content = "You investor and funding events expert"

    def process_header_texts(self, header_texts: List[str]):
        prompt = f"""There is text from company homepage. 
        Extract and summarize information about the company's main economic activity. Use only text provided: {header_texts}"""

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

    @weave.op()
    def generate_text_summary(self, texts: List[str], company_name: str) -> str:
        prompt = f"""Extract and summarize key information about the company {company_name}. 
        Use only text provided: {texts}.  Dont start with: Based on the provided text, here's a summary, etc. 
        Description should be short and informative, up to 5 sentences without markup. 
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
