__all__ = ["get_tgi_client", "TGIClient"]
from huggingface_hub import InferenceClient
from .base import BaseLLMClient


class TGIClient(BaseLLMClient):
    def __init__(self, client: InferenceClient) -> None:
        super().__init__(client)


def get_tgi_client() -> TGIClient:
    return TGIClient(
        client=InferenceClient(
            base_url="http://localhost:8080/v1/",
        )
    )