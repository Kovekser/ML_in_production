from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

model = get_registry().get("sentence-transformers").create(name="sentence-transformers/msmarco-distilbert-cos-v5", device="cpu")


class Words(LanceModel):
    id: str
    event_text: str = model.SourceField()
    url: str
    title: str
    vector: Vector(model.ndims()) = model.VectorField()
