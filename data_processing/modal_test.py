import modal
import pandas as pd

IMAGE_MODEL_DIR = "/model"
DATA_PATH = "/data.csv"


def download_model():
    from huggingface_hub import snapshot_download

    model_name = "sentence-transformers/msmarco-distilbert-cos-v5"
    snapshot_download(model_name, local_dir=IMAGE_MODEL_DIR)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "nltk==3.9.1",
        "pandas==2.2.2",
        "sentence-transformers==3.0.1",
        "transformers==4.44.2",
        "ray==2.35.0",
        "torch==2.4.0"
    )
    # Use huggingface's hi-perf hf-transfer library to download this large model.
    .run_function(download_model)
    .copy_local_file("example.csv", DATA_PATH)
)


app = modal.App("example-get-started", image=image)

@app.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2


@app.function(gpu="A100")
def gpu_embeddings(workers=5):
    import torch
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import ray
    import time
    from sentence_transformers import SentenceTransformer

    print("This code is running on a remote GPU worker!")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    start = time.monotonic()
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    stop_words = set(stopwords.words('english'))

    model = SentenceTransformer(IMAGE_MODEL_DIR, device=device)
    print(f"Model loaded. Time: {time.monotonic() - start} seconds.")

    @ray.remote(num_gpus=1)
    def run_embedding_generator_ray(model, data: pd.DataFrame):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required but not available.")

        print(f"Running on {torch.cuda.get_device_name()}")
        def generate_embeddings(model, text: str):
            word_tokens = word_tokenize(text)
            filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
            return model.encode(filtered_sentence, show_progress_bar=True, convert_to_numpy=True)

        start = time.monotonic()
        embeddings = []
        for text in data['event_text']:
            vector = generate_embeddings(model, text)
            embeddings.append(vector)
        print(f"Gathering embeddings for {len(data['event_text'])} rows took {time.monotonic() - start} seconds.")
        return embeddings

    data = pd.read_csv(DATA_PATH)
    print(f"Data loaded: {data.shape[0]} rows.")

    chunk_size = data.shape[0] // workers
    print(f"Chunk size: {chunk_size} for {workers} workers.")

    futures = []

    for chunk in [data[i:i + chunk_size] for i in range(0, data.shape[0], chunk_size)]:
        future = run_embedding_generator_ray.remote(model, chunk)
        futures.append(future)

    embeddings = ray.get(futures)
    print(f"Gathering embeddings for {len(data['event_text'])} rows took {time.monotonic() - start} seconds.")
    return embeddings


def main():
    f = modal.Function.lookup("example-get-started", "gpu_embeddings")
    # print(f.remote(3))
    print(f.remote(workers=4))

if __name__ == "__main__":
    main()