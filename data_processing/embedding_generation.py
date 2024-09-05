import modal
from sentence_transformers import SentenceTransformer
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import logging
from typing import Optional
import sys
import asyncio
import ray


model = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))


def generate_embeddings(model, text: str):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return model.encode(filtered_sentence, show_progress_bar=True, convert_to_numpy=True)


def run_embedding_generator_one_process(data: Optional[pd.DataFrame] = None) -> None:
    if data is None:
        data = pd.read_csv('example.csv')
    start = time.monotonic()
    embeddings = []
    for text in data['event_text']:
        vector = generate_embeddings(model=model, text=text)
        embeddings.append(vector)
    logging.info(f"Gathering embeddings for {len(data['event_text'])} rows took {time.monotonic() - start} seconds.")
    print(f"Gathering embeddings for {len(data['event_text'])} rows took {time.monotonic() - start} seconds.")
    return embeddings


def run_embedding_generator_process_pool(max_workers: int = 10):
    dataframe = pd.read_csv('example.csv')
    chunk_size = dataframe.shape[0] // max_workers

    start = time.monotonic()
    futures = []
    embeddings = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for chunk in [dataframe[i:i+chunk_size] for i in range(0, dataframe.shape[0], chunk_size)]:
            future = executor.submit(run_embedding_generator_one_process, chunk)
            futures.append(future)
        wait(futures)
    for future in futures:
        future_result = future.result()
        embeddings.append(future_result)
    print(f"Gathering embeddings for {len(dataframe['event_text'])} rows took {time.monotonic() - start} seconds.")
    return embeddings


async def run_embedding_generator_task_group(task_number: int = 4):
    dataframe = pd.read_csv('example.csv')
    chunk_size = dataframe.shape[0] // int(task_number)

    start = time.monotonic()
    tasks = []
    embeddings = []

    async with asyncio.TaskGroup() as tg:
        for chunk in [dataframe[i:i+chunk_size] for i in range(0, dataframe.shape[0], chunk_size)]:
            task = tg.create_task(run_embedding_generator_one_process(chunk))
            tasks.append(task)
    for task in tasks:
        task_result = task.result()
        embeddings.append(task_result)
    print(f"Gathering embeddings for {len(dataframe['event_text'])} rows took {time.monotonic() - start} seconds.")
    return embeddings


@ray.remote
def run_embedding_generator_ray(model, data: pd.DataFrame):
    start = time.monotonic()
    embeddings = []
    for text in data['event_text']:
        vector = generate_embeddings(model, text)
        embeddings.append(vector)
    logging.info(f"Gathering embeddings for {len(data['event_text'])} rows took {time.monotonic() - start} seconds.")
    print(f"Gathering embeddings for {len(data['event_text'])} rows took {time.monotonic() - start} seconds.")
    return embeddings


def run_ray_many(max_workers: int = 4):
    model = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    stop_words = set(stopwords.words('english'))

    dataframe = pd.read_csv('example.csv')
    chunk_size = dataframe.shape[0] // max_workers

    start = time.monotonic()
    futures = []

    for chunk in [dataframe[i:i + chunk_size] for i in range(0, dataframe.shape[0], chunk_size)]:
        future = run_embedding_generator_ray.remote(model, chunk)
        futures.append(future)

    embeddings = ray.get(futures)
    print(f"Gathering embeddings for {len(dataframe['event_text'])} rows took {time.monotonic() - start} seconds.")
    return embeddings


if __name__ == '__main__':
    my_function = globals()[sys.argv[1]]
    if len(sys.argv) > 2:
        args = sys.argv[2]
        my_function(args)
    else:
        my_function()

