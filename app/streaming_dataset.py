import logging

from uuid import uuid4
from streaming import MDSWriter, StreamingDataset
import os
from torch.utils.data import DataLoader
from csv import DictReader

# Local or remote directory path to store the output compressed files.
out_root = 'examples'
local_dir = 'local_cache'
for directory in [out_root, local_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Path {directory} created: {os.path.exists(directory)}")

# Compression algorithm name
compression = 'zstd'

columns = {
    "id": "str",
    "url": "str",
    "event_text": "str",
    "title": "str",
}

samples = []

with open('data.csv') as file:
    reader = DictReader(file)
    for row in reader:
        samples.append({
            "id": str(uuid4()),
            "url": row['url'],
            "event_text": row['event_text'],
            "title": row['title']

        })

# Use `MDSWriter` to iterate through the input data and write to a collection of `.mds` files.
with MDSWriter(out=out_root, columns=columns, compression=compression) as out:
    for sample in samples:
        out.write(sample)
logging.info(f"Data was written to {out_root} successfully.")

logging.info(f"Path {local_dir} exists: {os.path.exists(local_dir)}")
dataset = StreamingDataset(local=local_dir, remote=out_root, batch_size=1, split=None, shuffle=True)

# Create PyTorch DataLoader
dataloader = DataLoader(dataset, batch_size=1)
logging.info(f"DataLoader created successfully.")