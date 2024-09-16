import os
import logging
from csv import DictReader
from uuid import uuid4
from mds import MDSWriter, StreamingDataset  # Assuming MDSWriter and StreamingDataset are imported correctly
from torch.utils.data import DataLoader


class DatasetManager:
    def __init__(self, data_file, out_root='examples', local_dir='local_cache', compression='zstd'):
        self.compression = compression

        self._out_root = out_root
        self._local_dir = local_dir
        self._setup_directories()

        self.data_file = data_file
        self.columns = None
        self.samples = []

        self.load_data()

    def _setup_directories(self):
        """Create output and local directories if they don't exist."""
        for directory in [self._out_root, self._local_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Path {directory} created: {os.path.exists(directory)}")

    def load_data(self):
        """Load data from CSV and append it to samples."""
        with open(self.data_file) as file:
            reader = DictReader(file)
            self.samples.extend(list(reader))
            self.columns = reader.fieldnames
        logging.info("Data loaded successfully from CSV.")

    def write_mds(self):
        """Write data to MDS files."""
        with MDSWriter(out=self._out_root, columns=self.columns, compression=self.compression) as out:
            for sample in self.samples:
                out.write(sample)
        logging.info(f"Data was written to {self._out_root} successfully.")

    def create_dataloader(self, batch_size=1, shuffle=True):
        """Create a PyTorch DataLoader from StreamingDataset."""
        logging.info(f"Path {self._local_dir} exists: {os.path.exists(self._local_dir)}")
        dataset = StreamingDataset(local=self._local_dir, remote=self._out_root, batch_size=batch_size, split=None, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        logging.info("DataLoader created successfully.")
        return dataloader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    manager = DatasetManager(data_file='data.csv')
    manager.load_data()
    manager.write_mds()
    dataloader = manager.create_dataloader(batch_size=1)
