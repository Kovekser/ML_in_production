import pandas as pd
import timeit
import os

dataframe = pd.read_csv('example.csv', header=0, index_col=0)
dataframe.to_pickle('example.pkl')
dataframe.to_json('local_train.json')
dataframe.to_feather('example.feather')
dataframe.to_parquet('example.parquet')

for file_ in ('example.pkl', 'example.csv', 'example.feather', 'example.parquet', 'local_train.json'):
    print(f"File size of {file_}",os.path.getsize(file_))


def load_pickle():
    dataframe.to_pickle('example.pkl')


def read_pickle():
    filename = 'example.pkl'
    temp = pd.read_pickle(filename)
    print(f"Data_match for {filename} is:", dataframe.compare(temp))


def read_csv():
    filename = 'example.csv'
    temp = pd.read_csv(filename, header=0)
    print(f"Data_match for {filename} is:", dataframe.compare(temp))

def load_csv():
    dataframe.to_csv('example.csv')


def load_feather():
    dataframe.to_feather('example.feather')


def read_feather():
    filename = 'example.feather'
    temp = pd.read_feather(filename)
    print(f"Data_match for {filename} is:", dataframe.compare(temp))


def load_parquet():
    dataframe.to_parquet('example.parquet')


def read_parquet():
    filename = 'example.parquet'
    temp = pd.read_parquet(filename)
    print(f"Data_match for {filename} is:", dataframe.compare(temp))


def load_json():
    dataframe.to_json('local_train.json')


def read_json():
    filename = 'local_train.json'
    temp = pd.read_json(filename)
    print(f"Data_match for {filename} is:", dataframe.compare(temp))


if __name__ == '__main__':
    print('Load pickle:', timeit.timeit(load_pickle, number=100))
    print('Read pickle:', timeit.timeit(read_pickle, number=100))
    print('Load csv:', timeit.timeit(load_csv, number=100))
    print('Read csv:', timeit.timeit(read_csv, number=100))
    print('Load feather:', timeit.timeit(load_feather, number=100))
    print('Read feather:', timeit.timeit(read_feather, number=100))
    print('Load parquet:', timeit.timeit(load_parquet, number=100))
    print('Read parquet:', timeit.timeit(read_parquet, number=100))
    print('Load json:', timeit.timeit(load_json, number=100))
    print('Read json:', timeit.timeit(read_json, number=100))
