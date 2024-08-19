import logging
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from .main import get_application
from app.clients import get_minio_client, MinioClient


@pytest.fixture(scope="session")
def app() -> Generator[FastAPI, None, None]:
    app = get_application()
    yield app


@pytest.fixture(name="client")
def client_fixture(app: FastAPI) -> TestClient:
    client = TestClient(app)
    yield client


@pytest.fixture
def minio_client() -> MinioClient:
    client = get_minio_client()
    if client.client.bucket_exists("testbucket"):
        client.delete_many_objects("testbucket", recursive=True)
        client.client.remove_bucket("testbucket")
    yield client
    if client.client.bucket_exists("testbucket"):
        client.delete_many_objects("testbucket", recursive=True)
        client.client.remove_bucket("testbucket")
