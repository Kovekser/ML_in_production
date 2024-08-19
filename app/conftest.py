from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import urllib3
from app.clients.minio import MinioClient

from .main import get_application
from config import config


@pytest.fixture(scope="session")
def app() -> Generator[FastAPI, None, None]:
    app = get_application()
    yield app


@pytest.fixture(name="client")
def client_fixture(app: FastAPI) -> TestClient:
    client = TestClient(app)
    yield client


@pytest.fixture
def minio_client():
    client = MinioClient(http_client=urllib3.PoolManager())
    client.client.make_bucket("testbucket")
    yield client
