from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from .main import get_application


@pytest.fixture(scope="session")
def app() -> Generator[FastAPI, None, None]:
    app = get_application()
    yield app


@pytest.fixture(name="client")
def client_fixture(app: FastAPI) -> TestClient:
    client = TestClient(app)
    yield client
