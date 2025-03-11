from fastapi import APIRouter, FastAPI
from contextlib import asynccontextmanager
import wandb
from config import config

api_router = APIRouter()

@api_router.get("/", name="default")
async def root():
    return {"message": "Hello World"}


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore
    wandb.login(key=config.WANDB_API_KEY)
    yield


def get_application() -> FastAPI:
    _app = FastAPI(
        title="Test Server",
        description="Service for sending emails.",
        lifespan=lifespan,
    )
    _app.include_router(api_router)
    return _app


app = get_application()


