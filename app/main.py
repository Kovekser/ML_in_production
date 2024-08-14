from fastapi import APIRouter, FastAPI

api_router = APIRouter()

@api_router.get("/", name="default")
async def root():
    return {"message": "Hello World"}


def get_application() -> FastAPI:
    _app = FastAPI(
        title="Test Server",
        description="Service for sending emails.",
    )
    _app.include_router(api_router)
    return _app


app = get_application()


