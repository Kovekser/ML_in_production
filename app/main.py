from fastapi import FastAPI


def get_application() -> FastAPI:
    _app = FastAPI(
        title="Test Server",
        description="Service for sending emails.",
    )
    return _app


app = get_application()


@app.get("/")
async def root():
    return {"message": "Hello World"}