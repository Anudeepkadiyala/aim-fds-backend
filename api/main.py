from fastapi import FastAPI
from api.routers import deepfake
from api.routers import hate_speech, fake_news, fraud

app = FastAPI()


app.include_router(deepfake.router, prefix="/deepfake", tags=["Deepfake"])
app.include_router(hate_speech.router, tags=["Hate Speech"])
app.include_router(fake_news.router, tags=["Fake News"])
app.include_router(fraud.router, tags=["Fraud Detection"])