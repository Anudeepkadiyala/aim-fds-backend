from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware   # ✅ Added

from api.routers import deepfake
from api.routers import hate_speech, fake_news, fraud

app = FastAPI()

# ✅ Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow any frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(deepfake.router, prefix="/deepfake", tags=["Deepfake"])
app.include_router(hate_speech.router, tags=["Hate Speech"])
app.include_router(fake_news.router, tags=["Fake News"])
app.include_router(fraud.router, tags=["Fraud Detection"])
