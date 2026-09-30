from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.analyze import analyze_text


app = FastAPI(
    title="CraveShield AI 2",
    description="Personalized Trigger & Craving-State Intelligence API",
    version="1.0"
)


# --------------------------------------------------
# CORS
# Allows Flutter Web / Chrome to communicate
# with the FastAPI backend.
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class AnalysisRequest(BaseModel):
    text: str


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "CraveShield AI 2 API is running",
        "module": "Personalized Trigger & Craving-State Intelligence"
    }


# --------------------------------------------------
# AI 2 Analysis
# --------------------------------------------------

@app.post("/analyze")
def analyze(request: AnalysisRequest):

    result = analyze_text(
        request.text,
        save_history=True
    )

    return result