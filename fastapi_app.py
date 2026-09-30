
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from drift_detector import detect_behavioral_drift
from sequence_detector import analyze_behavior_sequence

app = FastAPI(title="CraveShield AI 4 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "CraveShield AI 4 API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "module": "AI 4 - Personal Behavioral Anomaly Detector"
    }


@app.post("/detect-drift")
def detect_drift(data: dict):
    baseline_days = data.get("baseline_days", [])
    recent_days = data.get("recent_days", [])

    return detect_behavioral_drift(
        baseline_days,
        recent_days
    )


@app.post("/analyze-sequence")
def analyze_sequence(data: dict):
    baseline_events = data.get("baseline_events", [])
    recent_events = data.get("recent_events", [])

    return analyze_behavior_sequence(
        baseline_events,
        recent_events
    )


@app.post("/analyze-all")
def analyze_all(data: dict):
    baseline_days = data.get("baseline_days", [])
    recent_days = data.get("recent_days", [])

    baseline_events = data.get("baseline_events", [])
    recent_events = data.get("recent_events", [])

    drift_result = detect_behavioral_drift(
        baseline_days,
        recent_days
    )

    sequence_result = analyze_behavior_sequence(
        baseline_events,
        recent_events
    )

    return {
        "behavioral_drift": drift_result,
        "sequence_analysis": sequence_result
    }