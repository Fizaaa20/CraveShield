from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from drift_detector import detect_behavioral_drift
from sequence_detector import analyze_behavior_sequence
from behavior_clustering import cluster_behavior, elbow_method
from forward_chaining import forward_chain
from text_classifier import classify_text


app = FastAPI(title="CraveShield AI 4 API")


# -----------------------------------
# CORS Configuration
# -----------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------
# Home
# -----------------------------------

@app.get("/")
def home():
    return {
        "message": "CraveShield AI 4 API is running!"
    }


# -----------------------------------
# Health Check
# -----------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "module": "AI 4 - Personal Behavioral Anomaly Detector"
    }


# -----------------------------------
# 1. Behavioral Drift Detection
# -----------------------------------

@app.post("/detect-drift")
def detect_drift(data: dict):

    baseline_days = data.get("baseline_days", [])
    recent_days = data.get("recent_days", [])

    return detect_behavioral_drift(
        baseline_days,
        recent_days
    )


# -----------------------------------
# 2. Behavioral Sequence Analysis
# -----------------------------------

@app.post("/analyze-sequence")
def analyze_sequence(data: dict):

    baseline_events = data.get("baseline_events", [])
    recent_events = data.get("recent_events", [])

    return analyze_behavior_sequence(
        baseline_events,
        recent_events
    )


# -----------------------------------
# 3. K-Means Behavioral Clustering
# -----------------------------------

@app.post("/cluster-behavior")
def cluster_behavior_api(data: dict):

    days = data.get("days", [])
    n_clusters = data.get("n_clusters", 2)

    return cluster_behavior(
        days,
        n_clusters
    )


# -----------------------------------
# 4. Elbow Method
# -----------------------------------

@app.post("/elbow-method")
def elbow_method_api(data: dict):

    days = data.get("days", [])
    max_k = data.get("max_k", 5)

    return elbow_method(
        days,
        max_k
    )


# -----------------------------------
# 5. Forward Chaining
# -----------------------------------

@app.post("/forward-chain")
def forward_chain_api(data: dict):

    return forward_chain(data)


# -----------------------------------
# 6. TF-IDF + Naive Bayes
# -----------------------------------

@app.post("/classify-text")
def classify_text_api(data: dict):

    text = data.get("text", "")

    return classify_text(text)


# -----------------------------------
# 7. Complete AI 4 Analysis
# -----------------------------------

@app.post("/analyze-all")
def analyze_all(data: dict):

    # Get input data
    baseline_days = data.get("baseline_days", [])
    recent_days = data.get("recent_days", [])

    baseline_events = data.get("baseline_events", [])
    recent_events = data.get("recent_events", [])

    journal_text = data.get("journal_text", "")


    # -----------------------------------
    # 1. Behavioral Drift Detection
    # -----------------------------------

    drift_result = detect_behavioral_drift(
        baseline_days,
        recent_days
    )


    # -----------------------------------
    # 2. Behavioral Sequence Analysis
    # -----------------------------------

    sequence_result = analyze_behavior_sequence(
        baseline_events,
        recent_events
    )


    # -----------------------------------
    # 3. K-Means Clustering
    # -----------------------------------

    clustering_result = cluster_behavior(
        recent_days,
        n_clusters=2
    )


    # -----------------------------------
    # 4. Elbow Method
    # -----------------------------------

    elbow_result = elbow_method(
        recent_days,
        max_k=5
    )


    # -----------------------------------
    # 5. TF-IDF + Naive Bayes
    # -----------------------------------

    text_result = classify_text(
        journal_text
    )


    # -----------------------------------
    # 6. Prepare Facts for Forward Chaining
    # -----------------------------------

    facts = {
        "app_usage_increased": False,
        "coping_sessions_decreased": False,
        "checkins_decreased": False,
        "journal_activity_decreased": False,

        "significant_behavioral_drift": (
            drift_result.get("status") == "significant_drift"
        ),

        "sequence_changed": (
            sequence_result.get("status")
            == "significant_sequence_change"
        )
    }


    # -----------------------------------
    # Compare Baseline and Recent Activity
    # -----------------------------------

    if baseline_days and recent_days:

        # App usage
        avg_baseline_app = sum(
            day.get("app_open_count", 0)
            for day in baseline_days
        ) / len(baseline_days)

        avg_recent_app = sum(
            day.get("app_open_count", 0)
            for day in recent_days
        ) / len(recent_days)


        # Coping sessions
        avg_baseline_coping = sum(
            day.get("coping_sessions", 0)
            for day in baseline_days
        ) / len(baseline_days)

        avg_recent_coping = sum(
            day.get("coping_sessions", 0)
            for day in recent_days
        ) / len(recent_days)


        # Check-ins
        avg_baseline_checkins = sum(
            day.get("checkin_completion", 0)
            for day in baseline_days
        ) / len(baseline_days)

        avg_recent_checkins = sum(
            day.get("checkin_completion", 0)
            for day in recent_days
        ) / len(recent_days)


        # Journaling
        avg_baseline_journal = sum(
            day.get("journal_count", 0)
            for day in baseline_days
        ) / len(baseline_days)

        avg_recent_journal = sum(
            day.get("journal_count", 0)
            for day in recent_days
        ) / len(recent_days)


        # -----------------------------------
        # Generate Facts
        # -----------------------------------

        facts["app_usage_increased"] = (
            avg_recent_app > avg_baseline_app
        )

        facts["coping_sessions_decreased"] = (
            avg_recent_coping < avg_baseline_coping
        )

        facts["checkins_decreased"] = (
            avg_recent_checkins < avg_baseline_checkins
        )

        facts["journal_activity_decreased"] = (
            avg_recent_journal < avg_baseline_journal
        )


    # -----------------------------------
    # 7. Forward Chaining
    # -----------------------------------

    forward_result = forward_chain(
        facts
    )


    # -----------------------------------
    # Final Combined Result
    # -----------------------------------

    return {

        "behavioral_drift": drift_result,

        "sequence_analysis": sequence_result,

        "behavior_clustering": clustering_result,

        "elbow_method": elbow_result,

        "text_classification": text_result,

        "forward_chaining": forward_result
    }