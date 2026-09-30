from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from risk_predictor import predict_risk


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="CraveShield A1-1 Risk Predictor API",
    description="Backend API for the Personalized Craving Risk Predictor",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INPUT MODEL
# ============================================================

class RiskInput(BaseModel):

    craving_level: int
    stress_level: int
    previous_craving: int
    previous_risk: float
    emotion_intensity: int
    anomaly_score: float

    mood: str
    situation: str
    trigger: str
    time_of_day: str
    day_of_week: str


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "CraveShield A1-1 Risk Predictor API is running",
        "status": "success"
    }


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.post("/predict")
def predict(data: RiskInput):

    # Prepare data exactly in the format
    # expected by the Random Forest model.

    user_data = pd.DataFrame([{

        "craving_level": data.craving_level,

        "stress_level": data.stress_level,

        "previous_craving": data.previous_craving,

        "previous_risk": data.previous_risk,

        "emotion_intensity": data.emotion_intensity,

        "anomaly_score": data.anomaly_score,

        "mood": data.mood.lower(),

        "situation": data.situation.lower(),

        "trigger": data.trigger.lower(),

        "time_of_day": data.time_of_day.lower(),

        "day_of_week": data.day_of_week.capitalize()

    }])


    # ========================================================
    # RANDOM FOREST PREDICTION
    # ========================================================

    probability, risk_score, risk_level = predict_risk(
        user_data
    )


    # ========================================================
    # PERSONAL TREND
    # ========================================================

    difference = risk_score - data.previous_risk


    if difference > 10:

        trend = "INCREASING"

    elif difference < -10:

        trend = "DECREASING"

    else:

        trend = "STABLE"


    # ========================================================
    # RISK FACTORS
    # ========================================================

    risk_factors = []


    if data.craving_level >= 7:

        risk_factors.append(
            "High current craving"
        )


    if data.stress_level >= 7:

        risk_factors.append(
            "High stress"
        )


    if data.previous_craving >= 7:

        risk_factors.append(
            "Previous craving level was high"
        )


    if data.previous_risk >= 70:

        risk_factors.append(
            "Previous high-risk pattern"
        )


    if data.emotion_intensity >= 7:

        risk_factors.append(
            "High emotional intensity"
        )


    if data.anomaly_score >= 70:

        risk_factors.append(
            "Unusual behavioral pattern detected"
        )


    if data.mood.lower() in [
        "stressed",
        "anxious",
        "sad"
    ]:

        risk_factors.append(
            "Challenging emotional state detected"
        )


    if data.trigger.lower() != "none":

        risk_factors.append(
            "A trigger has been reported"
        )


    if data.situation.lower() == "alone":

        risk_factors.append(
            "User is currently alone"
        )


    if not risk_factors:

        risk_factors.append(
            "No major high-risk factor detected"
        )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    if risk_level == "HIGH":

        analysis = (
            "Current craving and stress levels are "
            "contributing strongly to the predicted risk."
        )

    elif risk_level == "MEDIUM":

        analysis = (
            "Some current factors are contributing "
            "to the predicted risk. Continue monitoring "
            "your current state."
        )

    else:

        analysis = (
            "Current inputs indicate a relatively "
            "lower predicted risk at this check-in."
        )


    # ========================================================
    # JSON RESPONSE
    # ========================================================

    return {

        "status": "success",

        "risk_score": risk_score,

        "risk_level": risk_level,

        "model_probability": round(
            float(probability) * 100,
            2
        ),

        "personal_risk_trend": trend,

        "risk_change": difference,

        "risk_factors": risk_factors,

        "ai_analysis": analysis,

        "check_in": {

            "craving_level": data.craving_level,

            "stress_level": data.stress_level,

            "previous_craving": data.previous_craving,

            "previous_risk": data.previous_risk,

            "emotion_intensity": data.emotion_intensity,

            "anomaly_score": data.anomaly_score,

            "mood": data.mood,

            "situation": data.situation,

            "trigger": data.trigger,

            "time_of_day": data.time_of_day,

            "day_of_week": data.day_of_week

        }

    }