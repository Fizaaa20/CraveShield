# ============================================================
# CRAVESHIELD - A1-1
# PERSONALIZED CRAVING RISK PREDICTOR
# ============================================================

import os
from datetime import datetime

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_FILE = "risk_history.csv"

RANDOM_STATE = 42

LOW_RISK_LIMIT = 40
HIGH_RISK_LIMIT = 70


# ============================================================
# SYNTHETIC DEMO TRAINING DATA
# ============================================================

data = {
    "craving_level": [
        2, 3, 4, 8, 9, 7,
        2, 5, 8, 3, 6, 9,
        4, 7, 2, 8, 5, 9
    ],

    "stress_level": [
        2, 3, 4, 8, 9, 7,
        2, 5, 8, 3, 6, 9,
        4, 7, 2, 8, 5, 9
    ],

    "previous_craving": [
        2, 3, 3, 7, 8, 6,
        2, 4, 7, 2, 5, 8,
        3, 6, 2, 7, 4, 8
    ],

    "previous_risk": [
        15, 20, 25, 75, 85, 65,
        10, 45, 80, 20, 55, 90,
        30, 70, 15, 75, 40, 85
    ],

    "emotion_intensity": [
        2, 3, 4, 8, 9, 7,
        2, 5, 8, 3, 6, 9,
        4, 7, 2, 8, 5, 9
    ],

    "anomaly_score": [
        10, 15, 20, 80, 90, 70,
        5, 40, 85, 15, 60, 95,
        25, 75, 10, 80, 45, 90
    ],

    "mood": [
        "calm", "happy", "neutral", "stressed",
        "anxious", "sad", "calm", "neutral",
        "stressed", "happy", "anxious", "sad",
        "neutral", "stressed", "calm", "anxious",
        "neutral", "sad"
    ],

    "situation": [
        "with_family", "social", "with_family",
        "alone", "alone", "alone", "social",
        "with_family", "alone", "social", "alone",
        "alone", "social", "alone", "with_family",
        "alone", "social", "alone"
    ],

    "trigger": [
        "none", "none", "none", "stress",
        "stress", "emotional", "none", "none",
        "stress", "none", "emotional", "stress",
        "none", "stress", "none", "emotional",
        "none", "stress"
    ],

    "time_of_day": [
        "morning", "afternoon", "morning", "night",
        "night", "evening", "morning", "afternoon",
        "night", "morning", "evening", "night",
        "afternoon", "night", "morning", "evening",
        "afternoon", "night"
    ],

    "day_of_week": [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday", "Monday",
        "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday", "Monday", "Tuesday",
        "Wednesday", "Thursday"
    ],

    "risk": [
        0, 0, 0, 1, 1, 1,
        0, 0, 1, 0, 1, 1,
        0, 1, 0, 1, 0, 1
    ]
}


df = pd.DataFrame(data)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

numeric_features = [
    "craving_level",
    "stress_level",
    "previous_craving",
    "previous_risk",
    "emotion_intensity",
    "anomaly_score"
]

categorical_features = [
    "mood",
    "situation",
    "trigger",
    "time_of_day",
    "day_of_week"
]

all_features = numeric_features + categorical_features


# ============================================================
# CREATE RANDOM FOREST MODEL
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE
            )
        )
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

X = df[all_features]
y = df["risk"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# FINAL MODEL - TRAIN USING ALL DEMO DATA
# ============================================================

model.fit(X, y)


# ============================================================
# RISK PREDICTION FUNCTION
# ============================================================

def predict_risk(user_data):
    """
    Predict risk using the trained Random Forest model.

    This function is used by FastAPI.
    """

    probability = model.predict_proba(user_data)[0][1]

    risk_score = round(probability * 100)

    if risk_score < LOW_RISK_LIMIT:
        risk_level = "LOW"

    elif risk_score < HIGH_RISK_LIMIT:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    return probability, risk_score, risk_level


# ============================================================
# TERMINAL VERSION
# ============================================================

def get_integer(prompt, minimum, maximum):

    while True:

        try:

            value = int(input(prompt))

            if minimum <= value <= maximum:
                return value

            print(
                f"Please enter a value between "
                f"{minimum} and {maximum}."
            )

        except ValueError:

            print("Please enter a valid number.")


def get_text(prompt, allowed_values):

    while True:

        value = input(prompt).strip().lower()

        if value in allowed_values:
            return value

        print(
            "Please choose one of: "
            + ", ".join(allowed_values)
        )


def run_terminal_predictor():

    print("\n===================================================")
    print("             CRAVESHIELD - A1-1")
    print("       PERSONALIZED RISK PREDICTOR")
    print("===================================================")

    print("\nMODEL EVALUATION")
    print("---------------------------------------")

    print(f"Accuracy : {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\n===================================================")
    print("                 USER CHECK-IN")
    print("===================================================")

    print("\nEnter the current user information.")

    current_craving = get_integer(
        "Current craving level (1-10): ",
        1,
        10
    )

    stress = get_integer(
        "Stress level (1-10): ",
        1,
        10
    )

    previous_craving = get_integer(
        "Previous craving level (1-10): ",
        1,
        10
    )

    previous_risk = get_integer(
        "Previous risk score (0-100): ",
        0,
        100
    )

    emotion_intensity = get_integer(
        "Emotion intensity from A1-2 (1-10): ",
        1,
        10
    )

    anomaly_score = get_integer(
        "Anomaly score from A1-4 (0-100): ",
        0,
        100
    )

    mood = get_text(
        "Mood (calm/happy/neutral/stressed/anxious/sad): ",
        [
            "calm",
            "happy",
            "neutral",
            "stressed",
            "anxious",
            "sad"
        ]
    )

    situation = get_text(
        "Situation (alone/social/with_family): ",
        [
            "alone",
            "social",
            "with_family"
        ]
    )

    trigger = get_text(
        "Trigger (none/stress/emotional): ",
        [
            "none",
            "stress",
            "emotional"
        ]
    )

    time_of_day = get_text(
        "Time of day (morning/afternoon/evening/night): ",
        [
            "morning",
            "afternoon",
            "evening",
            "night"
        ]
    )

    day_of_week = get_text(
        "Day of week: ",
        [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday"
        ]
    ).capitalize()

    user_data = pd.DataFrame([{
        "craving_level": current_craving,
        "stress_level": stress,
        "previous_craving": previous_craving,
        "previous_risk": previous_risk,
        "emotion_intensity": emotion_intensity,
        "anomaly_score": anomaly_score,
        "mood": mood,
        "situation": situation,
        "trigger": trigger,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week
    }])

    probability, risk_score, risk_level = predict_risk(
        user_data
    )

    difference = risk_score - previous_risk

    if difference > 10:
        trend = "INCREASING"

    elif difference < -10:
        trend = "DECREASING"

    else:
        trend = "STABLE"

    risk_factors = []

    if current_craving >= 7:
        risk_factors.append(
            "Current craving level is high."
        )

    if stress >= 7:
        risk_factors.append(
            "Stress level is high."
        )

    if previous_craving >= 7:
        risk_factors.append(
            "Previous craving level was high."
        )

    if previous_risk >= 70:
        risk_factors.append(
            "Previous risk score was high."
        )

    if emotion_intensity >= 7:
        risk_factors.append(
            "Emotion intensity reported by A1-2 is high."
        )

    if anomaly_score >= 70:
        risk_factors.append(
            "A1-4 reported a high anomaly score."
        )

    if mood in ["stressed", "anxious", "sad"]:
        risk_factors.append(
            "Current mood indicates a challenging emotional state."
        )

    if trigger != "none":
        risk_factors.append(
            "A trigger has been reported."
        )

    if situation == "alone":
        risk_factors.append(
            "User is currently alone."
        )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    history_record = {
        "timestamp": timestamp,
        "craving_level": current_craving,
        "stress_level": stress,
        "previous_craving": previous_craving,
        "previous_risk": previous_risk,
        "emotion_intensity": emotion_intensity,
        "anomaly_score": anomaly_score,
        "mood": mood,
        "situation": situation,
        "trigger": trigger,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "trend": trend
    }

    new_record = pd.DataFrame([history_record])

    if os.path.exists(HISTORY_FILE):

        history = pd.read_csv(HISTORY_FILE)

        history = pd.concat(
            [history, new_record],
            ignore_index=True
        )

    else:

        history = new_record

    history.to_csv(
        HISTORY_FILE,
        index=False
    )

    print("\n===================================================")
    print("                PREDICTION RESULT")
    print("===================================================")

    print(f"\nRisk Score        : {risk_score}/100")
    print(f"Risk Level        : {risk_level}")
    print(f"Model Probability : {probability:.2f}")
    print(f"Previous Risk     : {previous_risk}/100")
    print(f"Risk Change       : {difference:+d}")
    print(f"Current Trend     : {trend}")

    print("\nWHY THIS PREDICTION?")
    print("---------------------------------------")

    if risk_factors:

        for number, factor in enumerate(
            risk_factors,
            start=1
        ):
            print(f"{number}. {factor}")

    else:

        print(
            "No major rule-based risk factors detected."
        )

    print("\n===================================================")
    print("             PERSONAL RISK ANALYSIS")
    print("===================================================")

    print(
        f"\nTotal predictions : {len(history)}"
    )

    print(
        f"Average risk      : "
        f"{history['risk_score'].mean():.1f}/100"
    )

    print(
        f"Highest risk      : "
        f"{history['risk_score'].max()}/100"
    )

    print(
        f"Lowest risk       : "
        f"{history['risk_score'].min()}/100"
    )

    print(
        f"Recent average    : "
        f"{history.tail(5)['risk_score'].mean():.1f}/100"
    )

    print("\nRisk Distribution")
    print("---------------------------------------")

    print(
        f"LOW    : "
        f"{(history['risk_level'] == 'LOW').sum()}"
    )

    print(
        f"MEDIUM : "
        f"{(history['risk_level'] == 'MEDIUM').sum()}"
    )

    print(
        f"HIGH   : "
        f"{(history['risk_level'] == 'HIGH').sum()}"
    )

    print("\n===================================================")
    print("              RECENT RISK TREND")
    print("===================================================")

    for index, row in history.tail(10).iterrows():

        score = int(row["risk_score"])

        bars = "█" * (score // 5)

        print(
            f"{index + 1:02d} | "
            f"{bars:<20} "
            f"{score}/100 "
            f"{row['risk_level']}"
        )

    print("\n===================================================")
    print("             RECENT PREDICTIONS")
    print("===================================================")

    print(
        history.tail(5)[
            [
                "timestamp",
                "risk_score",
                "risk_level",
                "trend"
            ]
        ].to_string(index=False)
    )

    print("\n===================================================")
    print(
        f"Predictions saved : {len(history)}"
    )
    print(
        f"History file      : {HISTORY_FILE}"
    )
    print("===================================================")

    print(
        "\nNOTE:"
    )

    print(
        "Training data is synthetic demo data."
    )

    print(
        "Risk thresholds are prototype conventions."
    )

    print(
        "This system is a student prototype, "
        "not a clinical prediction or diagnostic system."
    )


# ============================================================
# IMPORTANT
# ============================================================
#
# This means:
#
# python risk_predictor.py
#     -> runs terminal version
#
# from risk_predictor import model
#     -> ONLY loads the model
#     -> DOES NOT ask terminal questions
#
# This allows FastAPI to use the model.
# ============================================================

if __name__ == "__main__":
    run_terminal_predictor()