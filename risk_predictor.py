# ============================================================
# CRAVESHIELD - A1-1
# PERSONALIZED CRAVING RISK PREDICTOR
# ============================================================
#
# AI Model       : Random Forest Classifier
# Input Sources  : User + A1-2 + A1-4
# Output         : Risk Score + Risk Level + Explanation
# History        : risk_history.csv
#
# IMPORTANT:
# This project uses synthetic demo training data.
# It is a student prototype and NOT a clinical prediction
# or diagnostic system.
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
# 1. CONFIGURATION
# ============================================================

HISTORY_FILE = "risk_history.csv"

RANDOM_STATE = 42

# Prototype thresholds
LOW_RISK_LIMIT = 40
HIGH_RISK_LIMIT = 70


# ============================================================
# 2. SYNTHETIC DEMO TRAINING DATA
# ============================================================
#
# This data is ONLY for demonstrating the ML workflow.
# It is not real user data.
#
# risk:
# 0 = lower-risk training example
# 1 = higher-risk training example
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

    # Emotion intensity received from A1-2
    "emotion_intensity": [
        2, 3, 4, 8, 9, 7,
        2, 5, 8, 3, 6, 9,
        4, 7, 2, 8, 5, 9
    ],

    # Anomaly score received from A1-4
    "anomaly_score": [
        10, 15, 20, 80, 90, 70,
        5, 40, 85, 15, 60, 95,
        25, 75, 10, 80, 45, 90
    ],

    "mood": [
        "calm",
        "happy",
        "neutral",
        "stressed",
        "anxious",
        "sad",
        "calm",
        "neutral",
        "stressed",
        "happy",
        "anxious",
        "sad",
        "neutral",
        "stressed",
        "calm",
        "anxious",
        "neutral",
        "sad"
    ],

    "situation": [
        "with_family",
        "social",
        "with_family",
        "alone",
        "alone",
        "alone",
        "social",
        "with_family",
        "alone",
        "social",
        "alone",
        "alone",
        "social",
        "alone",
        "with_family",
        "alone",
        "social",
        "alone"
    ],

    "trigger": [
        "none",
        "none",
        "none",
        "stress",
        "stress",
        "emotional",
        "none",
        "none",
        "stress",
        "none",
        "emotional",
        "stress",
        "none",
        "stress",
        "none",
        "emotional",
        "none",
        "stress"
    ],

    "time_of_day": [
        "morning",
        "afternoon",
        "morning",
        "night",
        "night",
        "evening",
        "morning",
        "afternoon",
        "night",
        "morning",
        "evening",
        "night",
        "afternoon",
        "night",
        "morning",
        "evening",
        "afternoon",
        "night"
    ],

    "day_of_week": [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday"
    ],

    "risk": [
        0, 0, 0, 1, 1, 1,
        0, 0, 1, 0, 1, 1,
        0, 1, 0, 1, 0, 1
    ]
}


df = pd.DataFrame(data)


# ============================================================
# 3. FEATURE DEFINITIONS
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


X = df[all_features]
y = df["risk"]


# ============================================================
# 4. PREPROCESSING
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


# ============================================================
# 5. RANDOM FOREST MODEL
# ============================================================

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
# 6. MODEL EVALUATION
# ============================================================

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


# ============================================================
# 7. TRAIN FINAL MODEL USING ALL DEMO DATA
# ============================================================

model.fit(X, y)


# ============================================================
# 8. SAFE INTEGER INPUT FUNCTION
# ============================================================

def get_integer(prompt, minimum, maximum):
    """
    Ask the user for an integer within a specified range.
    """

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


# ============================================================
# 9. SAFE TEXT INPUT FUNCTION
# ============================================================

def get_text(prompt, allowed_values):
    """
    Ask the user for a value from a predefined list.
    """

    while True:

        value = input(prompt).strip().lower()

        if value in allowed_values:
            return value

        print(
            "Please choose one of: "
            + ", ".join(allowed_values)
        )


# ============================================================
# 10. COLLECT USER INPUT
# ============================================================

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

# Information received from A1-2
emotion_intensity = get_integer(
    "Emotion intensity from A1-2 (1-10): ",
    1,
    10
)

# Information received from A1-4
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


# ============================================================
# 11. CREATE USER DATAFRAME
# ============================================================

user_data = pd.DataFrame(
    [{
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
    }]
)


# ============================================================
# 12. PREDICT RISK
# ============================================================

probability = model.predict_proba(
    user_data
)[0][1]


risk_score = round(
    probability * 100
)


# ============================================================
# 13. CONVERT SCORE TO RISK LEVEL
# ============================================================

if risk_score < LOW_RISK_LIMIT:

    risk_level = "LOW"

elif risk_score < HIGH_RISK_LIMIT:

    risk_level = "MEDIUM"

else:

    risk_level = "HIGH"


# ============================================================
# 14. CURRENT VS PREVIOUS TREND
# ============================================================

difference = risk_score - previous_risk


if difference > 10:

    trend = "INCREASING"

elif difference < -10:

    trend = "DECREASING"

else:

    trend = "STABLE"


# ============================================================
# 15. EXPLAINABLE RISK FACTORS
# ============================================================
#
# These are simple human-readable explanations.
# They are NOT medical explanations.
# ============================================================

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


if mood in [
    "stressed",
    "anxious",
    "sad"
]:

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


# ============================================================
# 16. DISPLAY PREDICTION
# ============================================================

print("\n===================================================")
print("                PREDICTION RESULT")
print("===================================================")

print(f"\nRisk Score        : {risk_score}/100")

print(f"Risk Level        : {risk_level}")

print(
    f"Model Probability : {probability:.2f}"
)

print(
    f"Previous Risk     : {previous_risk}/100"
)

print(
    f"Risk Change       : {difference:+d}"
)

print(
    f"Current Trend     : {trend}"
)


# ============================================================
# 17. DISPLAY EXPLANATION
# ============================================================

print("\nWHY THIS PREDICTION?")

print("---------------------------------------")


if risk_factors:

    for number, factor in enumerate(
        risk_factors,
        start=1
    ):

        print(
            f"{number}. {factor}"
        )

else:

    print(
        "No major rule-based risk factors detected."
    )


# ============================================================
# 18. SAVE PREDICTION TO HISTORY
# ============================================================

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


new_record = pd.DataFrame(
    [history_record]
)


# ============================================================
# 19. CREATE OR UPDATE HISTORY FILE
# ============================================================

if os.path.exists(HISTORY_FILE):

    history = pd.read_csv(
        HISTORY_FILE
    )

    history = pd.concat(
        [
            history,
            new_record
        ],
        ignore_index=True
    )

else:

    history = new_record


history.to_csv(
    HISTORY_FILE,
    index=False
)


# ============================================================
# 20. PERSONAL RISK ANALYSIS
# ============================================================

print("\n===================================================")
print("             PERSONAL RISK ANALYSIS")
print("===================================================")


total_predictions = len(history)


average_risk = history[
    "risk_score"
].mean()


highest_risk = history[
    "risk_score"
].max()


lowest_risk = history[
    "risk_score"
].min()


# Last 5 records

recent_history = history.tail(5)


recent_average = recent_history[
    "risk_score"
].mean()


# ============================================================
# 21. RISK DISTRIBUTION
# ============================================================

low_count = (
    history["risk_level"] == "LOW"
).sum()


medium_count = (
    history["risk_level"] == "MEDIUM"
).sum()


high_count = (
    history["risk_level"] == "HIGH"
).sum()


print(
    f"\nTotal predictions : {total_predictions}"
)

print(
    f"Average risk      : {average_risk:.1f}/100"
)

print(
    f"Highest risk      : {highest_risk}/100"
)

print(
    f"Lowest risk       : {lowest_risk}/100"
)

print(
    f"Recent average    : {recent_average:.1f}/100"
)


print("\nRisk Distribution")
print("---------------------------------------")

print(
    f"LOW    : {low_count}"
)

print(
    f"MEDIUM : {medium_count}"
)

print(
    f"HIGH   : {high_count}"
)


# ============================================================
# 22. OVERALL PERSONAL TREND
# ============================================================

if total_predictions >= 2:

    first_risk = history[
        "risk_score"
    ].iloc[0]


    last_risk = history[
        "risk_score"
    ].iloc[-1]


    overall_difference = (
        last_risk - first_risk
    )


    if overall_difference > 10:

        overall_trend = "INCREASING"

    elif overall_difference < -10:

        overall_trend = "DECREASING"

    else:

        overall_trend = "STABLE"

else:

    overall_trend = "NOT ENOUGH DATA"


print(
    f"\nOverall Personal Trend: {overall_trend}"
)


# ============================================================
# 23. RECENT RISK GRAPH
# ============================================================

print("\n===================================================")
print("              RECENT RISK TREND")
print("===================================================")


for index, row in history.tail(10).iterrows():

    score = int(
        row["risk_score"]
    )

    bars = "█" * (
        score // 5
    )

    print(
        f"{index + 1:02d} | "
        f"{bars:<20} "
        f"{score}/100 "
        f"{row['risk_level']}"
    )


# ============================================================
# 24. RECENT PREDICTIONS TABLE
# ============================================================

print("\n===================================================")
print("             RECENT PREDICTIONS")
print("===================================================")


recent_columns = [
    "timestamp",
    "risk_score",
    "risk_level",
    "trend"
]


print(
    history.tail(5)[
        recent_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 25. FINAL STATUS
# ============================================================

print("\n===================================================")

print(
    f"Predictions saved : {total_predictions}"
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