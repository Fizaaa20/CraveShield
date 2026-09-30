import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


# -----------------------------
# Load datasets
# -----------------------------

train_df = pd.read_csv("../data/trigger_train.csv")
val_df = pd.read_csv("../data/trigger_validation.csv")
test_df = pd.read_csv("../data/trigger_test.csv")

print("Trigger datasets loaded successfully!")

print("Training examples:", len(train_df))
print("Validation examples:", len(val_df))
print("Test examples:", len(test_df))


# -----------------------------
# Create ML pipeline
# -----------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# -----------------------------
# Train
# -----------------------------

print("\nTraining Trigger AI...")

model.fit(
    train_df["text"],
    train_df["trigger"]
)

print("Training completed!")


# -----------------------------
# Validation
# -----------------------------

val_predictions = model.predict(val_df["text"])

print("\n===== VALIDATION RESULTS =====")

print(
    "Accuracy:",
    accuracy_score(
        val_df["trigger"],
        val_predictions
    )
)

print(
    classification_report(
        val_df["trigger"],
        val_predictions,
        zero_division=0
    )
)


# -----------------------------
# Test
# -----------------------------

test_predictions = model.predict(test_df["text"])

print("\n===== TEST RESULTS =====")

print(
    "Accuracy:",
    accuracy_score(
        test_df["trigger"],
        test_predictions
    )
)

print(
    classification_report(
        test_df["trigger"],
        test_predictions,
        zero_division=0
    )
)


# -----------------------------
# Save model
# -----------------------------

os.makedirs("../models", exist_ok=True)

joblib.dump(
    model,
    "../models/trigger_model.joblib"
)

print("\nTrigger model saved successfully!")

print(
    "Location: models/trigger_model.joblib"
)