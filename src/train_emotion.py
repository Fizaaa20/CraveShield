import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


# -----------------------------
# Load prepared datasets
# -----------------------------

train_df = pd.read_csv("../data/emotion_train.csv")
val_df = pd.read_csv("../data/emotion_validation.csv")
test_df = pd.read_csv("../data/emotion_test.csv")

print("Datasets loaded successfully!")
print("Training examples:", len(train_df))
print("Validation examples:", len(val_df))
print("Test examples:", len(test_df))


# -----------------------------
# Create TF-IDF + ML pipeline
# -----------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=30000,
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
# Train model
# -----------------------------

print("\nTraining Emotion AI...")

model.fit(
    train_df["text"],
    train_df["emotion"]
)

print("Training completed!")


# -----------------------------
# Validation evaluation
# -----------------------------

val_predictions = model.predict(val_df["text"])

print("\n===== VALIDATION RESULTS =====")
print("Accuracy:", accuracy_score(
    val_df["emotion"],
    val_predictions
))

print(
    classification_report(
        val_df["emotion"],
        val_predictions,
        zero_division=0
    )
)


# -----------------------------
# Final test evaluation
# -----------------------------

test_predictions = model.predict(test_df["text"])

print("\n===== TEST RESULTS =====")
print("Accuracy:", accuracy_score(
    test_df["emotion"],
    test_predictions
))

print(
    classification_report(
        test_df["emotion"],
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
    "../models/emotion_model.joblib"
)

print("\nEmotion model saved successfully!")
print("Location: models/emotion_model.joblib")