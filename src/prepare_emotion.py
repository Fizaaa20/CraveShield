from datasets import load_dataset
import pandas as pd
import os

# Load GoEmotions
dataset = load_dataset("google-research-datasets/go_emotions")

# GoEmotions → CraveShield mapping
emotion_map = {
    "anger": "Anger",
    "fear": "Fear",
    "nervousness": "Anxiety",
    "sadness": "Sadness",
    "grief": "Sadness",
    "remorse": "Guilt",
    "disappointment": "Frustration",
    "annoyance": "Frustration",
    "neutral": "Neutral / Positive",
    "joy": "Neutral / Positive",
    "optimism": "Neutral / Positive",
    "excitement": "Neutral / Positive",
    "relief": "Neutral / Positive"
}

# Get original label names
label_names = dataset["train"].features["labels"].feature.names

def convert_example(example):
    mapped_emotions = []

    for label_id in example["labels"]:
        original_label = label_names[label_id]

        if original_label in emotion_map:
            mapped_emotions.append(emotion_map[original_label])

    if not mapped_emotions:
        return None

    return mapped_emotions[0]


def prepare_split(split_name):
    rows = []

    for example in dataset[split_name]:
        emotion = convert_example(example)

        if emotion is not None:
            rows.append({
                "text": example["text"],
                "emotion": emotion
            })

    return pd.DataFrame(rows)


# Prepare datasets
train_df = prepare_split("train")
val_df = prepare_split("validation")
test_df = prepare_split("test")

# Create data folder
os.makedirs("../data", exist_ok=True)

# Save files
train_df.to_csv("../data/emotion_train.csv", index=False)
val_df.to_csv("../data/emotion_validation.csv", index=False)
test_df.to_csv("../data/emotion_test.csv", index=False)

print("\nEmotion dataset prepared successfully!")

print("\nTrain:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nEmotion distribution:")
print(train_df["emotion"].value_counts())

print("\nSample:")
print(train_df.head(10))