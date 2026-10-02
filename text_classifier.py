from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# -----------------------------------
# Demo Training Data
# -----------------------------------

TRAINING_TEXTS = [
    "I feel calm and positive today",
    "Today was a good and normal day",
    "I feel happy and relaxed",
    "Everything feels fine today",
    "I am feeling peaceful",

    "I feel stressed and overwhelmed",
    "Today was very difficult",
    "I am feeling anxious and worried",
    "I feel angry and frustrated",
    "I am struggling with strong cravings",
]

TRAINING_LABELS = [
    "normal",
    "normal",
    "normal",
    "normal",
    "normal",

    "stress",
    "stress",
    "stress",
    "stress",
    "stress",
]


# -----------------------------------
# Train TF-IDF + Naive Bayes
# -----------------------------------

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(TRAINING_TEXTS)

model = MultinomialNB()

model.fit(X_train, TRAINING_LABELS)


# -----------------------------------
# Classify New Text
# -----------------------------------

def classify_text(text):
    """
    Classifies journal/check-in text
    using TF-IDF and Naive Bayes.
    """

    if not text or not text.strip():
        return {
            "status": "error",
            "message": "Text cannot be empty."
        }

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    probabilities = model.predict_proba(text_vector)[0]

    confidence = float(max(probabilities))

    return {
        "status": "success",
        "algorithm": "TF-IDF + Naive Bayes",
        "text": text,
        "classification": prediction,
        "confidence": confidence
    }