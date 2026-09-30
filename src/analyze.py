from pathlib import Path
import joblib
import pandas as pd

from src.craving_signal import detect_craving_signal
from src.personal_pattern import get_personal_pattern

# ============================================================
# PATH CONFIGURATION
# ============================================================

AI2_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = AI2_DIR / "models"
DATA_DIR = AI2_DIR / "data"

# ============================================================
# LOAD TRAINED MODELS
# ============================================================

EMOTION_MODEL_PATH = MODEL_DIR / "emotion_model.joblib"
TRIGGER_MODEL_PATH = MODEL_DIR / "trigger_model.joblib"

emotion_model = joblib.load(EMOTION_MODEL_PATH)
trigger_model = joblib.load(TRIGGER_MODEL_PATH)


# ============================================================
# EMOTION DETECTION
# ============================================================

def detect_craveshield_emotion(text):

    text_lower = text.lower()

    # --------------------------------------------------------
    # Stress
    # --------------------------------------------------------

    stress_keywords = [
        "stress",
        "stressed",
        "stressful",
        "exam",
        "exams",
        "deadline",
        "assignment",
        "work pressure",
        "academic pressure",
        "overwhelmed",
        "too much work"
    ]

    if any(keyword in text_lower for keyword in stress_keywords):
        return "Stress"

    # --------------------------------------------------------
    # Loneliness
    # --------------------------------------------------------

    loneliness_keywords = [
        "lonely",
        "loneliness",
        "alone",
        "isolated",
        "no one",
        "nobody",
        "by myself",
        "feel left out"
    ]

    if any(keyword in text_lower for keyword in loneliness_keywords):
        return "Loneliness"

    # --------------------------------------------------------
    # Anxiety
    # --------------------------------------------------------

    anxiety_keywords = [
        "anxious",
        "anxiety",
        "worried",
        "worry",
        "nervous",
        "panic",
        "panicking",
        "scared about",
        "afraid"
    ]

    if any(keyword in text_lower for keyword in anxiety_keywords):
        return "Anxiety"

    # --------------------------------------------------------
    # Anger
    # --------------------------------------------------------

    anger_keywords = [
        "angry",
        "anger",
        "furious",
        "mad",
        "hate",
        "irritated"
    ]

    if any(keyword in text_lower for keyword in anger_keywords):
        return "Anger"

    # --------------------------------------------------------
    # Sadness
    # --------------------------------------------------------

    sadness_keywords = [
        "sad",
        "sadness",
        "crying",
        "cry",
        "heartbroken",
        "depressed",
        "hopeless"
    ]

    if any(keyword in text_lower for keyword in sadness_keywords):
        return "Sadness"

    # --------------------------------------------------------
    # Frustration
    # --------------------------------------------------------

    frustration_keywords = [
        "frustrated",
        "frustrating",
        "annoyed",
        "annoying",
        "fed up",
        "disappointed",
        "failure",
        "failed"
    ]

    if any(keyword in text_lower for keyword in frustration_keywords):
        return "Frustration"

    # --------------------------------------------------------
    # Guilt
    # --------------------------------------------------------

    guilt_keywords = [
        "guilty",
        "guilt",
        "regret",
        "regretful",
        "ashamed",
        "ashamed of"
    ]

    if any(keyword in text_lower for keyword in guilt_keywords):
        return "Guilt"

    # --------------------------------------------------------
    # Fear
    # --------------------------------------------------------

    fear_keywords = [
        "fear",
        "afraid",
        "terrified",
        "scared",
        "frightened"
    ]

    if any(keyword in text_lower for keyword in fear_keywords):
        return "Fear"

    # --------------------------------------------------------
    # Fallback to trained ML emotion model
    # --------------------------------------------------------

    prediction = emotion_model.predict([text])[0]

    return prediction


# ============================================================
# EMOTION INTENSITY
# ============================================================

def detect_intensity(text, craving_signal):

    text_lower = text.lower()

    # Strong craving automatically indicates high intensity
    if craving_signal == "Strong":
        return "High"

    high_intensity_words = [
        "extremely",
        "very",
        "really",
        "desperate",
        "terrified",
        "furious",
        "overwhelmed",
        "can't handle",
        "cannot handle",
        "breaking down",
        "out of control",
        "keep pressuring",
        "pressuring me",
        "strong pressure",
        "under pressure"
    ]

    if any(word in text_lower for word in high_intensity_words):
        return "High"

    medium_intensity_words = [
        "quite",
        "pretty",
        "upset",
        "worried",
        "stressed",
        "angry",
        "sad",
        "anxious",
        "frustrated",
        "lonely",
        "pressure"
    ]

    if any(word in text_lower for word in medium_intensity_words):
        return "Medium"

    return "Low"


# ============================================================
# TRIGGER KEYWORDS
# ============================================================

TRIGGER_KEYWORDS = {

    "Social Pressure": [
        "friends keep",
        "friends are",
        "peer pressure",
        "peer",
        "pressuring me",
        "pressure from friends",
        "asked me to",
        "asking me to",
        "friends asking",
        "friends pressuring",
        "group pressure"
    ],

    "Academic / Work Stress": [
        "exam",
        "exams",
        "assignment",
        "college",
        "studying",
        "study pressure",
        "project deadline",
        "deadline",
        "work pressure",
        "workload",
        "office pressure",
        "job pressure"
    ],

    "Relationship Conflict": [
        "argument with my partner",
        "fighting with my partner",
        "argument with my boyfriend",
        "argument with my girlfriend",
        "relationship problem",
        "relationship conflict",
        "breakup",
        "partner conflict"
    ],

    "Family Conflict": [
        "argument with my parents",
        "argument with my family",
        "fighting with my parents",
        "fighting with my family",
        "family argument",
        "family conflict",
        "parents arguing",
        "problem with my parents",
        "conflict with my parents"
    ],

    "Loneliness / Isolation": [
        "completely alone",
        "feel alone",
        "feeling alone",
        "feel lonely",
        "feeling lonely",
        "very lonely",
        "isolated",
        "no one is around",
        "nobody is around",
        "by myself"
    ],

    "Social Event / Celebration": [
        "party",
        "celebration",
        "birthday party",
        "wedding",
        "festival",
        "social event",
        "going out"
    ],

    "Boredom": [
        "i am bored",
        "i'm bored",
        "feeling bored",
        "nothing to do",
        "nothing is happening",
        "bored all day"
    ],

    "Financial Stress": [
        "money problems",
        "financial problems",
        "financial pressure",
        "money pressure",
        "can't pay my bills",
        "cannot pay my bills",
        "rent problem",
        "debt problem",
        "worried about money",
        "worried about bills",
        "money is stressing me"
    ],

    "Environmental Cue": [
        "at the bar",
        "saw someone drinking",
        "saw people drinking",
        "smell alcohol",
        "smelled alcohol",
        "saw a bottle",
        "bottle reminded me",
        "being around alcohol",
        "being around cigarettes",
        "environmental cue"
    ],

    "Negative Memory / Past Experience": [
        "old drinking days",
        "old smoking days",
        "old using days",
        "past drinking",
        "past smoking",
        "past experience",
        "negative memory",
        "bad memory",
        "old memory",
        "reminds me of my past",
        "memory of drinking",
        "memory of smoking"
    ],

    "Failure / Disappointment": [
        "i failed",
        "i have failed",
        "failure",
        "failed my exam",
        "failed the exam",
        "got rejected",
        "rejected from",
        "disappointed about",
        "disappointed with",
        "lost the opportunity",
        "didn't work",
        "did not work"
    ]
}


# ============================================================
# TRIGGER DETECTION
# ============================================================

def detect_triggers(text):

    text_lower = text.lower()

    scores = {}

    for trigger, keywords in TRIGGER_KEYWORDS.items():

        score = 0

        for keyword in keywords:

            if keyword in text_lower:
                score += 1

        if score > 0:
            scores[trigger] = score

    # No trigger detected
    if not scores:
        return "Other / Unknown", "None detected"

    # Sort triggers by score
    sorted_triggers = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    primary_trigger = sorted_triggers[0][0]

    # Find secondary trigger
    secondary_trigger = "None detected"

    for trigger, score in sorted_triggers[1:]:

        if trigger != primary_trigger:
            secondary_trigger = trigger
            break

    return primary_trigger, secondary_trigger


# ============================================================
# SAVE ANALYSIS TO PERSONAL HISTORY
# ============================================================

def save_analysis(emotion, primary_trigger):

    HISTORY_FILE = DATA_DIR / "pattern_history.csv"

    new_row = pd.DataFrame([
        {
            "emotion": emotion,
            "primary_trigger": primary_trigger
        }
    ])

    if HISTORY_FILE.exists():

        new_row.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        new_row.to_csv(
            HISTORY_FILE,
            mode="w",
            header=True,
            index=False
        )


# ============================================================
# MAIN AI 2 ANALYSIS
# ============================================================

def analyze_text(text, save_history=True):

    # --------------------------------------------------------
    # 1. Emotion
    # --------------------------------------------------------

    emotion = detect_craveshield_emotion(text)

    # --------------------------------------------------------
    # 2. Craving Signal
    # --------------------------------------------------------

    craving_signal = detect_craving_signal(text)

    # --------------------------------------------------------
    # 3. Emotion Intensity
    # --------------------------------------------------------

    emotion_intensity = detect_intensity(
        text,
        craving_signal
    )

    # --------------------------------------------------------
    # 4. Trigger
    # --------------------------------------------------------

    primary_trigger, secondary_trigger = detect_triggers(text)

    # --------------------------------------------------------
    # 5. Personal Pattern
    # --------------------------------------------------------

    pattern_result = get_personal_pattern()

    # --------------------------------------------------------
    # 6. Save only when requested
    # --------------------------------------------------------

    if save_history:

        save_analysis(
            emotion,
            primary_trigger
        )

    # --------------------------------------------------------
    # Final structured output
    # --------------------------------------------------------

    return {

        "emotion": emotion,

        "emotion_intensity": emotion_intensity,

        "primary_trigger": primary_trigger,

        "secondary_trigger": secondary_trigger,

        "craving_signal": craving_signal,

        "personal_pattern": pattern_result["pattern"]
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_text = (
        "I have a stressful exam tomorrow and "
        "I really want to drink. "
        "My friends are also asking me to come to a party."
    )

    print("\n")
    print("=" * 65)
    print("          CRAVESHIELD — AI 2 ANALYSIS")
    print("=" * 65)

    result = analyze_text(test_text)

    print("\nInput:")
    print(test_text)

    print("\nAI 2 Analysis:")

    print("Emotion              :", result["emotion"])
    print("Emotion Intensity    :", result["emotion_intensity"])
    print("Primary Trigger      :", result["primary_trigger"])
    print("Secondary Trigger    :", result["secondary_trigger"])
    print("Craving Signal       :", result["craving_signal"])
    print("Personal Pattern     :", result["personal_pattern"])

    print("\n")
    print("=" * 65)
    print("                  ANALYSIS COMPLETE")
    print("=" * 65)