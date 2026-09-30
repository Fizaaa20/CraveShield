import pandas as pd
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

AI2_DIR = Path(__file__).resolve().parent.parent
HISTORY_FILE = AI2_DIR / "data" / "pattern_history.csv"


# Minimum number of genuine analyses required
# before showing a personal pattern.
MIN_HISTORY = 3


# ============================================================
# GET PERSONAL PATTERN
# ============================================================

def get_personal_pattern():
    """
    Finds the user's most common Emotion → Trigger pattern
    from previous genuine AI 2 analyses.
    """

    # --------------------------------------------------------
    # No history file
    # --------------------------------------------------------

    if not HISTORY_FILE.exists():

        return {
            "pattern": "Not enough history yet",
            "message": (
                "Start using CraveShield to build "
                "your personal pattern."
            ),
            "count": 0,
            "history_count": 0
        }

    # --------------------------------------------------------
    # Read history
    # --------------------------------------------------------

    try:

        df = pd.read_csv(HISTORY_FILE)

    except Exception:

        return {
            "pattern": "Not enough history yet",
            "message": (
                "Keep using CraveShield so we can learn "
                "your personal patterns."
            ),
            "count": 0,
            "history_count": 0
        }

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "emotion",
            "primary_trigger"
        ]
    )

    history_count = len(df)

    # --------------------------------------------------------
    # Not enough history
    # --------------------------------------------------------

    if history_count < MIN_HISTORY:

        remaining = MIN_HISTORY - history_count

        return {
            "pattern": "Not enough history yet",
            "message": (
                f"Complete at least {remaining} more "
                f"analysis{'s' if remaining != 1 else ''} "
                "to discover your personal pattern."
            ),
            "count": 0,
            "history_count": history_count
        }

    # --------------------------------------------------------
    # Count Emotion–Trigger combinations
    # --------------------------------------------------------

    pattern_counts = (
        df.groupby(
            [
                "primary_trigger",
                "emotion"
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            by="count",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Get strongest pattern
    # --------------------------------------------------------

    top = pattern_counts.iloc[0]

    trigger = str(top["primary_trigger"])
    emotion = str(top["emotion"])
    count = int(top["count"])

    # --------------------------------------------------------
    # Personalized insight
    # --------------------------------------------------------

    message = (
        f"You most often experience "
        f"{emotion.lower()} when dealing with "
        f"{trigger.lower()}."
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "pattern": f"{trigger} → {emotion}",

        "message": message,

        "count": count,

        "history_count": history_count
    }


# ============================================================
# SAVE REAL USER ANALYSIS
# ============================================================

def save_analysis(emotion, primary_trigger):
    """
    Saves one genuine AI 2 analysis to the user's history.
    """

    HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    new_data = pd.DataFrame([
        {
            "emotion": emotion,
            "primary_trigger": primary_trigger
        }
    ])

    if HISTORY_FILE.exists():

        new_data.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        new_data.to_csv(
            HISTORY_FILE,
            mode="w",
            header=True,
            index=False
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = get_personal_pattern()

    print("\n")
    print("=" * 60)
    print("       CRAVESHIELD — PERSONAL PATTERN")
    print("=" * 60)

    print("\nPattern :", result["pattern"])
    print("Insight :", result["message"])

    if result["history_count"] > 0:
        print(
            "History entries:",
            result["history_count"]
        )

    if result["count"] > 0:
        print(
            "Pattern occurrences:",
            result["count"]
        )

    print("\n")
    print("=" * 60)