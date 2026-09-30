import re


# ============================================================
# STRONG CRAVING SIGNALS
# ============================================================

STRONG_PATTERNS = [

    r"\bwant to (drink|smoke|use|take)\b",

    r"\breally want\b",

    r"\bstrong urge\b",

    r"\bstrong craving\b",

    r"\bcraving (a|for|to)\b",

    r"\bneed (a|to)\b",

    r"\bi can't stop thinking about (drinking|smoking|using)\b",

    r"\bi feel like (drinking|smoking|using)\b",

    r"\bi want (a drink|to drink|to smoke|to use)\b",

    r"\bdesperate to (drink|smoke|use)\b",

    r"\breally feel like (drinking|smoking|using)\b",

    r"\bwant to drink again\b",

    r"\bwant to smoke again\b",

    r"\bwant to use again\b",
]


# ============================================================
# MODERATE CRAVING SIGNALS
# ============================================================

MODERATE_PATTERNS = [

    r"\bthinking about drinking\b",

    r"\bthinking about smoking\b",

    r"\bthinking about using\b",

    r"\bthought about drinking\b",

    r"\bthought about smoking\b",

    r"\bthinking of drinking\b",

    r"\bthinking of smoking\b",

    r"\btempted to drink\b",

    r"\btempted to smoke\b",

    r"\btempted to use\b",

    r"\btemptation\b",

    r"\bdesire to drink\b",

    r"\bdesire to smoke\b",

    r"\bdesire to use\b",

    r"\bmiss drinking\b",

    r"\bmiss smoking\b",

    r"\bmiss using\b",

    r"\bkeep thinking about drinking\b",

    r"\bkeep thinking about smoking\b",
]


# ============================================================
# POSSIBLE CRAVING SIGNALS
# ============================================================

POSSIBLE_PATTERNS = [

    r"\bsometimes i think about\b",

    r"\bpart of me wants\b",

    r"\bthinking about my old habit\b",

    r"\bremember drinking\b",

    r"\bremember smoking\b",

    r"\breminds me of drinking\b",

    r"\breminds me of smoking\b",

    r"\bold habits\b",

    r"\bused to drink\b",

    r"\bused to smoke\b",

    r"\bused to use\b",

    r"\bpressure.*\bdrink\b",

    r"\bpressure.*\bsmoke\b",

    r"\bpressuring me to drink\b",

    r"\bpressuring me to smoke\b",

    r"\basking me to drink\b",

    r"\basking me to smoke\b",
]


# ============================================================
# CRAVING SIGNAL DETECTOR
# ============================================================

def detect_craving_signal(text):

    text_lower = text.lower()

    # Strong signal
    for pattern in STRONG_PATTERNS:
        if re.search(pattern, text_lower):
            return "Strong"

    # Moderate signal
    for pattern in MODERATE_PATTERNS:
        if re.search(pattern, text_lower):
            return "Moderate"

    # Possible signal
    for pattern in POSSIBLE_PATTERNS:
        if re.search(pattern, text_lower):
            return "Possible"

    return "None"


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_cases = [

        "I had a stressful exam today.",

        "Sometimes I think about drinking when I am stressed.",

        "I have been thinking about drinking all evening.",

        "I really want to drink right now.",

        "My friends keep pressuring me to drink with them.",

        "My friends are asking me to drink at the party.",

        "This place reminds me of my old drinking days."
    ]


    print("\n")
    print("==========================================")
    print("       CRAVING SIGNAL TEST")
    print("==========================================")


    for text in test_cases:

        result = detect_craving_signal(text)

        print("\nText:", text)
        print("Craving Signal:", result)


    print("\n==========================================")