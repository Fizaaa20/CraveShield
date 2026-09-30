from analyze import analyze_text


test_cases = [

    # 1. Loneliness
    "I am feeling very lonely tonight. Everyone seems busy and I am completely alone.",

    # 2. Family conflict
    "I had a big argument with my parents today and I feel extremely angry and upset.",

    # 3. Financial stress
    "I am worried about money and my bills. The financial pressure is making me anxious.",

    # 4. Social pressure
    "My friends are going to a party and they keep pressuring me to drink with them.",

    # 5. Past memory + craving
    "This place reminds me of my old drinking days and I really want to drink again.",

]


print("\n")
print("=" * 65)
print("          CRAVESHIELD — AI 2 STRESS TEST")
print("=" * 65)


for i, text in enumerate(test_cases, start=1):

    result = analyze_text(text, save_history=False)

    print("\n")
    print("-" * 65)
    print(f"TEST CASE {i}")
    print("-" * 65)

    print("Input:")
    print(text)

    print("\nAI 2 Analysis:")
    print("Emotion              :", result["emotion"])
    print("Emotion Intensity    :", result["emotion_intensity"])
    print("Primary Trigger      :", result["primary_trigger"])
    print("Secondary Trigger    :", result["secondary_trigger"])
    print("Craving Signal       :", result["craving_signal"])
    print("Personal Pattern     :", result["personal_pattern"])


print("\n")
print("=" * 65)
print("                  TEST COMPLETE")
print("=" * 65)