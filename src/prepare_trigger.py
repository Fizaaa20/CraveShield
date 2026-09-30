import pandas as pd
import random
import os

from sklearn.model_selection import train_test_split


random.seed(42)


# ============================================================
# CRAVESHIELD TRIGGER EXAMPLES
# ============================================================

trigger_templates = {

    "Social Pressure": [
        "My friends keep pressuring me to drink.",
        "Everyone around me is telling me to try it.",
        "I feel pressured when my friends encourage me to use substances.",
        "My group keeps asking why I am not joining them.",
        "I am finding it difficult to say no when my friends insist.",
        "People around me keep pushing me to participate.",
        "My friends make fun of me when I refuse.",
        "I feel uncomfortable because everyone expects me to join.",
        "My classmates keep encouraging me to drink.",
        "I am worried my friends will judge me if I say no.",
    ],

    "Academic / Work Stress": [
        "I have an exam tomorrow and I am extremely stressed.",
        "My workload is making me feel overwhelmed.",
        "I have too many assignments to finish.",
        "The pressure of my exams is getting difficult to handle.",
        "My job deadlines are making me anxious.",
        "I have been working continuously and feel exhausted.",
        "My studies are putting a lot of pressure on me.",
        "I am stressed about completing my project on time.",
        "I cannot stop thinking about my upcoming exam.",
        "The pressure at work has become too much.",
    ],

    "Relationship Conflict": [
        "I had a huge argument with my partner.",
        "My relationship problems are making me upset.",
        "My partner and I are constantly fighting.",
        "I cannot stop thinking about the argument with my partner.",
        "Things between me and my partner have become difficult.",
        "I feel terrible after the fight with my girlfriend.",
        "My relationship is causing me a lot of stress.",
        "We had another disagreement and I feel overwhelmed.",
        "I am struggling after my breakup.",
        "The conflict with my partner is affecting me badly.",
    ],

    "Family Conflict": [
        "I had a serious argument with my parents.",
        "My family keeps fighting with me about everything.",
        "Things at home have become very stressful.",
        "I had a conflict with my brother.",
        "My parents are putting too much pressure on me.",
        "I cannot deal with the constant arguments at home.",
        "My family situation is making me upset.",
        "I feel stressed after fighting with my mother.",
        "There is a lot of tension in my family.",
        "My family does not understand what I am going through.",
    ],

    "Loneliness / Isolation": [
        "I feel completely alone tonight.",
        "Nobody is around and I feel isolated.",
        "I have been spending too much time alone.",
        "I wish I had someone to talk to right now.",
        "Everyone seems to have someone except me.",
        "I feel lonely even when I am surrounded by people.",
        "I have been avoiding everyone and staying by myself.",
        "There is nobody I can talk to about this.",
        "I feel disconnected from everyone around me.",
        "Being alone tonight is making things difficult.",
    ],

    "Social Event / Celebration": [
        "There is a party tonight and everyone will be drinking.",
        "I am going to a celebration where people may use substances.",
        "My friend's birthday party is coming up.",
        "There will be alcohol at the event tonight.",
        "I am nervous about attending the party.",
        "Everyone is celebrating and I am worried about staying sober.",
        "I have been invited to a party where I used to drink.",
        "The upcoming celebration reminds me of old habits.",
        "I have to attend a social gathering tonight.",
        "There is a big celebration this weekend.",
    ],

    "Boredom": [
        "I am extremely bored and have nothing to do.",
        "I have been sitting around all day with nothing happening.",
        "There is nothing interesting to keep me occupied.",
        "I feel restless because I have nothing to do.",
        "I am bored at home and cannot think of anything to do.",
        "The day feels empty and repetitive.",
        "I keep looking for something to occupy my time.",
        "I have too much free time today.",
        "I am tired of doing the same thing every day.",
        "Being bored is making me think about old habits.",
    ],

    "Financial Stress": [
        "I am worried because I cannot pay my bills.",
        "Money problems are causing me a lot of stress.",
        "I am struggling financially this month.",
        "I do not know how I will manage my expenses.",
        "My financial situation has become difficult.",
        "I am worried about losing my job and running out of money.",
        "I have too many expenses and not enough money.",
        "Debt is making me feel overwhelmed.",
        "I am stressed about my rent and bills.",
        "I cannot stop worrying about my finances.",
    ],

    "Environmental Cue": [
        "Walking past the old bar reminded me of drinking.",
        "The smell of alcohol immediately brought back memories.",
        "Seeing people smoking triggered an old memory.",
        "I passed the place where I used to drink.",
        "The bottle on the table caught my attention.",
        "The smell of smoke made me think about using again.",
        "I saw my old drinking spot today.",
        "Being in that environment brought back old habits.",
        "Seeing someone use the substance triggered memories.",
        "The familiar smell made me uncomfortable.",
    ],

    "Negative Memory / Past Experience": [
        "I suddenly remembered a difficult time from my past.",
        "Thinking about what happened years ago made me upset.",
        "An old painful memory came back today.",
        "I keep remembering the mistakes I made in the past.",
        "Something reminded me of a bad experience.",
        "I cannot stop thinking about what happened before.",
        "A painful memory from my past came back unexpectedly.",
        "Remembering my old habits made me feel uncomfortable.",
        "I keep replaying a difficult experience in my head.",
        "My past keeps coming back to my thoughts.",
    ],

    "Failure / Disappointment": [
        "I failed my exam and feel terrible.",
        "Things did not work out the way I expected.",
        "I am disappointed with myself.",
        "My project failed and I feel frustrated.",
        "I did not get the result I wanted.",
        "I made a mistake and cannot stop thinking about it.",
        "I feel like a failure after what happened.",
        "I worked hard but still did not succeed.",
        "The rejection was really disappointing.",
        "I am upset because my plans did not work out.",
    ],

    "Other / Unknown": [
        "I am having a difficult day.",
        "Something feels wrong today.",
        "I am struggling to deal with everything.",
        "I do not know what is bothering me.",
        "I feel overwhelmed but cannot explain why.",
        "Today has been really difficult.",
        "I am having a hard time coping right now.",
        "I feel uncomfortable and I do not know the reason.",
        "A lot is going on and I cannot identify the main problem.",
        "I just do not feel like myself today.",
    ],
}


# ============================================================
# CREATE DATASET
# ============================================================

rows = []

for trigger, templates in trigger_templates.items():

    for i in range(150):

        # Pick a base sentence
        text = random.choice(templates)

        # Small variations
        variations = [
            "",
            " Right now.",
            " Today.",
            " Lately.",
            " At the moment.",
            " This has been difficult for me.",
        ]

        text = text + random.choice(variations)

        rows.append({
            "text": text,
            "trigger": trigger
        })


# Shuffle
random.shuffle(rows)

df = pd.DataFrame(rows)


# ============================================================
# SPLIT DATA
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["trigger"]
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["trigger"]
)


# ============================================================
# SAVE DATA
# ============================================================

os.makedirs("../data", exist_ok=True)

train_df.to_csv(
    "../data/trigger_train.csv",
    index=False
)

validation_df.to_csv(
    "../data/trigger_validation.csv",
    index=False
)

test_df.to_csv(
    "../data/trigger_test.csv",
    index=False
)


# ============================================================
# SHOW RESULTS
# ============================================================

print("\nTrigger dataset prepared successfully!")

print("\nTrain:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))
print("Total:", len(df))

print("\nTrigger distribution:")
print(train_df["trigger"].value_counts())

print("\nSample:")
print(train_df.head(10))