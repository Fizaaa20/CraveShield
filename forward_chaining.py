def forward_chain(facts):
    """
    Forward Chaining rule engine for AI 4.

    The system starts with known facts and applies
    rules to generate conclusions.
    """

    conclusions = []

    # Rule 1
    if (
        facts.get("app_usage_increased", False)
        and facts.get("coping_sessions_decreased", False)
    ):
        conclusions.append({
            "rule": "R1",
            "conclusion": "possible_behavioral_concern",
            "message": "App usage increased while coping activity decreased."
        })

    # Rule 2
    if (
        facts.get("checkins_decreased", False)
        and facts.get("journal_activity_decreased", False)
    ):
        conclusions.append({
            "rule": "R2",
            "conclusion": "reduced_recovery_engagement",
            "message": "Check-ins and journaling activity have decreased."
        })

    # Rule 3
    if facts.get("significant_behavioral_drift", False):
        conclusions.append({
            "rule": "R3",
            "conclusion": "significant_behavioral_change",
            "message": "Significant behavioral drift was detected."
        })

    # Rule 4
    if facts.get("sequence_changed", False):
        conclusions.append({
            "rule": "R4",
            "conclusion": "activity_sequence_change",
            "message": "The recent activity sequence differs from the baseline."
        })

    # Rule 5
    if (
        facts.get("significant_behavioral_drift", False)
        and facts.get("sequence_changed", False)
    ):
        conclusions.append({
            "rule": "R5",
            "conclusion": "high_attention_needed",
            "message": "Behavioral drift and activity-sequence changes were detected together."
        })

    if not conclusions:
        conclusions.append({
            "rule": "NONE",
            "conclusion": "no_rule_triggered",
            "message": "No significant behavioral rule was triggered."
        })

    return {
        "status": "success",
        "algorithm": "Forward Chaining",
        "facts": facts,
        "conclusions": conclusions
    }