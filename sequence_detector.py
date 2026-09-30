from datetime import datetime
from collections import Counter


def analyze_behavior_sequence(baseline_events, recent_events):
    """
    Compare the usual order of activities with recent activity order.
    Each event must contain: event_type and timestamp.
    """

    if len(baseline_events) < 2 or len(recent_events) < 2:
        return {
            "status": "insufficient_data",
            "message": "At least 2 events are required in each period."
        }

    def get_sequence(events):
        sorted_events = sorted(
            events,
            key=lambda event: datetime.fromisoformat(
                event["timestamp"]
            )
        )

        return [event["event_type"] for event in sorted_events]

    baseline_sequence = get_sequence(baseline_events)
    recent_sequence = get_sequence(recent_events)

    baseline_transitions = Counter(
        zip(baseline_sequence, baseline_sequence[1:])
    )

    recent_transitions = Counter(
        zip(recent_sequence, recent_sequence[1:])
    )

    baseline_patterns = set(baseline_transitions)
    recent_patterns = set(recent_transitions)

    missing_patterns = sorted(
        [list(pattern) for pattern in baseline_patterns - recent_patterns]
    )

    new_patterns = sorted(
        [list(pattern) for pattern in recent_patterns - baseline_patterns]
    )

    total_patterns = len(baseline_patterns | recent_patterns)

    changed_count = len(missing_patterns) + len(new_patterns)

    drift_score = (
        changed_count / total_patterns
        if total_patterns > 0 else 0
    )

    if drift_score >= 0.6:
        status = "significant_sequence_change"
    elif drift_score >= 0.3:
        status = "possible_sequence_change"
    else:
        status = "stable_sequence"

    return {
        "status": status,
        "baseline_sequence": baseline_sequence,
        "recent_sequence": recent_sequence,
        "missing_patterns": missing_patterns,
        "new_patterns": new_patterns,
        "sequence_drift_score": round(drift_score, 2),
        "message": (
            "The recent activity order differs from the baseline."
            if status != "stable_sequence"
            else "The activity order is broadly similar."
        )
    }