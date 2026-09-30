
import numpy as np

FEATURES = [
    "journal_count",
    "app_open_count",
    "coping_sessions",
    "checkin_completion",
    "sleep_time_shift",
]


def detect_behavioral_drift(baseline_days, recent_days):
    if len(baseline_days) < 7:
        return {
            "status": "insufficient_data",
            "message": "At least 7 baseline days are required.",
        }

    if len(recent_days) < 2:
        return {
            "status": "insufficient_data",
            "message": "At least 2 recent days are required.",
        }

    baseline = {}
    recent = {}
    changes = []

    for feature in FEATURES:
        old_values = np.array(
            [day.get(feature, 0) for day in baseline_days],
            dtype=float,
        )
        new_values = np.array(
            [day.get(feature, 0) for day in recent_days],
            dtype=float,
        )

        old_median = float(np.median(old_values))
        new_median = float(np.median(new_values))

        mad = float(np.median(np.abs(old_values - old_median)))
        scale = max(1.4826 * mad, 1.0)

        deviation = abs(new_median - old_median) / scale
        difference = new_median - old_median

        baseline[feature] = round(old_median, 2)
        recent[feature] = round(new_median, 2)

        changes.append({
            "feature": feature,
            "baseline": round(old_median, 2),
            "recent": round(new_median, 2),
            "difference": round(difference, 2),
            "direction": (
                "increased" if difference > 0
                else "decreased" if difference < 0
                else "unchanged"
            ),
            "deviation": round(deviation, 2),
        })

    drift_score = float(
        np.mean([min(item["deviation"], 5) for item in changes])
    )

    changed_features = [
        item for item in changes
        if item["deviation"] >= 2
    ]

    if drift_score >= 2 and len(changed_features) >= 2:
        status = "significant_drift"
    elif drift_score >= 1:
        status = "possible_drift"
    else:
        status = "stable"

    return {
        "status": status,
        "drift_score": round(drift_score, 2),
        "changed_feature_count": len(changed_features),
        "baseline": baseline,
        "recent": recent,
        "changes": changes,
        "message": (
            "Multiple behavioral changes detected."
            if status == "significant_drift"
            else "Some behavioral changes detected."
            if status == "possible_drift"
            else "No strong behavioral drift detected."
        ),
    }