from flask import Flask, request, jsonify
from flask_cors import CORS

from drift_detector import detect_behavioral_drift
from sequence_detector import analyze_behavior_sequence

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "module": "CraveShield AI 4",
        "name": "Behavioral Drift Detector",
        "status": "running",
    })


@app.route("/detect-drift", methods=["POST"])
def detect_drift():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON data is required"}), 400

    baseline_days = data.get("baseline_days", [])
    recent_days = data.get("recent_days", [])

    if not isinstance(baseline_days, list) or not isinstance(
        recent_days, list
    ):
        return jsonify({
            "error": "baseline_days and recent_days must be lists"
        }), 400

    try:
        result = detect_behavioral_drift(
            baseline_days,
            recent_days,
        )
        return jsonify(result)

    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.route("/analyze-sequence", methods=["POST"])
def analyze_sequence():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON data is required"}), 400

    baseline_events = data.get("baseline_events", [])
    recent_events = data.get("recent_events", [])

    if not isinstance(baseline_events, list) or not isinstance(
        recent_events, list
    ):
        return jsonify({
            "error": "baseline_events and recent_events must be lists"
        }), 400

    try:
        result = analyze_behavior_sequence(
            baseline_events,
            recent_events,
        )
        return jsonify(result)

    except (TypeError, ValueError, KeyError) as error:
        return jsonify({"error": str(error)}), 400

@app.route("/analyze-all", methods=["POST"])
def analyze_all():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON data is required"}), 400

    baseline_days = data.get("baseline_days", [])
    recent_days = data.get("recent_days", [])
    baseline_events = data.get("baseline_events", [])
    recent_events = data.get("recent_events", [])

    try:
        drift_result = detect_behavioral_drift(
            baseline_days,
            recent_days,
        )

        sequence_result = analyze_behavior_sequence(
            baseline_events,
            recent_events,
        )

        return jsonify({
            "module": "CraveShield AI 4",
            "behavioral_drift": drift_result,
            "sequence_analysis": sequence_result,
        })

    except (TypeError, ValueError, KeyError) as error:
        return jsonify({"error": str(error)}), 400
if __name__ == "__main__":
    app.run(debug=True, port=5000)