import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
} from "react-native";
import Slider from "@react-native-community/slider";

const API_URL = "http://127.0.0.1:8000";

type PredictionResult = {
  status: string;
  risk_score: number;
  risk_level: string;
  model_probability: number;
  personal_risk_trend: string;
  risk_change: number;
  risk_factors: string[];
  ai_analysis: string;
  check_in: {
    craving_level: number;
    stress_level: number;
    previous_craving: number;
    previous_risk: number;
    emotion_intensity: number;
    anomaly_score: number;
    mood: string;
    situation: string;
    trigger: string;
    time_of_day: string;
    day_of_week: string;
  };
};

export default function Index() {
  // -----------------------------
  // USER INPUT STATES
  // -----------------------------

  const [craving, setCraving] = useState(8);
  const [stress, setStress] = useState(7);
  const [previousCraving, setPreviousCraving] = useState(6);
  const [previousRisk, setPreviousRisk] = useState(65);
  const [emotionIntensity, setEmotionIntensity] = useState(7);
  const [anomalyScore, setAnomalyScore] = useState(60);

  const [mood, setMood] = useState("stressed");
  const [situation, setSituation] = useState("alone");
  const [trigger, setTrigger] = useState("stress");
  const [timeOfDay, setTimeOfDay] = useState("evening");
  const [dayOfWeek, setDayOfWeek] = useState("Monday");

  // -----------------------------
  // RESULT STATES
  // -----------------------------

  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  // -----------------------------
  // ANALYZE RISK
  // -----------------------------

  const analyzeRisk = async () => {
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          craving_level: craving,
          stress_level: stress,
          previous_craving: previousCraving,
          previous_risk: previousRisk,
          emotion_intensity: emotionIntensity,
          anomaly_score: anomalyScore,
          mood: mood,
          situation: situation,
          trigger: trigger,
          time_of_day: timeOfDay,
          day_of_week: dayOfWeek,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data: PredictionResult = await response.json();

      setResult(data);
    } catch (error) {
      console.error("Prediction error:", error);

      Alert.alert(
        "Connection Error",
        "Could not connect to the CraveShield AI server. Make sure FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  };

  // -----------------------------
  // CHECK AGAIN
  // -----------------------------

  const checkAgain = () => {
    setResult(null);
  };

  // -----------------------------
  // RESULT SCREEN
  // -----------------------------

  if (result) {
    return (
      <ScrollView
        style={styles.container}
        contentContainerStyle={styles.content}
      >
        <View style={styles.header}>
          <Text style={styles.logo}>CraveShield</Text>
          <Text style={styles.module}>A1-1 Risk Predictor</Text>
        </View>

        <View style={styles.resultCard}>
          <Text style={styles.resultTitle}>YOUR CURRENT RISK</Text>

          <Text style={styles.riskScore}>{result.risk_score}</Text>

          <Text style={styles.outOf}>/100</Text>

          <View style={styles.riskBadge}>
            <Text style={styles.riskBadgeText}>{result.risk_level}</Text>
          </View>

          <Text style={styles.trend}>
            ↑ {result.personal_risk_trend}
          </Text>
        </View>

        {/* AI ANALYSIS */}

        <View style={styles.card}>
          <Text style={styles.cardTitle}>AI ANALYSIS</Text>

          <Text style={styles.analysisText}>
            {result.ai_analysis}
          </Text>
        </View>

        {/* WHY THIS PREDICTION */}

        <View style={styles.card}>
          <Text style={styles.cardTitle}>WHY THIS PREDICTION?</Text>

          {result.risk_factors.map((factor, index) => (
            <View key={index} style={styles.factorRow}>
              <Text style={styles.bullet}>•</Text>
              <Text style={styles.factorText}>{factor}</Text>
            </View>
          ))}
        </View>

        {/* PERSONAL RISK SUMMARY */}

        <View style={styles.card}>
          <Text style={styles.cardTitle}>PERSONAL RISK SUMMARY</Text>

          <DetailRow
            label="Current Risk"
            value={`${result.risk_score}/100`}
          />

          <DetailRow
            label="Previous Risk"
            value={`${result.check_in.previous_risk}/100`}
          />

          <DetailRow
            label="Risk Change"
            value={`+${result.risk_change}`}
          />

          <DetailRow
            label="Model Confidence"
            value={`${result.model_probability}%`}
          />

          <DetailRow
            label="Personal Trend"
            value={result.personal_risk_trend}
          />
        </View>

        {/* YOUR CHECK-IN */}

        <View style={styles.card}>
          <Text style={styles.cardTitle}>YOUR CHECK-IN</Text>

          <DetailRow
            label="Craving"
            value={`${result.check_in.craving_level}/10`}
          />

          <DetailRow
            label="Stress"
            value={`${result.check_in.stress_level}/10`}
          />

          <DetailRow
            label="Previous Craving"
            value={`${result.check_in.previous_craving}/10`}
          />

          <DetailRow
            label="Previous Risk"
            value={`${result.check_in.previous_risk}/100`}
          />

          <DetailRow
            label="Emotion Intensity"
            value={`${result.check_in.emotion_intensity}/10`}
          />

          <DetailRow
            label="Behavior Anomaly"
            value={`${result.check_in.anomaly_score}/100`}
          />

          <DetailRow
            label="Mood"
            value={result.check_in.mood}
          />

          <DetailRow
            label="Situation"
            value={result.check_in.situation}
          />

          <DetailRow
            label="Trigger"
            value={result.check_in.trigger}
          />

          <DetailRow
            label="Time"
            value={result.check_in.time_of_day}
          />
        </View>

        {/* CHECK AGAIN */}

        <TouchableOpacity
          style={styles.primaryButton}
          onPress={checkAgain}
        >
          <Text style={styles.primaryButtonText}>CHECK AGAIN</Text>
        </TouchableOpacity>

        <Text style={styles.disclaimer}>
          This prediction is an AI-based prototype estimate and is not a
          medical or clinical diagnosis.
        </Text>
      </ScrollView>
    );
  }

  // -----------------------------
  // INPUT SCREEN
  // -----------------------------

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
    >
      {/* HEADER */}

      <View style={styles.header}>
        <Text style={styles.logo}>CraveShield</Text>
        <Text style={styles.module}>A1-1 Risk Predictor</Text>
      </View>

      <Text style={styles.mainTitle}>Let's check your state</Text>

      <Text style={styles.subtitle}>
        Enter your current information to calculate your personalized risk.
      </Text>

      {/* CRAVING */}

      <SliderField
        title="How strong is your craving?"
        value={craving}
        minimum={1}
        maximum={10}
        step={1}
        onChange={setCraving}
        display={`${craving}/10`}
      />

      {/* STRESS */}

      <SliderField
        title="Stress Level"
        value={stress}
        minimum={1}
        maximum={10}
        step={1}
        onChange={setStress}
        display={`${stress}/10`}
      />

      {/* PREVIOUS CRAVING */}

      <SliderField
        title="Previous Craving"
        value={previousCraving}
        minimum={1}
        maximum={10}
        step={1}
        onChange={setPreviousCraving}
        display={`${previousCraving}/10`}
      />

      {/* PREVIOUS RISK */}

      <SliderField
        title="Previous Risk"
        value={previousRisk}
        minimum={0}
        maximum={100}
        step={1}
        onChange={setPreviousRisk}
        display={`${previousRisk}/100`}
      />

      {/* EMOTION */}

      <SliderField
        title="Emotion Intensity"
        value={emotionIntensity}
        minimum={1}
        maximum={10}
        step={1}
        onChange={setEmotionIntensity}
        display={`${emotionIntensity}/10`}
      />

      {/* ANOMALY */}

      <SliderField
        title="Behavior Anomaly Score"
        value={anomalyScore}
        minimum={0}
        maximum={100}
        step={1}
        onChange={setAnomalyScore}
        display={`${anomalyScore}/100`}
      />

      {/* MOOD */}

      <SelectionCard
        title="How are you feeling?"
        options={[
          "calm",
          "happy",
          "neutral",
          "stressed",
          "anxious",
          "sad",
        ]}
        selected={mood}
        onSelect={setMood}
      />

      {/* SITUATION */}

      <SelectionCard
        title="Situation"
        options={["alone", "social", "with_family"]}
        selected={situation}
        onSelect={setSituation}
      />

      {/* TRIGGER */}

      <SelectionCard
        title="Trigger"
        options={["none", "stress", "emotional"]}
        selected={trigger}
        onSelect={setTrigger}
      />

      {/* TIME */}

      <SelectionCard
        title="Time of Day"
        options={["morning", "afternoon", "evening", "night"]}
        selected={timeOfDay}
        onSelect={setTimeOfDay}
      />

      {/* DAY */}

      <SelectionCard
        title="Day"
        options={[
          "Monday",
          "Tuesday",
          "Wednesday",
          "Thursday",
          "Friday",
          "Saturday",
          "Sunday",
        ]}
        selected={dayOfWeek}
        onSelect={setDayOfWeek}
      />

      {/* ANALYZE BUTTON */}

      <TouchableOpacity
        style={[
          styles.primaryButton,
          loading && styles.disabledButton,
        ]}
        onPress={analyzeRisk}
        disabled={loading}
      >
        {loading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator color="#FFFFFF" />
            <Text style={styles.primaryButtonText}>
              ANALYZING...
            </Text>
          </View>
        ) : (
          <Text style={styles.primaryButtonText}>
            ANALYZE RISK
          </Text>
        )}
      </TouchableOpacity>

      <Text style={styles.disclaimer}>
        This is an AI-based prototype for personalized risk estimation.
      </Text>
    </ScrollView>
  );
}

// ======================================================
// SLIDER COMPONENT
// ======================================================

type SliderFieldProps = {
  title: string;
  value: number;
  minimum: number;
  maximum: number;
  step: number;
  onChange: (value: number) => void;
  display: string;
};

function SliderField({
  title,
  value,
  minimum,
  maximum,
  step,
  onChange,
  display,
}: SliderFieldProps) {
  return (
    <View style={styles.card}>
      <View style={styles.sliderHeader}>
        <Text style={styles.cardTitle}>{title}</Text>

        <Text style={styles.sliderValue}>{display}</Text>
      </View>

      <Slider
        style={styles.slider}
        minimumValue={minimum}
        maximumValue={maximum}
        step={step}
        value={value}
        onValueChange={onChange}
        minimumTrackTintColor="#2E7D5B"
        maximumTrackTintColor="#CFE8DC"
        thumbTintColor="#2E7D5B"
      />
    </View>
  );
}

// ======================================================
// SELECTION COMPONENT
// ======================================================

type SelectionCardProps = {
  title: string;
  options: string[];
  selected: string;
  onSelect: (value: string) => void;
};

function SelectionCard({
  title,
  options,
  selected,
  onSelect,
}: SelectionCardProps) {
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>{title}</Text>

      <View style={styles.optionsContainer}>
        {options.map((option) => {
          const isSelected = selected === option;

          return (
            <TouchableOpacity
              key={option}
              style={[
                styles.optionButton,
                isSelected && styles.optionButtonSelected,
              ]}
              onPress={() => onSelect(option)}
            >
              <Text
                style={[
                  styles.optionText,
                  isSelected && styles.optionTextSelected,
                ]}
              >
                {formatOption(option)}
              </Text>

              {isSelected && (
                <Text style={styles.checkMark}>✓</Text>
              )}
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
}

// ======================================================
// DETAIL ROW
// ======================================================

function DetailRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text style={styles.detailValue}>{value}</Text>
    </View>
  );
}

// ======================================================
// FORMAT TEXT
// ======================================================

function formatOption(value: string) {
  return value
    .replace("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

// ======================================================
// STYLES
// ======================================================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F6FBF8",
  },

  content: {
    padding: 20,
    paddingBottom: 40,
    maxWidth: 700,
    width: "100%",
    alignSelf: "center",
  },

  header: {
    alignItems: "center",
    marginBottom: 24,
    paddingTop: 10,
  },

  logo: {
    fontSize: 28,
    fontWeight: "800",
    color: "#1F5C43",
  },

  module: {
    fontSize: 15,
    color: "#5C786B",
    marginTop: 4,
  },

  mainTitle: {
    fontSize: 25,
    fontWeight: "700",
    color: "#183D2E",
    marginBottom: 7,
  },

  subtitle: {
    fontSize: 14,
    lineHeight: 21,
    color: "#688276",
    marginBottom: 18,
  },

  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 18,
    padding: 18,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: "#E2F0E8",

    shadowColor: "#1F5C43",
    shadowOffset: {
      width: 0,
      height: 3,
    },
    shadowOpacity: 0.06,
    shadowRadius: 8,

    elevation: 2,
  },

  cardTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: "#214C3A",
  },

  sliderHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },

  sliderValue: {
    fontSize: 15,
    fontWeight: "700",
    color: "#2E7D5B",
  },

  slider: {
    width: "100%",
    height: 40,
    marginTop: 8,
  },

  optionsContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 9,
    marginTop: 13,
  },

  optionButton: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#F1F8F4",
    borderWidth: 1,
    borderColor: "#D6EADF",
    borderRadius: 12,
    paddingVertical: 10,
    paddingHorizontal: 13,
  },

  optionButtonSelected: {
    backgroundColor: "#D9F0E2",
    borderColor: "#2E7D5B",
  },

  optionText: {
    color: "#527264",
    fontSize: 14,
    textTransform: "capitalize",
  },

  optionTextSelected: {
    color: "#205A42",
    fontWeight: "700",
  },

  checkMark: {
    color: "#2E7D5B",
    fontWeight: "800",
    marginLeft: 6,
  },

  primaryButton: {
    backgroundColor: "#2E7D5B",
    borderRadius: 15,
    minHeight: 54,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 6,
    marginBottom: 18,

    shadowColor: "#2E7D5B",
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.18,
    shadowRadius: 8,

    elevation: 3,
  },

  disabledButton: {
    opacity: 0.7,
  },

  primaryButtonText: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "800",
    letterSpacing: 0.5,
  },

  loadingContainer: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },

  resultCard: {
    backgroundColor: "#FFFFFF",
    borderRadius: 22,
    padding: 28,
    alignItems: "center",
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#E0EFE7",

    shadowColor: "#1F5C43",
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.08,
    shadowRadius: 10,

    elevation: 3,
  },

  resultTitle: {
    fontSize: 14,
    fontWeight: "800",
    color: "#658073",
    letterSpacing: 1,
  },

  riskScore: {
    fontSize: 64,
    fontWeight: "800",
    color: "#1F5C43",
    marginTop: 12,
  },

  outOf: {
    fontSize: 16,
    color: "#789086",
    marginTop: -8,
  },

  riskBadge: {
    backgroundColor: "#FBE4E4",
    paddingHorizontal: 22,
    paddingVertical: 9,
    borderRadius: 20,
    marginTop: 15,
  },

  riskBadgeText: {
    color: "#B83A3A",
    fontWeight: "800",
    fontSize: 14,
  },

  trend: {
    color: "#C27628",
    fontSize: 15,
    fontWeight: "700",
    marginTop: 14,
  },

  analysisText: {
    color: "#536E61",
    fontSize: 15,
    lineHeight: 23,
    marginTop: 10,
  },

  factorRow: {
    flexDirection: "row",
    marginTop: 11,
  },

  bullet: {
    color: "#2E7D5B",
    fontSize: 20,
    marginRight: 9,
    lineHeight: 20,
  },

  factorText: {
    flex: 1,
    color: "#536E61",
    fontSize: 14,
    lineHeight: 21,
  },

  detailRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: "#EDF4F0",
  },

  detailLabel: {
    color: "#657D72",
    fontSize: 14,
  },

  detailValue: {
    color: "#234F3D",
    fontSize: 14,
    fontWeight: "700",
    textTransform: "capitalize",
  },

  disclaimer: {
    textAlign: "center",
    color: "#8A9B93",
    fontSize: 12,
    lineHeight: 18,
    marginTop: 3,
    paddingHorizontal: 15,
  },
});