# CraveShield - A1-1 Personalized Craving Risk Predictor

## Overview

A1-1 is the Personalized Craving Risk Predictor module of CraveShield.

The purpose of this module is to estimate the user's current craving-risk level by analyzing behavioral, emotional, historical, and contextual inputs.

The module uses a Random Forest machine learning model and produces a risk score, risk level, risk trend, and understandable factors that contributed to the prediction.

## Objective

The objective of A1-1 is to answer:

> "Based on the user's current condition and previous patterns, what is the predicted craving-risk level, and why?"

## Machine Learning Model

The current implementation uses:

- Random Forest Classifier
- Pandas
- Scikit-learn

The model converts the prediction probability into a prototype risk score from 0 to 100.

## Input Features

The model uses the following information:

### Current User Inputs

- Current craving level (1-10)
- Stress level (1-10)
- Previous craving level (1-10)
- Previous risk score (0-100)
- Mood
- Situation
- Trigger
- Time of day
- Day of week

### Future Integrated Application Inputs

The final CraveShield application is intended to collect real user-provided information through the application.

The A1-1 module can receive relevant outputs from other modules when the complete system is integrated.

For the current independent A1-1 implementation, these modules are not directly connected.

## Output

A1-1 generates:

- Risk score (0-100)
- Risk level
- Model probability
- Current risk trend
- Risk-factor explanation
- Prediction history
- Average risk
- Highest risk
- Lowest risk
- Recent average risk
- Recent prediction records

## Risk Levels

The current prototype uses the following thresholds:

- 0-39: LOW
- 40-69: MEDIUM
- 70-100: HIGH

These thresholds are prototype conventions and are not clinical or medical standards.

## Risk Trend

The module compares the current predicted risk with the previous risk score.

Possible trends are:

- INCREASING
- DECREASING
- STABLE

The overall personal trend is also calculated from saved prediction history when sufficient records are available.

## Explainable Risk Factors

The system provides simple explanations for factors associated with the prediction, such as:

- High current craving
- High stress
- High previous craving
- High previous risk
- High emotional intensity
- Unusual behavioral pattern
- Challenging emotional state
- Reported trigger
- Being alone

These explanations are rule-based supporting information and should not be interpreted as medical conclusions.

## Prediction History

Each prediction can be saved locally with:

- Timestamp
- User input values
- Risk score
- Risk level
- Risk trend

The history is stored in:

```text
risk_history.csv