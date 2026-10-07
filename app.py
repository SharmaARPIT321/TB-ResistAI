import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TB-ResistAI",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# 3. FINAL MODEL CONFIGURATION
# ============================================================

FINAL_CONFIG = {
    "Rifampicin": {
        "folder": "rifampicin",
        "model_name": "Random Forest",
        "feature_count": 30,
        "threshold": 0.72,
    },

    "Isoniazid": {
        "folder": "isoniazid",
        "model_name": "Logistic Regression",
        "feature_count": 2,
        "threshold": 0.50,
    },

    "Ethambutol": {
        "folder": "ethambutol",
        "model_name": "Random Forest",
        "feature_count": 8,
        "threshold": 0.32,
    },
}


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(drug):

    config = FINAL_CONFIG[drug]

    model_path = (
        MODEL_DIR
        / config["folder"]
        / "model.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    return joblib.load(model_path)


# ============================================================
# 5. LOAD FEATURE MANIFEST
# ============================================================

@st.cache_data
def load_features(drug):

    config = FINAL_CONFIG[drug]

    feature_path = (
        MODEL_DIR
        / config["folder"]
        / "features.json"
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature manifest not found:\n{feature_path}"
        )

    with open(feature_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support several possible JSON formats
    if isinstance(data, list):
        features = data

    elif isinstance(data, dict):

        if "features" in data:
            features = data["features"]

        elif "feature_names" in data:
            features = data["feature_names"]

        else:
            raise ValueError(
                f"Could not find feature list in {feature_path}"
            )

    else:
        raise ValueError(
            f"Unsupported feature JSON format: {feature_path}"
        )

    return [str(x) for x in features]


# ============================================================
# 6. GET RESISTANCE PROBABILITY
# ============================================================

def get_resistance_probability(model, X):

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The saved model does not support predict_proba()."
        )

    probabilities = model.predict_proba(X)[0]

    classes = getattr(model, "classes_", None)

    # Most of your binary models should use 0/1
    if classes is not None:

        # Find class corresponding to resistant = 1
        if 1 in classes:
            resistant_index = list(classes).index(1)
            return float(probabilities[resistant_index])

        # Handle string class labels
        normalized = [str(c).strip().lower() for c in classes]

        resistant_labels = {
            "resistant",
            "r",
            "1"
        }

        for i, label in enumerate(normalized):

            if label in resistant_labels:
                return float(probabilities[i])

    # Fallback for standard binary classifier
    return float(probabilities[-1])


# ============================================================
# 7. CREATE MODEL INPUT
# ============================================================

def create_feature_vector(selected_mutations, features):

    vector = []

    selected_set = set(selected_mutations)

    for feature in features:

        if feature in selected_set:
            vector.append(1)
        else:
            vector.append(0)

    return np.array(vector).reshape(1, -1)


# ============================================================
# 8. PAGE HEADER
# ============================================================

st.title("🧬 TB-ResistAI")

st.subheader(
    "Genomic Mutation-Based Tuberculosis Drug-Resistance Prediction"
)

st.write(
    """
    Enter the detected mutation profile to evaluate predicted
    resistance across the evaluated first-line TB drugs.
    """
)

st.info(
    """
    Research prototype only. Predictions are not a substitute
    for laboratory drug-susceptibility testing or clinical
    decision-making.
    """
)


# ============================================================
# 9. LOAD ALL FEATURE LISTS
# ============================================================

try:

    rif_features = load_features("Rifampicin")
    inh_features = load_features("Isoniazid")
    emb_features = load_features("Ethambutol")

except Exception as e:

    st.error(str(e))

    st.stop()


# ============================================================
# 10. COMBINE FEATURES
# ============================================================

all_features = sorted(
    set(
        rif_features
        + inh_features
        + emb_features
    )
)


# ============================================================
# 11. MUTATION INPUT
# ============================================================

st.header("🧬 Mutation Input")

st.write(
    f"""
    Select the mutations detected in the sample.

    The application will automatically determine which
    mutations belong to the feature set of each drug model.
    """
)

selected_mutations = st.multiselect(
    "Select detected mutations",
    options=all_features,
    help="Select all mutations detected in the sample."
)


# ============================================================
# 12. SHOW SELECTED MUTATIONS
# ============================================================

if selected_mutations:

    st.success(
        f"{len(selected_mutations)} mutation(s) selected."
    )

    with st.expander("View selected mutations"):

        for mutation in selected_mutations:
            st.write(f"✓ {mutation}")

else:

    st.warning(
        "Please select at least one mutation before analysis."
    )


# ============================================================
# 13. MODEL INFORMATION
# ============================================================

st.header("🤖 Final Model Configuration")

model_table = pd.DataFrame({

    "Drug": [
        "Rifampicin",
        "Isoniazid",
        "Ethambutol"
    ],

    "Final Model": [
        "Random Forest",
        "Logistic Regression",
        "Random Forest"
    ],

    "Features": [
        30,
        2,
        8
    ],

    "Decision Threshold": [
        0.72,
        0.50,
        0.32
    ]
})

st.dataframe(
    model_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 14. ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔬 Analyze Mutation Profile",
    type="primary",
    use_container_width=True
)


# ============================================================
# 15. RUN PREDICTIONS
# ============================================================

if analyze:

    if not selected_mutations:

        st.error(
            "Please select at least one mutation."
        )

        st.stop()

    results = []

    drugs = [
        "Rifampicin",
        "Isoniazid",
        "Ethambutol"
    ]

    for drug in drugs:

        try:

            config = FINAL_CONFIG[drug]

            model = load_model(drug)

            features = load_features(drug)

            X = create_feature_vector(
                selected_mutations,
                features
            )

            # Verify feature count
            if X.shape[1] != len(features):

                raise ValueError(
                    f"{drug}: feature vector mismatch."
                )

            probability = get_resistance_probability(
                model,
                X
            )

            threshold = config["threshold"]

            prediction = (
                "Resistant"
                if probability >= threshold
                else "Predicted susceptible"
            )

            results.append({

                "Drug": drug,

                "Model": config["model_name"],

                "Resistance Probability":
                    probability,

                "Threshold":
                    threshold,

                "Prediction":
                    prediction,

            })

        except Exception as e:

            st.error(
                f"{drug}: {str(e)}"
            )

    # ========================================================
    # 16. DISPLAY RESULTS
    # ========================================================

    if results:

        result_df = pd.DataFrame(results)

        st.header("📊 Prediction Summary")

        display_df = result_df.copy()

        display_df[
            "Resistance Probability"
        ] = (
            display_df[
                "Resistance Probability"
            ] * 100
        ).round(2).astype(str) + "%"

        display_df[
            "Threshold"
        ] = (
            display_df["Threshold"] * 100
        ).round(0).astype(int).astype(str) + "%"

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # 17. PROBABILITY VISUALIZATION
        # ====================================================

        st.subheader(
            "Resistance Probability Comparison"
        )

        chart_df = result_df[
            ["Drug", "Resistance Probability"]
        ].copy()

        chart_df = chart_df.set_index("Drug")

        chart_df[
            "Resistance Probability"
        ] *= 100

        st.bar_chart(
            chart_df
        )

        # ====================================================
        # 18. INTERPRETATION
        # ====================================================

        lowest = result_df.loc[
            result_df["Resistance Probability"].idxmin()
        ]

        st.subheader(
            "🔎 Model Interpretation"
        )

        st.write(
            f"""
            Among the evaluated drugs, the model predicts the
            lowest resistance probability for **{lowest["Drug"]}**
            ({lowest["Resistance Probability"] * 100:.1f}%).
            """
        )

        st.warning(
            """
            A lower model-predicted resistance probability does
            not constitute a recommendation to use that drug.
            Clinical treatment decisions require laboratory
            susceptibility testing and professional assessment.
            """
        )


# ============================================================
# 19. MODEL VALIDATION
# ============================================================

st.header("📈 Model Validation")

st.write(
    """
    The models were selected based on their performance during
    model evaluation and external/generalization analysis.
    """
)

validation_table = pd.DataFrame({

    "Drug": [
        "Rifampicin",
        "Isoniazid",
        "Ethambutol"
    ],

    "Selected Model": [
        "Random Forest",
        "Logistic Regression",
        "Random Forest"
    ],

    "Feature Count": [
        30,
        2,
        8
    ],

    "Threshold": [
        0.72,
        0.50,
        0.32
    ],

})

st.dataframe(
    validation_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 20. EXPLANATION OF METRICS
# ============================================================

with st.expander("What do the evaluation metrics mean?"):

    st.markdown(
        """
        **ROC-AUC** measures the model's ability to distinguish
        between resistant and susceptible samples across
        classification thresholds.

        **F1-score** balances precision and recall.

        **Sensitivity** measures the ability to identify resistant
        samples.

        **Specificity** measures the ability to identify susceptible
        samples.

        **External validation** evaluates how well the model
        generalizes to data that was not used during model
        development.

        These metrics describe model performance. They are not
        themselves individual-patient treatment recommendations.
        """
    )


# ============================================================
# 21. RESEARCH DISCLAIMER
# ============================================================

st.divider()

st.caption(
    """
    TB-ResistAI is a research prototype for exploring
    mutation-based tuberculosis drug-resistance prediction.
    It is not intended for clinical diagnosis, prescription,
    or treatment selection.
    """
)
