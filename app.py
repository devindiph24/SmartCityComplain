import streamlit as st
import joblib

st.set_page_config(page_title="Smart City Complaint Router", page_icon="🏙️", layout="centered")


@st.cache_resource
def load_models():
    category_model = joblib.load("complaint_classifier.joblib")
    toxicity_model = joblib.load("toxic_classifier.joblib")
    return category_model, toxicity_model


category_model, toxicity_model = load_models()

st.title("🏙️ Smart City Complaint Router")
st.write(
    "Submit a citizen complaint below. The system automatically **routes it "
    "to the right city department** and **flags abusive submissions** for "
    "moderator review, the way a real municipal 311/service-request system would."
)

message = st.text_area(
    "Describe the issue",
    height=140,
    placeholder="e.g. There's a big pothole on Elm Street that's been damaging cars for a week.",
)

col1, col2 = st.columns([1, 3])
with col1:
    submit_clicked = st.button("Submit complaint", type="primary", use_container_width=True)

if submit_clicked:
    if not message.strip():
        st.warning("Please describe the issue first.")
    else:
        # Stage 1: toxicity check
        tox_pred = toxicity_model.predict([message])[0]
        tox_proba = toxicity_model.predict_proba([message])[0]
        tox_classes = list(toxicity_model.classes_)
        tox_confidence = tox_proba[tox_classes.index(tox_pred)]

        if tox_pred == "Toxic":
            st.error(
                f"🚨 This submission was flagged as **abusive/toxic** "
                f"(confidence: {tox_confidence:.1%}) and has been routed to a "
                f"moderator instead of a department."
            )
        else:
            # Stage 2: category routing (only shown for non-toxic submissions)
            cat_pred = category_model.predict([message])[0]
            cat_proba = category_model.predict_proba([message])[0]
            cat_classes = list(category_model.classes_)
            cat_confidence = cat_proba[cat_classes.index(cat_pred)]

            st.success(f"✅ Routed to: **{cat_pred}** (confidence: {cat_confidence:.1%})")

            with st.expander("See confidence for all departments"):
                for cls, p in sorted(zip(cat_classes, cat_proba), key=lambda x: -x[1]):
                    st.write(f"{cls}: {p:.1%}")

st.divider()
with st.expander("About this app"):
    st.write(
        "**Two-stage pipeline:**\n"
        "1. A toxicity classifier (TF-IDF + Logistic Regression, trained on the "
        "Surge AI toxicity dataset) screens for abusive submissions.\n"
        "2. Non-toxic complaints are routed to one of 6 departments using a "
        "multi-class TF-IDF + Logistic Regression model — Traffic & Roads, "
        "Sanitation & Waste, Water & Utilities, Public Safety, Noise Complaints, "
        "Parks & Public Spaces.\n\n"
        "The routing model was evaluated with entire sentence templates held out "
        "of training (not just individual rows), to avoid the model simply "
        "memorizing phrasing. On genuinely unseen phrasing it reaches **~61% "
        "accuracy** across 6 balanced classes (vs. ~17% random chance). "
        "See the README for details, including a genuine confusion the model "
        "learned between fire-hydrant-related Public Safety and Water & "
        "Utilities complaints."
    )
