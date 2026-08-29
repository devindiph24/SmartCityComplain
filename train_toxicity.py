"""
Retrain the toxicity classifier, augmented with our complaint dataset as
extra 'Not Toxic' examples.

WHY: the original toxicity model was trained only on the Surge AI toxicity
dataset (general online comments). When applied to civic-complaint text, it
produced false positives on legitimate but negatively-worded complaints
(e.g. "...it smells terrible", "...dangerous for kids") because those
words correlated with toxicity in its original (out-of-domain) training
distribution. Since every complaint in data/complaints.csv is, by
construction, a legitimate non-toxic complaint, adding them as extra
"Not Toxic" training examples is a cheap and effective domain-adaptation
fix: it teaches the model what non-toxic *complaint-style* negativity
looks like, without needing any new toxic-labeled data.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# 1. Load and combine data
tox_df = pd.read_csv("data/toxicity_en.csv")
tox_df["is_toxic"] = tox_df["is_toxic"].str.strip()
tox_df = tox_df.rename(columns={"text": "text"})[["text", "is_toxic"]]

complaints_df = pd.read_csv("data/complaints.csv")
complaints_df["is_toxic"] = "Not Toxic"
complaints_df = complaints_df[["text", "is_toxic"]]

combined = pd.concat([tox_df, complaints_df], ignore_index=True)
print(f"Combined dataset: {len(combined)} rows")
print(combined["is_toxic"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    combined["text"], combined["is_toxic"], test_size=0.2, random_state=42, stratify=combined["is_toxic"]
)

# 2. Build & train pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=3)),
])
pipeline.fit(X_train, y_train)

# 3. Evaluate on the held-out mixed test set
y_pred = pipeline.predict(X_test)
print(f"\nOverall test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred))

# 4. Specifically re-check the complaint-domain false positives that motivated this fix
print("\n--- Sanity check on previously-misclassified complaint text ---")
sanity_checks = [
    ("The garbage has not been collected on Baker Street for a week, it smells terrible.", "Not Toxic"),
    ("The playground swings at the park are rusted and dangerous for kids.", "Not Toxic"),
    ("You people are useless idiots, fix the streetlight now!", "Toxic"),
    ("There is a huge pothole on Baker Street damaging cars every day.", "Not Toxic"),
]
for text, expected in sanity_checks:
    pred = pipeline.predict([text])[0]
    status = "OK" if pred == expected else "MISS"
    print(f"{status:4s} pred={pred:10s} expected={expected:10s} | {text[:55]}")

# 5. Save
joblib.dump(pipeline, "model/toxic_classifier.joblib")
print("\nSaved augmented model to model/toxic_classifier.joblib")
