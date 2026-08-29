"""
Train a TF-IDF + Logistic Regression multi-class classifier that routes
a citizen complaint to the correct city department/category, and save
the trained pipeline to model/complaint_classifier.joblib

IMPORTANT: the dataset is generated from a small set of sentence templates
per category (see generate_dataset.py). A naive random row-level train/test
split would let the same template appear in both train and test (just with
a different street/time filled in), which inflates accuracy by letting the
model memorize templates instead of learning the category. To get an honest
estimate, we split by TEMPLATE instead: entire templates are held out for
testing, so the test sentences use phrasing the model never saw at all.
"""
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# 1. Load data
df = pd.read_csv("data/complaints.csv")
print(f"Loaded {len(df)} complaints. Class balance:\n{df['category'].value_counts()}")

# 2. Template-level split: hold out ~25% of templates per category for testing
rng = pd.Series(df["template_id"].unique()).sample(frac=1, random_state=42)
test_templates = set()
for category in df["category"].unique():
    cat_templates = [t for t in rng if t.startswith(f"{category}::")]
    n_hold = max(1, round(len(cat_templates) * 0.25))
    test_templates.update(cat_templates[:n_hold])

test_mask = df["template_id"].isin(test_templates)
train_df, test_df = df[~test_mask], df[test_mask]
print(f"\nTrain: {len(train_df)} rows from {train_df['template_id'].nunique()} templates")
print(f"Test:  {len(test_df)} rows from {test_df['template_id'].nunique()} templates (unseen phrasing)")

X_train, y_train = train_df["text"], train_df["category"]
X_test, y_test = test_df["text"], test_df["category"]

# 3. Build pipeline: TF-IDF vectorizer -> Logistic Regression (multinomial)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
    ("clf", LogisticRegression(max_iter=1000)),
])

# 4. Train
pipeline.fit(X_train, y_train)

# 5. Evaluate on genuinely unseen phrasing
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy (unseen templates): {acc:.4f}\n")
print(classification_report(y_test, y_pred))

# 6. Save the trained pipeline
joblib.dump(pipeline, "model/complaint_classifier.joblib")
print("Saved model to model/complaint_classifier.joblib")
