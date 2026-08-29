# Smart City Complaint Router

An end-to-end AI web application that automatically **routes citizen complaints to the correct city department** and **flags abusive submissions for moderator review** — the two things a real municipal 311/service-request system needs to do before a human ever reads the message. Built for the MSc AI *End-to-End AI Application Development and Cloud Deployment* assignment.

## 1. Problem Statement
Cities receive large volumes of non-emergency complaints (potholes, missed garbage collection, noise, water/power outages, park damage, safety hazards) through phone, web, or app-based 311-style systems. Two problems slow this down: (1) manually reading and routing every complaint to the correct department is slow and doesn't scale, and (2) abusive or harassing submissions still need to be identified before staff engage with them, ideally automatically.

## 2. Use Case
- **City/municipal service portals**: automatically pre-sort incoming complaints (Traffic & Roads, Sanitation & Waste, Water & Utilities, Public Safety, Noise Complaints, Parks & Public Spaces) before they reach the relevant department, and separately flag abusive submissions for moderator handling instead of department routing.
- **Community platforms/HOAs**: any smaller-scale system where residents submit free-text issues that need triage.
- **Teaching example**: a two-stage NLP pipeline (moderation + multi-class routing) built and evaluated with attention to a real methodological pitfall (see Section 5).

## 3. Solution Overview
The app runs text through two independently trained models:
1. **Toxicity screen** — a binary classifier flags abusive/toxic submissions. If flagged, the complaint is routed to a moderator instead of a department.
2. **Department router** — for non-toxic complaints, a 6-class classifier predicts which city department the complaint belongs to, with a confidence score.

Both stages use TF-IDF text features with Logistic Regression — a fast, explainable, and easy-to-audit approach appropriate for a moderation-adjacent civic system.

## 4. Dataset
This project uses two datasets:

- **Toxicity data**: the [Surge AI Toxicity Dataset](https://github.com/surge-ai/toxicity) — 1,000 human-labeled English comments (`Toxic` / `Not Toxic`).
- **Complaint routing data**: a **synthetically generated dataset** (`generate_dataset.py`), because real multi-category municipal 311 datasets (e.g. NYC/Chicago Open Data) require API/domain access not available in this project's development environment, and are also heavily dominated by one city's specific categories/agencies. The generator produces realistic complaint sentences from ~18-20 phrasing templates per category × 6 categories (Traffic & Roads, Sanitation & Waste, Water & Utilities, Public Safety, Noise Complaints, Parks & Public Spaces), with randomized street names and time expressions, producing 877 unique complaint texts. **This is disclosed transparently rather than presented as real municipal data** — see Section 5 for how it was evaluated honestly, and "Limitations & Future Work" for how to swap in a real dataset.

**Domain-adaptation step**: all 877 complaint texts are also reused as additional `Not Toxic` training examples for the toxicity model (see Section 5) — this is legitimate or "free" extra data, since every complaint in the generated set is non-toxic by construction.

## 5. AI/ML Approach

### Models
- **Toxicity classifier**: `TfidfVectorizer` (unigrams+bigrams, English stopwords removed) + `LogisticRegression(class_weight="balanced", C=3)`.
- **Complaint router**: `TfidfVectorizer` (unigrams+bigrams) + `LogisticRegression` (multinomial, 6 classes).
- **Frameworks/libraries**: scikit-learn, pandas, joblib, Streamlit.

### A methodological issue found and fixed during development
Initially, evaluating the complaint router with a standard random train/test split gave **100% accuracy** — a red flag, not a good sign. Because the dataset is generated from a limited set of sentence templates, a random row-level split let the same template appear in both train and test (just with a different street/time filled in), so the model was memorizing template shapes, not learning the categories.

**Fix**: the evaluation was redone with a **template-level split** — entire templates (not just rows) are held out for testing, so test sentences use phrasing the model never saw in any form during training. This is a stricter, more honest test of generalization.
- Complaint router accuracy on **held-out, unseen templates**: **~61%** across 6 balanced classes (vs. ~17% random-chance baseline).
- The confusion matrix shows a genuine, interpretable failure mode: complaints about **fire hydrants** get split between "Public Safety" and "Water & Utilities" — which makes sense, since a damaged hydrant genuinely sits at the intersection of both departments in real cities too.

### A second issue found and fixed: domain mismatch in the toxicity model
The toxicity model, trained only on general online comments, initially produced **false positives on legitimate complaints** — e.g. "...it smells terrible" and "...dangerous for kids" were flagged as toxic, because negative-but-ordinary words correlated with toxicity in its original (out-of-domain) training data.

**Fix**: the 877 complaint texts (all non-toxic by construction) were added as extra `Not Toxic` training examples — a cheap, effective domain-adaptation step. After retraining:
- Overall toxicity accuracy improved to **~90%** (from ~85% on the original, narrower dataset).
- All previously-misclassified legitimate complaints are now correctly classified as `Not Toxic`, while genuinely abusive text is still correctly flagged.

## 6. Application Architecture
```
                         ┌─────────────────────────┐
                         │      User Browser         │
                         │   (complaint text input)   │
                         └────────────┬────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │     Streamlit App          │
                         │        (app.py)            │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                     ▼
       ┌─────────────────────────┐          ┌─────────────────────────┐
       │  Stage 1: Toxicity check  │  ─────▶  │ Stage 2: Department router │
       │ (TF-IDF + LogReg,          │  if not  │  (TF-IDF + LogReg,          │
       │  toxic_classifier.joblib)  │  toxic   │   complaint_classifier.joblib)│
       └─────────────────────────┘          └─────────────────────────┘
```
- `generate_dataset.py` — builds the synthetic complaint dataset.
- `train.py` — trains the complaint router (with the template-level split described above).
- `train_toxicity.py` — retrains the toxicity model with the domain-adaptation fix.
- `model/complaint_classifier.joblib`, `model/toxic_classifier.joblib` — the two trained pipelines.
- `app.py` — the Streamlit web app that ties both models together.

## 7. Technology Stack
- **Language**: Python 3.11
- **ML**: scikit-learn, pandas, joblib
- **Web app**: Streamlit
- **Cloud/Deployment**: Streamlit Community Cloud
- **Containerization (fallback)**: Docker

## 8. Local Setup Instructions
```bash
git clone <your-repo-url>
cd smart-city-complaints
pip install -r requirements.txt

# (Optional) Regenerate data and retrain both models
python generate_dataset.py
python train.py
python train_toxicity.py

streamlit run app.py
```
The app will be available at `http://localhost:8501`.

## 9. Deployment Details
Deployed on **Streamlit Community Cloud** (free tier), which builds and serves the app directly from this GitHub repository — satisfying the assignment's cloud deployment requirement without needing a managed cloud account or Docker.

**Live app URL**: `<add your deployed Streamlit Cloud URL here>`

To deploy your own copy: push this repo to GitHub (public) → go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → "New app" → select this repo, branch `main`, main file `app.py` → Deploy.

## 10. API/Web Application Usage
1. Open the deployed app URL (or `http://localhost:8501` locally).
2. Type or paste a complaint into the text box.
3. Click **Submit complaint**.
4. The app either flags the submission as toxic (routed to a moderator), or shows the predicted department with a confidence score and a breakdown across all 6 departments.

## 11. Docker Instructions (fallback, if cloud deployment is unavailable)
```bash
docker build -t smart-city-complaints .
docker run -p 8501:8501 smart-city-complaints
```
The app will be available at `http://localhost:8501`.

## Limitations & Future Work
- The complaint-routing dataset is synthetically generated (see Section 4) due to environment access constraints, not sourced from a real city. Swapping in a real 311 dataset (e.g. downloading NYC Open Data's 311 Service Requests and mapping its complaint-type field to these 6 categories) would be a natural next step, and the same template-level evaluation methodology would still apply.
- At ~61% held-out accuracy across 6 classes, the router is meaningfully better than chance but not production-grade — more templates, more lexical variety, or real-world data would likely close the gap further.
- The toxicity/router pipeline is a simple two-stage system; a production system would also want human-in-the-loop review for low-confidence predictions in both stages.

---
*Built for the MSc AI "End-to-End AI Application Development and Cloud Deployment" assignment.*
