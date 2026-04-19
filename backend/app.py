# =========================
# Flask Backend API
# =========================

from flask import Flask, request, jsonify
import pickle
import re
import nltk
import os
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from skills import skills_db
from flask_cors import CORS
from predict_resume import predict_resume


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
CORS(app)

# Load ML models


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
tfidf = pickle.load(open(os.path.join(BASE_DIR, "tfidf.pkl"), "rb"))
le = pickle.load(open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb"))

# -------------------------
# PDF → Text
# -------------------------
def extract_text_from_pdf(file):
    text = ""
    reader = PdfReader(file)
    
    for page in reader.pages:
        text += page.extract_text() or ""
    
    return text

# -------------------------
# Clean text
# -------------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    return text

# -------------------------
# Skill extraction
# -------------------------
def extract_skills(text, role):
    detected = []
    role_skills = skills_db.get(role, [])
    
    for skill in role_skills:
        if skill.lower() in text:
            detected.append(skill)
    
    return detected

# -------------------------
# Missing skills
# -------------------------
def get_missing_skills(detected, role):
    required = skills_db.get(role, [])
    return [skill for skill in required if skill not in detected]

# -------------------------
# Match score (fixed version)
# -------------------------
def calculate_match_score(detected, role):
    required = skills_db.get(role, [])
    
    if len(required) == 0:
        return 0
    
    return round((len(detected) / len(required)) * 100, 2)

# -------------------------
# MAIN API
# -------------------------
@app.route('/')
def home():
    return " Resume Analyzer API is Running!"
    
@app.route('/predict', methods=['POST'])
def predict():
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    # Extract text
    raw_text = extract_text_from_pdf(file)
    
    if raw_text.strip() == "":
        return jsonify({"error": "Could not extract text"}), 400
    
    # Clean
    cleaned = clean_text(raw_text)
    
    # Predict role
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    role = le.inverse_transform(prediction)[0]
    
    # Skills
    # detected = extract_skills(cleaned, role)
    detected_clean = extract_skills(cleaned, role)
    detected_raw = extract_skills(raw_text, role)
    detected = list(set(detected_clean + detected_raw))
    missing = get_missing_skills(detected, role)
    
    # Score
    match_score = calculate_match_score(detected, role)
    
    return jsonify({
        "role": role,
        "match_score": match_score,
        "detected_skills": detected,
        "missing_skills": missing
    })

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)