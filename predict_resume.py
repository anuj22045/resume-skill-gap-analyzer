import pickle
import re
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from skills import skills_db
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# load model 
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))


# extract text from pdf 
def extract_text_from_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    
    for page in reader.pages:
        text += page.extract_text() or ""
    
    return text

# Clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    return " ".join(words)

def extract_skills(text, role):
    detected = []
    
    role_skills = skills_db.get(role, [])
    
    for skill in role_skills:
        if skill.lower() in text:
            detected.append(skill)
    
    return detected


def get_missing_skills(detected, role):
    required = skills_db.get(role, [])
    
    missing = [skill for skill in required if skill not in detected]
    
    return missing

def calculate_Similarity_score(resume_text, role):
    required_skills = skills_db.get(role, [])
    
    # Convert required skills list into single string
    required_text = " ".join(required_skills)
    
    # Convert both to vectors using SAME tfidf
    resume_vec = tfidf.transform([resume_text])
    required_vec = tfidf.transform([required_text])
    
    # Cosine similarity
    score = cosine_similarity(resume_vec, required_vec)[0][0]
    
    return round(score * 100, 2)

def calculate_match_score(detected, role):
    required = skills_db.get(role, [])
    
    if len(required) == 0:
        return 0
    
    matched = len(detected)
    total = len(required)
    
    score = (matched / total) * 100
    
    return round(score, 2)


# Predict function
def predict_resume(file_path):
    raw_text = extract_text_from_pdf(file_path)
    
    if raw_text.strip() == "":
        return {
        "role": "Unknown",
        "detected_skills": [],
        "missing_skills": [],
        "match_score": 0,
        "Similarity_score":0,
        "error": "Could not extract text from PDF"
    }
    
    cleaned = clean_text(raw_text)
    
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    role = le.inverse_transform(prediction)[0]
    
    # Skill logic
    detected_skills = extract_skills(cleaned, role)
    missing_skills = get_missing_skills(detected_skills, role)
    match_score = calculate_match_score(detected_skills, role)
    similarity_score = calculate_Similarity_score(cleaned, role)
    
    return {
        "role": role,
        "detected_skills": detected_skills,
        "missing_skills": missing_skills,
        "match_score": match_score,
        "Similarity_score": similarity_score
    }



# TEST YOUR MODEL
if __name__ == "__main__":
    file_path = "Resume_Anuj.pdf"
    
    result = predict_resume(file_path)
    
    if "error" in result:
        print("Error:", result["error"])
    else:
        print("\nRole:", result["role"])
        print("\n Match Score:", result["match_score"], "%")
        print("\n Match Score:", result["Similarity_score"], "%")
        print("\n Skills:", result["detected_skills"])
        print("\nMissing:", result["missing_skills"])

    