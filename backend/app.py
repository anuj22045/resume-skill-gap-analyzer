from flask import Flask, request, jsonify
import fitz
import re
import pickle

app = Flask(__name__)

model = pickle.load(open("../model/model.pkl", "rb"))
tfidf = pickle.load(open("../model/tfidf.pkl", "rb"))
le = pickle.load(open("../model/label_encoder.pkl", "rb"))

# clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    try:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += str(page.get_text())
    except:
        return ""
    return text


@app.route('/')
def home():
    return "Flask is running 🚀"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['resume']

        # file type check
        if not file.filename.endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        text = extract_text_from_pdf(file)

        if not text.strip():
            return jsonify({"error": "Could not extract text from PDF"}), 400

        cleaned_text = clean_text(text)

        if len(cleaned_text) < 50:
            return jsonify({"error": "Resume text too short"}), 400

        # IMPORTANT FIX
        vector = tfidf.transform([cleaned_text]).toarray()

        prediction = model.predict(vector)
        predicted_role = le.inverse_transform(prediction)[0]

        print("------cleaned text----------")
        print(cleaned_text[:300])

        return jsonify({
            "message": "Resume processed successfully",
            "predicted_role": predicted_role
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)