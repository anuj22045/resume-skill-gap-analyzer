import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv("data/Resume.csv")
df.columns = df.columns.str.strip()

tech_roles = [
    "INFORMATION-TECHNOLOGY",
    "ENGINEERING"
]

df = df[df['Category'].isin(tech_roles)]

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    
    return " ".join(words)

df['cleaned_resume'] = df['Resume_str'].apply(clean_text)

df = df[df['cleaned_resume'].str.strip() != ""]


tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    stop_words='english'
)
X = tfidf.fit_transform(df['cleaned_resume'])

le = LabelEncoder()
y = le.fit_transform(df['Category'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("All files saved successfully!")