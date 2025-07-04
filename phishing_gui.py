import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import re

# ---- INITIAL SETUP ----
nltk.download('stopwords')
stopwords_en = stopwords.words('english')
stopwords_es = stopwords.words('spanish')

# Load dataset
df = pd.read_csv("Phishing_Email.csv")
df = df.drop(columns=['Unnamed: 0'], errors='ignore')
df = df.dropna(subset=['Email Text'])
df.columns = ['text', 'label']
df['label'] = df['label'].map({'Phishing Email': 'phishing', 'Safe Email': 'legit'})

# Train model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopwords_en + stopwords_es)),
    ('clf', MultinomialNB())
])
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# ---- DANGER PHRASES ----
danger_phrases = [
    "haz clic aquí", "verifica tu cuenta", "urgente", "confirmar identidad", "evitar suspensión",
    "click here", "verify your account", "urgent", "unauthorized access", "reset your password"
]

def highlight_danger(text):
    for phrase in danger_phrases:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub(f"⚠️ **{phrase.upper()}** ⚠️", text)
    return text

# ---- STREAMLIT APP ----
st.set_page_config(page_title="Email Phishing Detector", page_icon="🔒")
st.title("📧 Email Phishing Detector (English & Español)")

lang = st.radio("🌍 Language / Idioma:", ["English", "Español"])
text_label = "Paste your email below:" if lang == "English" else "Pega tu correo abajo:"
btn_text = "Analyze" if lang == "English" else "Analizar"
result_label = "Result" if lang == "English" else "Resultado"
phishing_msg = "PHISHING EMAIL ⚠️" if lang == "English" else "CORREO PHISHING ⚠️"
legit_msg = "LEGITIMATE EMAIL ✅" if lang == "English" else "CORREO LEGÍTIMO ✅"

user_input = st.text_area(f"✉️ {text_label}", height=200)

if st.button(f"🔍 {btn_text}"):
    if not user_input.strip():
        st.warning("Please enter some text." if lang == "English" else "Por favor, introduce texto.")
    else:
        pred = pipeline.predict([user_input])[0]
        proba = pipeline.predict_proba([user_input])[0]
        confidence = max(proba) * 100
        result_text = phishing_msg if pred == "phishing" else legit_msg

        # Highlight suspicious phrases
        highlighted = highlight_danger(user_input)
        st.markdown("### ✉️ Email Analysis")
        st.markdown(highlighted, unsafe_allow_html=True)

        # Show result
        st.markdown(f"### ✅ {result_label}: **{result_text}**")
        st.markdown(f"🎯 **Confidence:** `{confidence:.2f}%`")

        # Generate report
        report_text = f"""
        --- PHISHING DETECTION REPORT ---

        Prediction: {pred.upper()}
        Confidence: {confidence:.2f}%
        Language: {lang}
        -----
        Original Email:
        {user_input}

        Detected Flags:
        {', '.join([p for p in danger_phrases if p.lower() in user_input.lower()])}
        """

        st.download_button(
            "📄 Download Report", 
            report_text, 
            file_name="phishing_report.txt"
        )
