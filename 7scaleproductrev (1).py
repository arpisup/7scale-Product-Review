import streamlit as st
import pickle
import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ================= LOAD =================
@st.cache_resource
def load_all():
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    nlp = spacy.load("en_core_web_sm")
    return model, vectorizer, nlp

model, vectorizer, nlp = load_all()

# ================= NLP SETUP =================
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {"not", "no", "never"}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ================= SENTIMENT SCALE =================
def get_sentiment_7scale(score):
    if score >= 2: return "Strongly Positive"
    elif score >= 1: return "Positive"
    elif score > 0: return "Slightly Positive"
    elif score == 0: return "Neutral"
    elif score > -1: return "Slightly Negative"
    elif score > -2: return "Negative"
    else: return "Strongly Negative"

# ================= UI =================
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("📊 Product Review Sentiment Analyzer")

# ================= INPUT =================
review = st.text_area("Enter a review")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review")

    else:
        cleaned = clean_text(review)
        vector = vectorizer.transform([cleaned])

        pred = model.predict(vector)[0]
        score = model.decision_function(vector)[0]
        sentiment7 = get_sentiment_7scale(score)

        st.subheader("🔍 Results")
        st.write(f"**Binary:** {pred}")
        st.write(f"**7-Scale:** {sentiment7}")
        st.write(f"**Score:** {score:.2f}")

        if pred == "Positive":
            st.success("👍 Positive")
        else:
            st.error("👎 Negative")

        # ================= POS TAGGING =================
        st.subheader("🧠 POS Tagging")

        doc = nlp(review)

        pos_data = [(t.text, t.pos_) for t in doc]
        st.table(pd.DataFrame(pos_data, columns=["Token", "POS"]))

        # POS Graph
        pos_counts = Counter([t.pos_ for t in doc if not t.is_space])

        fig, ax = plt.subplots()
        ax.bar(pos_counts.keys(), pos_counts.values())
        plt.xticks(rotation=45)
        ax.set_title("POS Distribution")

        st.pyplot(fig)

        # ================= FEATURE EXTRACTION =================
        st.subheader("🔍 Extracted Noun Features")

        nouns = [t.text for t in doc if t.pos_ == "NOUN"]
        st.write(nouns)

        # ================= DEPENDENCY RELATIONS =================
        st.subheader("🔗 Feature-Sentiment Pairs")

        pairs = []

        for token in doc:
            if token.dep_ == "amod" and token.head.pos_ == "NOUN":
                pairs.append((token.head.text, token.text))

        if pairs:
            st.write(pairs[:10])
        else:
            st.write("No strong feature-opinion pairs found")
