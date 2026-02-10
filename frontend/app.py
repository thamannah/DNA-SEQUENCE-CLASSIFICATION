import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# =======================
# Page config
# =======================
st.set_page_config(
    page_title="DNA Classifier",
    page_icon="🧬",
    layout="centered"
)

# =======================
# Global CSS
# =======================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0a1a2f, #020b14);
    color: white;
}
section[data-testid="stMain"] > div {
    max-width: 1200px;
    margin: auto;
    padding: 2rem 3rem;
    background: transparent;
    box-shadow: none;
    border-radius: 0;
}
h2 {
    color: #4aa3ff;
    font-weight: 800;
    text-align: center;
}
h3 {
    color: #7ec8ff;
}
div[data-testid="stCaptionContainer"] {
    text-align: center;
    color: #cbd5e1;
}
input {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 6px !important;
    border: 1px solid #374151 !important;
}
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #081426, #020b14);
    color: #dbeafe;
}
div[data-testid="stTextInput"] label {
    color: #cbd5e1 !important;
    font-size: 14px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# =======================
# Header
# =======================
st.markdown(
    "<h2 style='color:#4aa3ff; text-align:center;'>🧬 DNA Sequence Classifier</h2>",
    unsafe_allow_html=True
)
st.caption("Machine learning–based classification of genomic DNA sequences")
st.markdown("---")

# =======================
# Sidebar
# =======================
st.sidebar.markdown("## ℹ️ About the Project")
st.sidebar.info(
    """
    This system uses a **Naive Bayes classifier** trained on DNA sequences.

    It classifies sequences into:
    - **Promoter regions**
    - **Coding regions**
    - **Non-Coding regions**

    The application provides a clean interface for
    genomic sequence input and visualization of
    classification results.
    """
)

# =======================
# Load & clean dataset
# =======================
data = pd.read_csv("../dataset/dna_sequences.csv")

# Drop rows with missing values (NaN)
rows_before = len(data)
data = data.dropna()
rows_after = len(data)


# =======================
# Feature extraction
# =======================
k = 3
vectorizer = CountVectorizer(analyzer="char", ngram_range=(k, k))
X = vectorizer.fit_transform(data["sequence"])
y = data["label"]

# =======================
# Train model
# =======================
model = MultinomialNB()
model.fit(X, y)

# =======================
# Input & Prediction
# =======================
st.markdown(
    "<p style='font-size:22px; font-weight:600; color:white;'>🔤 Input DNA Sequence</p>",
    unsafe_allow_html=True
)

user_input = st.text_input("Enter sequence (A, T, G, C only)")

if user_input:
    seq_features = vectorizer.transform([user_input])
    prediction = model.predict(seq_features)[0]

    st.subheader("🧪 Classification Result")
    st.success(f"Predicted Class: **{prediction}**")

    st.subheader("📊 Prediction Confidence")
    probs = model.predict_proba(seq_features)[0]
    labels = model.classes_

    st.bar_chart(pd.DataFrame({"Probability": probs}, index=labels))