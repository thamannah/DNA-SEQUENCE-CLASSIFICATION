# =======================
# IMPORTS
# =======================
import streamlit as st
import pandas as pd
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =======================
# PAGE CONFIG (FIRST STREAMLIT COMMAND)
# =======================
st.set_page_config(
    page_title="DNA Classifier",
    page_icon="🧬",
    layout="centered"
)

# =======================
# SIMPLE AUTHENTICATION
# =======================
USER_CREDENTIALS = {
    "admin": "1234",
    "researcher": "dna2025"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 DNA Classifier Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        username = username.strip().lower()
        password = password.strip()

        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
        else:
            st.error("❌ Invalid Credentials")

def logout():
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()
else:
    st.sidebar.button("Logout", on_click=logout)

# =======================
# CUSTOM CSS
# =======================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0a1a2f, #020b14);
    color: white;
}
section[data-testid="stMain"] > div {
    max-width: 1100px;
    margin: auto;
    padding: 2rem 3rem;
}
h2 {
    color: #4aa3ff;
    font-weight: 800;
    text-align: center;
}
h3 {
    color: #7ec8ff;
}
input {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 6px !important;
    border: 1px solid #374151 !important;
}
/* Sidebar Background */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #081426, #020b14);
    color: #dbeafe;
}

/* Logout Button - Theme Synced */
section[data-testid="stSidebar"] button {
    background: linear-gradient(180deg, #0f1f36, #081426) !important;
    color: #7ec8ff !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: 1px solid #1e3a8a !important;
    transition: 0.3s ease-in-out !important;
    text-align: center;
    margin: 10px auto; 
    align-items: center;
    
}

/* Hover Effect */
section[data-testid="stSidebar"] button:hover {
    background: linear-gradient(180deg, #1e3a8a, #0f1f36) !important;
    color: white !important;
    box-shadow: 0px 0px 12px rgba(74, 163, 255, 0.4);
}
</style>
""", unsafe_allow_html=True)

# =======================
# HEADER
# =======================
st.markdown("<h2>🧬 DNA Sequence Classifier</h2>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:#cbd5e1; font-size:16px;'>Machine learning–based genomic sequence classification</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# =======================
# LOAD DATASET
# =======================
data = pd.read_csv("../dataset/dna_sequences.csv")

rows_before = len(data)
data = data.dropna()
rows_after = len(data)

# =======================
# FEATURE EXTRACTION
# =======================
k = 3
vectorizer = CountVectorizer(analyzer="char", ngram_range=(k, k))
X = vectorizer.fit_transform(data["sequence"])
y = data["label"]

# =======================
# TRAIN TEST SPLIT
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# =======================
# SIDEBAR INFO
# =======================
st.sidebar.markdown("## ℹ️ About")
st.sidebar.info("""
This system uses a Naive Bayes classifier trained on DNA sequences.

Classes:
- Promoter regions
- Coding regions
- Non-Coding regions
""")

st.sidebar.markdown("## 📊 Dataset Info")
st.sidebar.write(f"Rows before cleaning: {rows_before}")
st.sidebar.write(f"Rows after cleaning: {rows_after}")
st.sidebar.bar_chart(data["label"].value_counts())

st.sidebar.markdown("## 📈 Model Performance")
st.sidebar.metric("Accuracy", f"{accuracy*100:.2f}%")

# =======================
# USER INPUT
# =======================
st.markdown("### 🔤 Input DNA Sequence")
user_input = st.text_input("Enter sequence (A, T, G, C only)")

if user_input:
    user_input = user_input.upper()

    if not re.fullmatch("[ATGC]+", user_input):
        st.error("❌ Invalid DNA sequence! Only A, T, G, C allowed.")
    else:
        length = len(user_input)
        gc_content = (user_input.count("G") + user_input.count("C")) / length * 100

        st.markdown("### 📏 Sequence Analysis")
        st.write(f"Length: {length} bases")
        st.write(f"GC Content: {gc_content:.2f}%")

        seq_features = vectorizer.transform([user_input])
        prediction = model.predict(seq_features)[0]

        st.markdown("### 🧪 Classification Result")

        if prediction.lower() == "promoter":
            st.success("🟢 Promoter Region Detected")
        elif prediction.lower() == "coding":
            st.info("🔵 Coding Region Detected")
        else:
            st.warning("🟡 Non-Coding Region Detected")

        st.markdown("### 📊 Prediction Confidence")
        probs = model.predict_proba(seq_features)[0]
        labels = model.classes_

        st.bar_chart(pd.DataFrame({"Probability": probs}, index=labels))