# =======================
# IMPORTS
# =======================
import streamlit as st
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt

# =======================
# PAGE CONFIG
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
# HEADER
# =======================
st.markdown("<h2> 🧬 DNA Sequence Classifier</h2>", unsafe_allow_html=True)
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

# Robust label normalization
data["label"] = data["label"].astype(str).str.strip().str.lower()
data["label"] = data["label"].replace({
    "noncoding": "non-coding",
    "non coding": "non-coding",
    "non_coding": "non-coding",
    "non-coding": "non-coding"  # unify spelling
})

# Debugging: show unique labels and sample rows
st.write("Unique labels in dataset:", data["label"].unique())

# =======================
# BALANCE DATASET SAFELY
# =======================
promoter = data[data["label"] == "promoter"]
coding = data[data["label"] == "coding"]
noncoding = data[data["label"] == "non-coding"]

if len(noncoding) > 0:
    noncoding_upsampled = resample(
        noncoding,
        replace=True,
        n_samples=max(len(promoter), len(coding)),
        random_state=42
    )
    data_balanced = pd.concat([promoter, coding, noncoding_upsampled])
else:
    st.warning("⚠️ No non-coding sequences found after cleaning. Training only on promoter and coding.")
    data_balanced = pd.concat([promoter, coding])

# =======================
# FEATURE EXTRACTION
# =======================
k = st.sidebar.slider("Select k-mer size", min_value=3, max_value=6, value=4)
vectorizer = CountVectorizer(analyzer="char", ngram_range=(k, k))
X = vectorizer.fit_transform(data_balanced["sequence"])
y = data_balanced["label"]

# =======================
# TRAIN TEST SPLIT
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# RANDOM FOREST MODEL
# =======================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# =======================
# SIDEBAR INFO
# =======================
st.sidebar.markdown("## ℹ️ About")
st.sidebar.info("""
This system uses a Random Forest classifier trained on DNA sequences.

Classes:
- Promoter regions
- Coding regions
- Non-Coding regions
""")

st.sidebar.markdown("## 📊 Dataset Info")
st.sidebar.write(f"Rows before cleaning: {rows_before}")
st.sidebar.write(f"Rows after cleaning: {rows_after}")
st.sidebar.bar_chart(data_balanced["label"].value_counts())

# Pie chart for class distribution
st.sidebar.markdown("## 🧬 Class Distribution")
fig, ax = plt.subplots()
data_balanced["label"].value_counts().plot.pie(
    autopct="%1.1f%%", colors=["#4aa3ff", "#7ec8ff", "#facc15"], ax=ax
)
ax.set_ylabel("")
st.sidebar.pyplot(fig)

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
        if prediction == "promoter":
            st.success("🟢 Promoter Region Detected")
        elif prediction == "coding":
            st.info("🔵 Coding Region Detected")
        elif prediction == "non-coding":
            st.warning("🟡 Non-Coding Region Detected")
        else:
            st.error(f"❓ Unknown classification: {prediction}")

        st.markdown("### 📊 Prediction Confidence")
        probs = model.predict_proba(seq_features)[0]
        labels = model.classes_
        st.bar_chart(pd.DataFrame({"Probability": probs}, index=labels))