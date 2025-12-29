import streamlit as st
from transformers import pipeline

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Deep Learning NER",
    page_icon="🧠",
    layout="centered"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #2c3e50;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #555;
}
.entity-box {
    background-color: white;
    padding: 15px;
    margin-top: 10px;
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown('<div class="main-title">🧠 Deep Learning NER App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Named Entity Recognition using BERT</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )

ner_model = load_model()

# ---------------- Input ----------------
text = st.text_area(
    "✍️ Enter text:",
    "Virat Kohli was born in Delhi and plays cricket for India",
    height=140
)

# ---------------- Button ----------------
if st.button("🔍 Analyze Text"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        results = ner_model(text)

        st.subheader("📌 Extracted Entities")

        if results:
            for ent in results:
                st.markdown(
                    f"""
                    <div class="entity-box">
                        <b>Entity:</b> {ent['word']} <br>
                        <b>Type:</b> {ent['entity_group']} <br>_]()
