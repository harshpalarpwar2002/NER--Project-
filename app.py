import streamlit as st
import spacy
from spacy import displacy

# Load SpaCy model
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# Page configuration
st.set_page_config(
    page_title="NER App",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        color: #2c3e50;
        font-size: 40px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #34495e;
        font-size: 18px;
    }
    .entity-box {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🧠 Named Entity Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by SpaCy & Streamlit</div>', unsafe_allow_html=True)

st.markdown("---")

# Text input
text = st.text_area(
    "✍️ Enter text to analyze:",
    "Virat Kohli was born in Delhi and plays cricket for India",
    height=150
)

# Button
if st.button("🔍 Analyze Text"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        doc = nlp(text)

        st.subheader("📌 Extracted Entities")

        if doc.ents:
            for ent in doc.ents:
                st.markdown(
                    f"""
                    <div class="entity-box">
                        <b>Entity:</b> {ent.text} <br>
                        <b>Label:</b> {ent.label_}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No entities found")

        # Visualization
        st.subheader("🎨 Entity Visualization")
        html = displacy.render(doc, style="ent", jupyter=False)
        st.components.v1.html(html, scrolling=True, height=300)
