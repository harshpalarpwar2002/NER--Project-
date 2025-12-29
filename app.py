import streamlit as st
import spacy
import spacy_streamlit
import os

# Page configuration
st.set_page_config(page_title="Entity Explorer", page_icon="🔍", layout="wide")

# FAIL-SAFE: Function to ensure model is loaded
@st.cache_resource
def load_model():
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        # If model is not found, download it automatically
        with st.spinner(f"Downloading {model_name}... please wait."):
            os.system(f"python -m spacy download {model_name}")
            return spacy.load(model_name)

nlp = load_model()

# --- UI Setup ---
st.title("🔍 Named Entity Recognition Explorer")
st.markdown("Extract entities like **People, Places, and Dates** instantly.")

# User Input
text = st.text_area("Enter your text below:", 
                    "Virat Kohli was born in Delhi and plays cricket for India.", 
                    height=150)

if st.button("Analyze Text"):
    doc = nlp(text)
    
    # Stylish NER Visualization
    spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, show_table=False)
    
    # Detailed Data Table
    st.subheader("Extracted Data")
    attrs = ["text", "label_", "start", "end"]
    data = [[getattr(ent, attr) for attr in attrs] for ent in doc.ents]
    st.table(data)
