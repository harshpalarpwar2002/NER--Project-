import streamlit as st
import spacy
import spacy_streamlit

# 1. Page Configuration for a modern look
st.set_page_config(page_title="NER Explorer", page_icon="🏷️", layout="wide")

# 2. Robust Model Loader (prevents the "Oh No" crash)
@st.cache_resource
def load_model():
    try:
        # Tries to load the small English model
        return spacy.load("en_core_web_sm")
    except OSError:
        # Fallback for local setup if not yet downloaded
        import os
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_model()

# 3. Main Interface
st.title("🏷️ Named Entity Recognition (NER) Explorer")
st.markdown("Extract and visualize entities from your text using **spaCy**.")

# Sidebar for Input
st.sidebar.header("Input Settings")
default_text = "Virat Kohli was born in Delhi and plays cricket for India."
user_input = st.sidebar.text_area("Paste your text here:", value=default_text, height=200)

# 4. Displaying Results
if user_input:
    doc = nlp(user_input)
    
    # Attractive NER visualization using spacy-streamlit
    st.subheader("Interactive Entity Visualization")
    spacy_streamlit.visualize_ner(
        doc, 
        labels=nlp.get_pipe("ner").labels, 
        show_table=True, 
        title=None
    )
else:
    st.info("Please enter some text to start the analysis.")
