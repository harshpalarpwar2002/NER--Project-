import streamlit as st
import spacy
import spacy_streamlit

# Page configuration for a stylish look
st.set_page_config(page_title="Entity Explorer", page_icon="🔍", layout="wide")

# Load the spaCy model
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# Header Section
st.title("🔍 Named Entity Recognition (NER) Explorer")
st.markdown("""
Extract real-world objects like **People, Places, and Organizations** from your text instantly using spaCy.
""")

st.divider()

# Sidebar for Input
st.sidebar.header("Configuration")
default_text = "Virat Kohli was born in Delhi and plays cricket for India."
user_input = st.sidebar.text_area("Enter Text to Analyze:", value=default_text, height=200)

# Main Interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualization")
    if user_input:
        doc = nlp(user_input)
        # Using spacy-streamlit for the attractive "brushed" highlighting
        spacy_streamlit.visualize_ner(
            doc, 
            labels=nlp.get_pipe("ner").labels, 
            show_table=False, 
            title=None
        )
    else:
        st.warning("Please enter some text in the sidebar to begin.")

with col2:
    st.subheader("Entity Data")
    if user_input:
        doc = nlp(user_input)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        if entities:
            # Displaying as a clean table
            import pandas as pd
            df = pd.DataFrame(entities, columns=["Entity", "Label"])
            st.dataframe(df, use_container_width=True)
            
            # Brief explanation of labels
            with st.expander("What do these labels mean?"):
                st.write("**GPE**: Countries, Cities, States")
                st.write("**PERSON**: People, including fictional")
                st.write("**ORG**: Companies, Agencies, Institutions")
        else:
            st.info("No entities detected.")

st.sidebar.info("Built with Streamlit & spaCy")
