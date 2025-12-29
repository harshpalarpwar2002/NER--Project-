import streamlit as st
import spacy

st.title("Named Entity Recognition App")

# Load model
nlp = spacy.load("en_core_web_sm")

text = st.text_input(
    "Enter text:",
    "Virat Kohli was born in Delhi and plays cricket for India"
)

if st.button("Analyze"):
    doc = nlp(text)

    if doc.ents:
        for ent in doc.ents:
            st.write("Entity:", ent.text)
            st.write("Label:", ent.label_)
            st.write("-----")
    else:
        st.write("No entities found")
