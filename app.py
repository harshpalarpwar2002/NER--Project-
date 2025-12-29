import streamlit as st
import spacy
from spacy import displacy

# Load model safely
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Page config
st.set_page_config(page_title="NER App", page_icon="🧠")

# Title
st.title("🧠 Named Entity Recognition App")
st.write("Extract entities like **Person, Location, Organization**")

# Text input
text = st.text_area(
    "Enter text below:",
    "Virat Kohli was born in Delhi and plays cricket for India",
    height=120
)

# Button
if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        doc = nlp(text)

        st.subheader("📌 Entities Found")

        if doc.ents:
            for ent in doc.ents:
                st.write(f"**Entity:** {ent.text}")
                st.write(f"**Label:** {ent.label_}")
                st.write("---")
        else:
            st.info("No entities found")

        # Visualization
        st.subheader("🎨 Visualization")
        html = displacy.render(doc, style="ent", jupyter=False)
        st.components.v1.html(html, height=300, scrolling=True)
