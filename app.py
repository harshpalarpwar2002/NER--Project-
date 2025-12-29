import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(
    page_title="Deep Learning NER",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 Deep Learning Named Entity Recognition")
st.write("Using BERT model from Hugging Face Transformers")

# Load model (cached)
@st.cache_resource
def load_ner_model():
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )

ner_model = load_ner_model()

# Input text
text = st.text_area(
    "Enter text:",
    "Virat Kohli was born in Delhi and plays cricket for India",
    height=120
)

# Button
if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        results = ner_model(text)

        st.subheader("📌 Extracted Entities")

        if results:
            for ent in results:
                st.write(f"**Entity:** {ent['word']}")
                st.write(f"**Label:** {ent['entity_group']}")
                st.write(f"**Confidence:** {round(ent['score'], 3)}")
                st.write("---")
        else:
            st.info("No entities found")
