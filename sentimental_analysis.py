import streamlit as st
from transformers import pipeline
import json

# ---------------- Load sentiment model ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

sentiment_pipeline = load_model()

# ---------------- Streamlit UI ----------------
st.title("Sentiment Analysis App")
st.write("Enter multiple comments in JSON dictionary format and get Positive / Negative / Neutral results.")

# Default JSON dictionary
default_text = """{
    "review1": "I love this product, it works perfectly",
    "review2": "This is the worst service I have ever used",
    "review3": "The experience was okay, nothing special"
}"""

# Text area for input
input_text = st.text_area("Enter reviews (JSON dictionary)", value=default_text, height=200)

# Button to analyze
if st.button("Analyze Sentiments"):
    try:
        # Parse JSON input
        data = json.loads(input_text)
        results = {}

        # Analyze each comment
        for id, text in data.items():
            prediction = sentiment_pipeline(text)[0]
            label = prediction["label"]
            score = prediction["score"]

            if score < 0.6:
                sentiment = "Neutral"
            elif label == "POSITIVE":
                sentiment = "Positive"
            else:
                sentiment = "Negative"

            results[id] = {"text": text, "sentiment": sentiment}

        # Show individual results
        st.subheader("Sentiment Results")
        for id, res in results.items():
            st.write(f"**{id}**: {res['sentiment']} → {res['text']}")

        # Show summary counts
        pos_count = sum(1 for r in results.values() if r["sentiment"] == "Positive")
        neg_count = sum(1 for r in results.values() if r["sentiment"] == "Negative")
        neu_count = sum(1 for r in results.values() if r["sentiment"] == "Neutral")

        st.subheader("Summary Counts")
        st.write(f"Positive: {pos_count}")
        st.write(f"Negative: {neg_count}")
        st.write(f"Neutral: {neu_count}")

    except Exception as e:
        st.error(f"Error: {e}")
