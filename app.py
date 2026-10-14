
import os, re, json, pickle
import numpy as np
import streamlit as st
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, "flipitnews_rnn.keras"))
    with open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb") as f: tokenizer = pickle.load(f)
    with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f: encoder = pickle.load(f)
    with open(os.path.join(BASE_DIR, "stopwords.pkl"), "rb") as f: stop_words = pickle.load(f)
    with open(os.path.join(BASE_DIR, "config.json")) as f: config = json.load(f)
    return model, tokenizer, encoder, stop_words, config

def preprocess(text, stop_words):
    text = str(text).lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    return [w for w in word_tokenize(text) if w not in stop_words and len(w) > 1]

st.set_page_config(page_title="FlipItNews Classifier", page_icon="📰")
st.title("📰 FlipItNews")
st.subheader("AI-Powered News Classification")
st.write("Paste a news article to predict its category using the trained RNN.")
try:
    model, tokenizer, encoder, stop_words, config = load_artifacts()
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()

article = st.text_area("News article", height=220, placeholder="Paste your news article here...")
if st.button("Classify News", type="primary"):
    if not article.strip():
        st.warning("Please enter a news article first.")
    else:
        tokens = preprocess(article, stop_words)
        if not tokens:
            st.warning("Please enter meaningful English text.")
        else:
            seq = tokenizer.texts_to_sequences([" ".join(tokens)])
            padded = pad_sequences(seq, maxlen=config["max_len"], padding="post", truncating="post")
            probs = model.predict(padded, verbose=0)[0]
            order = np.argsort(probs)[::-1]
            idx = int(order[0])
            st.success(f"Predicted category: {encoder.inverse_transform([idx])[0]}")
            st.metric("Estimated confidence", f"{float(probs[idx]):.2%}")
            st.write("### Category-wise probabilities")
            st.bar_chart({str(encoder.classes_[i]): float(probs[i]) for i in order})
            st.caption("Confidence is an estimated model probability, not a guarantee.")
