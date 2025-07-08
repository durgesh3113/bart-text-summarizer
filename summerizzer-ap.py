import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load model and tokenizer (with caching to prevent reloading on every run)
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Function to generate summary
def summarize(text, min_len, max_len):
    if not text.strip():
        return "⚠️ Please enter some text first."

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=max_len,
        min_length=min_len,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI setup
st.set_page_config(page_title="BART Text Summarizer", layout="centered")
st.title("🧠 BART Text Summarizer")
st.write("Paste any long article or document below and click summarize to get a short version.")

# User text input
text = st.text_area("📄 Enter your text here:", height=300, placeholder="Paste or type your article...")

# Sliders for summary length
st.write("🔧 Customize summary length:")
col1, col2 = st.columns(2)
min_len = col1.slider("Minimum length", 30, 100, 50)
max_len = col2.slider("Maximum length", 100, 300, 150)

# Summarize button
if st.button("✨ Summarize Now"):
    with st.spinner("Generating summary..."):
        summary = summarize(text, min_len, max_len)

    st.subheader("📝 Summary:")
    st.write(summary)

    # Download option
    if "⚠️" not in summary:
        st.download_button("💾 Download Summary", summary, file_name="summary.txt")

st.markdown("---")
st.caption("Built with ❤️ using Hugging Face Transformers & Streamlit")
