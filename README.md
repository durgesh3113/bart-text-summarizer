# Text Summarizer using BART

A clean, beginner-friendly web app that summarizes long articles or documents using Facebook's pre-trained BART model. Built with  Hugging Face Transformers and Streamlit.

---

## Features

- Paste or type any long text to summarize
-  Adjustable summary length with sliders
-  One-click summarization
-  Download summary as a text file
-  Powered by `facebook/bart-large-cnn`
-  Simple, responsive Streamlit UI

---

## Requirements

Install required Python libraries:
| Library            | Purpose                                                          |
| ------------------ | ---------------------------------------------------------------- |
| streamlit   | To create the interactive web-based user interface (UI)          |
| transformers | To load the BART model and tokenizer from Hugging Face           |
| torch       | Backend framework (PyTorch) used by the BART model for inference |


bash
pip install -r requirements.txt


follow link to run the application
https://bart-text-summarizer-2xgjdnyaxkkk2fuxuqvfal.streamlit.app/
