import os
import streamlit as tf

# ✅ CRITICAL: Check and build the database FIRST before importing query!
# This prevents query.py from throwing a FileNotFoundError when the cloud server boots up.
if not os.path.exists("vectorstore"):
    import embed 

# Now it is completely safe to import query
from query import ask_question

tf.set_page_config(page_title="CSE Dept Chatbot", page_icon="🎓")
tf.title("🎓 CSE Department RAG Chatbot")

# Simple user chat loop
query = tf.text_input("Ask anything about placements, faculty, results, patents, or department events:")

if query:
    with tf.spinner("Thinking..."):
        response = ask_question(query)
        tf.write(response)