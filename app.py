import os
import streamlit as tf
from query import ask_question

# ✅ Automatically build the database on Streamlit Cloud if it doesn't exist
if not os.path.exists("vectorstore"):
    with tf.spinner("Initializing first-time cloud database setup..."):
        # This imports and runs your embed code dynamically on the server
        import embed 

tf.set_page_config(page_title="CSE Dept Chatbot", page_icon="🎓")
tf.title("🎓 CSE Department RAG Chatbot")

# Simple user chat loop
query = tf.text_input("Ask anything about placements, faculty, results, patents, or department events:")

if query:
    with tf.spinner("Thinking..."):
        response = ask_question(query)
        tf.write(response)