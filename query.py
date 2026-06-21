import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if not os.path.exists("vectorstore"):
    raise FileNotFoundError("❌ 'vectorstore' directory not found. Please run embed.py first!")

db = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

# k=5 targets a highly concise, information-rich context payload
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Connecting to Qwen2.5-72B-Instruct for stable, accurate serverless inference
client = InferenceClient(
    model="Qwen/Qwen2.5-72B-Instruct",
    token=HF_TOKEN
)

def ask_question(query):
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return "❌ Not found in dataset"

    print("\n📂 Retrieved Context Files:")
    context_parts = []
    for d in docs:
        print(f"- {d.metadata.get('source')}")
        context_parts.append(d.page_content)

    context = "\n\n---\n\n".join(context_parts)

    system_instruction = """You are an advanced, accurate assistant for the Computer Science Engineering (CSE) Department.

Rules:
1. Answer the question completely using ONLY the facts explicitly stated in the context blocks.
2. Turn data layouts into clean, human-readable answers (use bullet points where helpful). Do not output raw data lines.
3. If the answer cannot be factually verified from the context blocks, reply with: "Not found in dataset"."""

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Context blocks:\n{context}\n\nQuestion:\n{query}"}
            ],
            max_tokens=512,
            temperature=0.01
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Inference Error: {str(e)}"