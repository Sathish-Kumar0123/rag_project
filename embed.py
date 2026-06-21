import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data"
documents = []

if not os.path.exists(DATA_PATH):
    print("❌ Error: 'data' folder not found!")
    exit()

# ✅ Load all your .txt files dynamically
for file in os.listdir(DATA_PATH):
    if file.endswith(".txt"):
        file_path = os.path.join(DATA_PATH, file)
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
            
            # Split your files by your custom dividers
            if "---" in file_content:
                chunks = file_content.split("---")
            else:
                chunks = file_content.split("# Chunk")
                
            for chunk in chunks:
                clean_text = chunk.strip()
                if len(clean_text) > 20:
                    documents.append(
                        Document(
                            page_content=f"Source File: {file}\n\n{clean_text}",
                            metadata={"source": file}
                        )
                    )

print(f"Loaded {len(documents)} text chunks from your files.")

# ✅ Embeddings using standard MiniLM
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ✅ Save Vector Store locally
db = FAISS.from_documents(documents, embedding_model)
db.save_local("vectorstore")
print("✅ Vector DB successfully updated with your text files!")
