import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "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")

# Load PDFs
print("Loading your PQC papers...")
loader = PyPDFDirectoryLoader("papers/")
documents = loader.load()
print(f"Loaded {len(documents)} pages!")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks!")

# Embeddings
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load database
print("Loading database...")
vectordb = Chroma(persist_directory="database/", embedding_function=embeddings)
print("Database loaded!")

# OpenAI compatible HuggingFace client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

def ask(question):
    # R - Retrieve
    docs = vectordb.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # A - Augment
    prompt = f"""You are a PQC expert teacher. Use the context below to answer clearly.

Context:
{context}

Question: {question}

Answer:"""

    # G - Generate
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.5
    )
    return response.choices[0].message.content

# Test!
question = "What is post quantum cryptography and why do we need it?"
print(f"\nQuestion: {question}")
print("Thinking...")
answer = ask(question)
print(f"\nAnswer: {answer}")