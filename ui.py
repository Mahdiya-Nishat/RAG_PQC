import os
import gradio as gr
import requests
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Your token
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# Load everything
print("🌙 Initializing your PQC Tutor...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="database/", embedding_function=embeddings)
print("✨ Ready!")

from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)


def ask(question, history):
    # Retrieve
    docs = vectordb.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Get sources with page numbers!!
    sources = []
    for doc in docs:
        filename = doc.metadata.get("source", "Unknown").split("\\")[-1]
        page = doc.metadata.get("page", "?")
        sources.append(f"📄 **{filename}** — Page {page + 1}")

    prompt = f"""You are a PQC expert teacher. Use the context below to answer clearly and kindly.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.5
    )

    answer = response.choices[0].message.content

    # Show sources nicely!!
    sources_text = "\n\n---\n🔍 **References from PQC papers:**\n" + "\n".join(sources)

    return answer + sources_text


# Custom CSS - cozy night galaxy theme!!
css = """
body {
    background: linear-gradient(135deg, #0a0015 0%, #1a0030 40%, #0d0025 100%) !important;
    background-attachment: fixed !important;
}

.gradio-container {
    background: transparent !important;
    max-width: 800px !important;
    margin: auto !important;
}

#title {
    text-align: center;
    color: #e0aaff;
    font-size: 2.5em;
    font-family: 'Georgia', serif;
    text-shadow: 0 0 20px #c77dff, 0 0 40px #9d4edd;
    padding: 20px;
}

#subtitle {
    text-align: center;
    color: #c77dff;
    font-size: 1em;
    margin-bottom: 20px;
}

.chatbot {
    background: rgba(20, 0, 40, 0.8) !important;
    border: 1px solid #7b2fff !important;
    border-radius: 20px !important;
    box-shadow: 0 0 30px rgba(157, 78, 221, 0.3) !important;
}

.message {
    border-radius: 15px !important;
}

.user message {
    background: linear-gradient(135deg, #3c096c, #5a189a) !important;
}

footer {display: none !important}

#stars {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
}
"""

# Build UI
with gr.Blocks(css=css, title="🌙 The PQC BoT") as demo:
    gr.HTML("""
        <div id='title'>🐱✨ PQC Bot ✨🐱</div>
        <div id='subtitle'>🌙 The Cat guide to Post-Quantum Cryptography 🌙</div>
        <div style='text-align:center; font-size:2em;'>🌟 ⭐ 💫 ✨ 🌟 ⭐ 💫 ✨ 🌟</div>
    """)

    chatbot = gr.ChatInterface(
        fn=ask,
        chatbot=gr.Chatbot(
            height=450,
            avatar_images=("👤", "🐱"),
        ),
        textbox=gr.Textbox(
            placeholder="🌙 Ask me PQC Queries ...",
            container=False,
        ),
        examples=[
            "What is post quantum cryptography?",
            "How does CRYSTALS-Kyber work?",
            "What is lattice based cryptography?",
            "Why does quantum computing break RSA?",
            "What are NIST PQC standards?",
        ],
        title="",
    )

print("🌙 Ask Your PQC Queries...")
demo.launch(css=css)