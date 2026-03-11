import os
import gradio as gr
from openai import OpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

HF_TOKEN = os.environ.get("HF_TOKEN", "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")

print("🌙 Initializing your PQC Tutor...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="database/", embedding_function=embeddings)
client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)
print("✨ Ready!")

def ask(question, history):
    docs = vectordb.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = []
    for doc in docs:
        filename = doc.metadata.get("source", "Unknown").split("\\")[-1]
        page = doc.metadata.get("page", "?")
        sources.append(f"📄 **{filename}** — Page {int(page)+1}")

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
    sources_text = "\n\n---\n🔍 **References from PQC papers:**\n" + "\n".join(set(sources))
    return answer + sources_text

css = """
* { box-sizing: border-box; }

body, .gradio-container {
    background: #ffffff !important;
    max-width: 100% !important;
    padding: 0 40px !important;
}

#title {
    text-align: center;
    color: #222;
    font-size: 2.2em;
    font-family: 'Georgia', serif;
    padding: 20px 0 5px 0;
}

#subtitle {
    text-align: center;
    color: #666;
    font-size: 0.95em;
    margin-bottom: 15px;
}

.chatbot {
    background: #ffffff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
}

.user .message {
    background: #f5f5f5 !important;
    color: #111111 !important;
    border-radius: 18px 18px 4px 18px !important;
    border: none !important;
    padding: 12px 16px !important;
}

.bot .message {
    background: #f5f5f5 !important;
    color: #111111 !important;
    border-radius: 18px 18px 18px 4px !important;
    border: 1px solid #e0e0e0 !important;
    padding: 12px 16px !important;
}

.textbox textarea {
    background: #ffffff !important;
    border: 1.5px solid #4f46e5 !important;
    border-radius: 12px !important;
    color: #111 !important;
    font-size: 1em !important;
}

button.primary {
    background: #222222 !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
}

button.primary:hover {
    background: #4338ca !important;
}

footer { display: none !important; }
"""

with gr.Blocks(title="🌙 PQC Bot") as demo:
    gr.HTML("""
        <div id='title'>🐱✨ PQC Bot ✨🐱</div>
        <div id='subtitle'>🌙 The Cat Guide to Post-Quantum Cryptography 🌙</div>
        <div style='text-align:center; font-size:2em;'>🌟 ⭐ 💫 ✨ 🌟 ⭐ 💫 ✨ 🌟</div>
    """)

    gr.ChatInterface(
        fn=ask,
        chatbot=gr.Chatbot(
            height=450,
            avatar_images=("👤", "🐱"),
            show_label=False,
        ),
        textbox=gr.Textbox(
            placeholder="🌙 Ask me PQC Queries...",
            container=False,
        ),
        examples=[
            "What is post quantum cryptography?",
            "How does CRYSTALS-Kyber work?",
            "What is lattice based cryptography?",
            "Why does quantum computing break RSA?",
            "What are NIST PQC standards?",
        ],
        submit_btn="✨ Ask",
    )

print("🌙 Ask Your PQC Queries...")
demo.launch(css=css)