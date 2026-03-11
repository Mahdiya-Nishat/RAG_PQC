# RAG_PQC 🔐⚛️

A Retrieval-Augmented Generation (RAG) based AI assistant specialized in **Post-Quantum Cryptography (PQC)**. Ask any question about PQC and get answers grounded in real research papers.

## 🚀 Live Demo
👉 [Try it here](https://huggingface.co/spaces/Mahdiya-Nishat/RAG_PQC)

## 🧠 What is this?
RAG_PQC is an AI-powered research assistant that:
- Reads and understands PQC research papers
- Answers questions using real paper content
- Shows exact PDF name and page number for every answer
- Built with RAG architecture — no hallucinations, only paper-grounded answers

## ⚙️ Tech Stack
| Component | Technology |
|-----------|-----------|
| Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Qwen 2.5 7B via HuggingFace |
| UI | Gradio |
| Hosting | HuggingFace Spaces |

## 📚 Topics Covered
- Lattice-based cryptography
- CRYSTALS-Kyber and CRYSTALS-Dilithium
- NIST PQC standardization
- Quantum-resistant protocols
- Post-quantum signatures
- Healthcare and IoT security with PQC

## 🏗️ Architecture
```
User Question
      ↓
ChromaDB searches 2191 chunks from research papers
      ↓
Top 3 relevant chunks retrieved
      ↓
Qwen 2.5 7B reads chunks + generates answer
      ↓
Answer + PDF source + page number returned
```

## 🛠️ Run Locally
```bash
git clone https://github.com/Mahdiya-Nishat/RAG_PQC.git
cd RAG_PQC
pip install -r requirements.txt
```

Add your HuggingFace token:
```python
export HF_TOKEN="your_token_here"
```

Run:
```bash
python app.py
```

## 📁 Project Structure
```
RAG_PQC/
├── app.py              ← main application
├── requirements.txt    ← dependencies
├── papers/             ← PQC research papers (PDFs)
├── database/           ← ChromaDB vector store
└── .gitignore
```

## 👩‍💻 Author
**Mahdiya Nishat**
- GitHub: [@Mahdiya-Nishat](https://github.com/Mahdiya-Nishat)
- HuggingFace: [@Mahdiya](https://huggingface.co/Mahdiya)git add .
git commit -m "added README 📚"
git push --force