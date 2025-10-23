# 🧠 Ragify — Ask Anything From Your Files

Ragify is an **AI-powered Retrieval-Augmented Generation (RAG)** system that allows you to **upload any PDF file** and ask natural-language questions about its contents.  
It combines **document parsing, embeddings, and LLMs** to deliver accurate, context-aware answers grounded in your own data.

---

## 🚀 Features
- 📄 Supports multiple file
- 🧩 Splits long documents into semantic chunks for efficient retrieval
- 🔍 Generates embeddings using **OpenAI’s `gpt-4o-mini`**
- 🤖 Retrieves relevant chunks and answers with **LLM-powered Q&A**
- 🗂️ Built with **FastAPI**, **LlamaIndex**, and **OpenAI API**
- ⚡ Easily extendable to use vector DBs like **FAISS**, **Pinecone**, or **Chroma**

---

## 🧰 Tech Stack
- **Python 3.10+**
- **FastAPI**
- **LlamaIndex / LangChain**
- **OpenAI API**
- **SentenceSplitter / Embeddings**
- **Uvicorn**
- **dotenv**

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/Ragify.git
cd Ragify
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# or
.venv\Scripts\activate         # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set up environment variables  
Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## ▶️ Run the Application

### Run locally with Uvicorn:
```bash
uvicorn main:app --reload
```

Then open your browser at  
👉 **http://127.0.0.1:8000**  
or test the API at  
👉 **http://127.0.0.1:8000/docs**

---

## 🧪 Example Usage

```bash
# 1. Load and chunk your document
python scripts/load_and_chunk.py path/to/file.pdf

# 2. Embed and upsert document chunks
python scripts/embed_and_upsert.py

# 3. Ask questions about your file
curl -X POST "http://127.0.0.1:8000/query"      -H "Content-Type: application/json"      -d '{"question": "Summarize the document", "top_k": 3}'
```


---

## 💡 Future Improvements
- Add a web UI for uploading and querying files
- Integrate persistent vector database (FAISS / Pinecone)
- Add multi-document support
- Implement authentication and session memory

---

## 👤 Author
**Loza Tadesse**  
Graduate Researcher & Data Scientist  
📧 lozatadesse@gmail.com | 🌐 [linkedin.com/in/lozatadesse](https://linkedin.com/in/lozatadesse)
