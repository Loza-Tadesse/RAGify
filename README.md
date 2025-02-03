# RAGify

Ask questions about your PDF files using AI.

## Quick Start

```bash
# Install
git clone https://github.com/Loza-Tadesse/RAGify.git
cd RAGify
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure (create .env file)
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
echo "OPENAI_API_KEY=your_key_here" >> .env

# Run
streamlit run src/streamlit_app.py
```

Open **http://localhost:8501** and upload PDFs!

## Features

- 📄 Upload and query PDF documents
- Powered by Claude Sonnet & OpenAI
- In-memory vector storage (no database needed)
- One command to run

## Author

**Loza Tadesse** | [LinkedIn](https://linkedin.com/in/lozatadesse)
