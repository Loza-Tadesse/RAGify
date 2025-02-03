"""Standalone Streamlit UI for RAGify - No external servers needed!"""
from pathlib import Path
import streamlit as st
import atexit
import shutil

from ragify.config import settings
from ragify.document_processor import DocumentLoader, EmbeddingService
from ragify.memory_vector_store import InMemoryVectorStore
from ragify.rag_service import RAGService
from ragify.models import RAGQueryResult


st.set_page_config(page_title="RAGify - Upload & Query PDFs", page_icon="📄", layout="centered")


def cleanup_uploads():
    """Clean up uploaded files when the session ends."""
    uploads_dir = Path(settings.upload_dir)
    if uploads_dir.exists():
        shutil.rmtree(uploads_dir)
        st.write("🧹 Cleaned up uploaded files")


# Register cleanup to run when Python exits
atexit.register(cleanup_uploads)


@st.cache_resource
def get_rag_service() -> RAGService:
    """Get cached RAG service instance with in-memory vector store."""
    memory_store = InMemoryVectorStore()
    return RAGService(vector_store=memory_store)


def save_uploaded_pdf(file) -> Path:
    """Save uploaded file to disk."""
    uploads_dir = Path(settings.upload_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


def ingest_pdf(pdf_path: Path) -> int:
    """Ingest a PDF into the RAG system."""
    rag_service = get_rag_service()
    result = rag_service.ingest_document(str(pdf_path), source_id=pdf_path.name)
    return result.ingested


def query_rag(question: str, top_k: int = 5) -> dict:
    """Query the RAG system and get an answer."""
    rag_service = get_rag_service()
    
    # Search for relevant context
    search_result = rag_service.search_context(question, top_k)
    
    # Build prompt
    user_content = rag_service.build_prompt(question, search_result.contexts)
    
    # Generate answer using LLM
    import openai
    from anthropic import Anthropic
    
    # Try Anthropic first if configured
    if settings.anthropic_api_key:
        try:
            client = Anthropic(api_key=settings.anthropic_api_key)
            response = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=1024,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": user_content}
                ]
            )
            answer = response.content[0].text.strip()
        except Exception:
            # Fallback to OpenAI
            client = openai.OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model=settings.openai_model,
                max_tokens=1024,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You answer questions using only the provided context."},
                    {"role": "user", "content": user_content}
                ]
            )
            answer = response.choices[0].message.content.strip()
    else:
        # Use OpenAI
        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=1024,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        )
        answer = response.choices[0].message.content.strip()
    
    return {
        "answer": answer,
        "sources": search_result.sources,
        "num_contexts": len(search_result.contexts)
    }


# UI Layout
st.title("📄 RAGify - Upload & Query PDFs")
st.caption("100% standalone - No Docker, no external servers, just Python!")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("RAGify uses:")
    st.write("- 🤖 Claude Sonnet (or OpenAI fallback)")
    st.write("- 🔍 OpenAI embeddings")
    st.write("- � In-memory vector storage")
    
    st.divider()
    
    st.header("⚙️ Configuration")
    st.write(f"**LLM**: {settings.anthropic_model if settings.anthropic_api_key else settings.openai_model}")
    st.write(f"**Embeddings**: {settings.embedding_model}")
    
    # Show vector store stats
    rag_service = get_rag_service()
    vector_count = rag_service.vector_store.count()
    st.write(f"**Vectors stored**: {vector_count}")
    
    # Add cleanup button
    st.divider()
    if st.button("🧹 Clear All Files", type="secondary"):
        cleanup_uploads()
        st.success("Uploaded files cleared!")
        st.rerun()
    
    if not settings.anthropic_api_key and not settings.openai_api_key:
        st.warning("⚠️ No API keys configured! Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")

# Upload Section
st.header("📤 Upload a PDF")
uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Processing PDF..."):
        try:
            path = save_uploaded_pdf(uploaded)
            num_chunks = ingest_pdf(path)
            st.success(f"✅ Successfully ingested **{path.name}** ({num_chunks} chunks)")
            st.caption("You can now ask questions about this document below.")
        except Exception as e:
            st.error(f"❌ Failed to process PDF: {e}")

st.divider()

# Query Section
st.header("💬 Ask Questions")

with st.form("rag_query_form"):
    question = st.text_input("Your question", placeholder="What is this document about?")
    top_k = st.number_input("Number of chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask", type="primary")

    if submitted and question.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                result = query_rag(question.strip(), int(top_k))
                
                st.subheader("💡 Answer")
                st.write(result["answer"] or "(No answer generated)")
                
                if result["sources"]:
                    st.caption(f"📚 Sources ({len(result['sources'])} documents)")
                    for s in result["sources"]:
                        st.write(f"- {s}")
                
                st.caption(f"🔍 Retrieved {result['num_contexts']} relevant chunks")
                
            except Exception as e:
                st.error(f"❌ Query failed: {e}")
                if "api_key" in str(e).lower():
                    st.info("💡 Set your API keys in the .env file (ANTHROPIC_API_KEY or OPENAI_API_KEY)")
