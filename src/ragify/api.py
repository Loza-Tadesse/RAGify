"""FastAPI application and Inngest functions."""
import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api

from ragify.config import settings
from ragify.models import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult
from ragify.rag_service import RAGService
from ragify.llm import get_llm_adapter


# Initialize Inngest client
inngest_client = inngest.Inngest(
    app_id=settings.inngest_app_id,
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

# Initialize RAG service
rag_service = RAGService()


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    """Ingest a PDF document into the RAG system."""
    pdf_path = ctx.event.data["pdf_path"]
    source_id = ctx.event.data.get("source_id", pdf_path)
    
    def _ingest() -> RAGUpsertResult:
        return rag_service.ingest_document(pdf_path, source_id)
    
    result = await ctx.step.run("ingest-document", _ingest, output_type=RAGUpsertResult)
    return result.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    """Query the RAG system and generate an answer."""
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    
    # Search for relevant context
    def _search() -> RAGSearchResult:
        return rag_service.search_context(question, top_k)
    
    found = await ctx.step.run("search-context", _search, output_type=RAGSearchResult)
    
    # Build prompt
    user_content = rag_service.build_prompt(question, found.contexts)
    
    # Get LLM adapter
    adapter = get_llm_adapter()
    
    # Generate answer
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )
    
    answer = res["choices"][0]["message"]["content"].strip()
    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts)
    }


# Create FastAPI app
app = FastAPI(title="RAGify API", version="0.1.0")

# Register Inngest functions
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
