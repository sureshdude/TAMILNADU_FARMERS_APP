import os
import shutil
import tempfile
import time
import tiktoken
from dotenv import load_dotenv
from tiktoken import _tiktoken
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is missing in your .env file.")

app = FastAPI(title="Tamil Nadu Farmers Scheme RAG API")

FAISS_INDEX_DIR = "faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


def get_vector_store():
    """Load existing FAISS index from disk if available."""
    index_file = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    if os.path.exists(index_file):
        return FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return None


@app.get("/indexed-files")
async def list_indexed_files():
    """Returns a list of PDF filenames currently stored in the FAISS index."""
    vector_store = get_vector_store()
    if vector_store is None:
        return {"files": []}

    files = sorted({
        doc.metadata.get("source_file", "Unknown PDF")
        for doc in vector_store.docstore._dict.values()
    })
    return {"files": files}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Ingests a PDF, splits into chunks, embeds in batches, and merges into FAISS."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the PDF.",
            )

        for doc in documents:
            doc.metadata["source_file"] = file.filename

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No selectable text found in this PDF (it may be a scanned image PDF).",
            )

        # Batch chunks so each OpenAI embedding request stays under 300,000 tokens
        BATCH_SIZE = 100
        vector_store = get_vector_store()

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)

            # Brief pause between batches to avoid OpenAI TPM rate limits on multi-PDF uploads
            if i + BATCH_SIZE < len(chunks):
                time.sleep(1)

        if vector_store is not None:
            vector_store.save_local(FAISS_INDEX_DIR)

        return {
            "status": "success",
            "filename": file.filename,
            "pages_processed": len(documents),
            "chunks_indexed": len(chunks),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/chat", response_model=ChatResponse)
async def chat_with_schemes(payload: ChatRequest):
    """Retrieves relevant Tamil Nadu farmer scheme context from FAISS and answers."""
    vector_store = get_vector_store()
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="Vector DB is empty. Please upload at least one Tamil Nadu Farmers Scheme PDF first.",
        )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    retrieved_docs = retriever.invoke(payload.question)

    context_blocks = []
    sources = set()
    for doc in retrieved_docs:
        src_name = doc.metadata.get("source_file", "Uploaded PDF")
        page_num = doc.metadata.get("page", 0) + 1
        sources.add(f"{src_name} (Page {page_num})")
        context_blocks.append(f"\n{doc.page_content}")

    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert assistant for Tamil Nadu Government Farmers Schemes "
            "(such as Uzhavar Sandhai, Kalaignarin All Village Integrated Agricultural Development Programme, "
            "crop insurance, subsidies, and agricultural loans). "
            "Answer the user's question strictly using the provided context from the uploaded PDFs. "
            "Include eligibility criteria, subsidy details, and required documents if present in the context. "
            "If the answer is not available in the context, state clearly that the uploaded documents do not contain that information.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": payload.question})

    return ChatResponse(answer=answer, sources=sorted(sources))


@app.delete("/reset-db")
async def reset_vector_db():
    """Clears the local FAISS index."""
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)
    return {"status": "FAISS vector database cleared."}