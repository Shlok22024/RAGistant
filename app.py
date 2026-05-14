import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
import fitz
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.llms.mock import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


load_dotenv()

APP_TITLE = "ResearchGPT"
APP_SUBTITLE = "RAG Based Research Paper Assistant"
CHROMA_DIR = Path(".chroma")
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


CITATION_PROMPT = PromptTemplate(
    "You are ResearchGPT, a careful research-paper assistant.\n"
    "Answer only from the provided context. Cite the evidence inline using "
    "bracketed source labels such as [1] or [2]. If the context does not answer "
    "the question, say that the uploaded paper does not provide enough evidence.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer with concise reasoning and source citations."
)


@dataclass(frozen=True)
class RagSettings:
    embedding_provider: str
    openai_api_key: str
    openai_embed_model: str
    openai_chat_model: str
    local_embed_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.lower()).strip("-")
    return value[:48] or "paper"


def make_collection_name(files: tuple[tuple[str, bytes], ...], settings: RagSettings) -> str:
    digest = hashlib.sha256()
    names: list[str] = []

    for file_name, file_bytes in files:
        digest.update(file_name.encode("utf-8"))
        digest.update(file_bytes)
        names.append(slugify(file_name.removesuffix(".pdf")))

    digest.update(
        (
            f"{settings.embedding_provider}|{settings.openai_embed_model}|"
            f"{settings.local_embed_model}|{settings.chunk_size}|{settings.chunk_overlap}"
        ).encode("utf-8")
    )
    readable_name = slugify("-".join(names))[:36]
    return f"researchgpt-{readable_name}-{digest.hexdigest()[:10]}"


def read_pdf_pages(file_bytes: bytes, file_name: str) -> list[Document]:
    documents: list[Document] = []

    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True).strip()
            if not text:
                continue

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": file_name,
                        "page_label": str(page_number),
                        "source_label": f"{file_name}, page {page_number}",
                    },
                )
            )

    return documents


@st.cache_resource(show_spinner=False)
def get_embedding_model(
    embedding_provider: str,
    openai_api_key: str,
    openai_embed_model: str,
    local_embed_model: str,
):
    if embedding_provider == "OpenAI":
        if not openai_api_key:
            raise ValueError("Add an OpenAI API key or switch to local Hugging Face embeddings.")
        return OpenAIEmbedding(model=openai_embed_model, api_key=openai_api_key)

    return HuggingFaceEmbedding(model_name=local_embed_model)


@st.cache_resource(show_spinner=False)
def get_llm(openai_api_key: str, openai_chat_model: str):
    if openai_api_key:
        return OpenAI(model=openai_chat_model, api_key=openai_api_key, temperature=0.1)
    return MockLLM(max_tokens=512)


@st.cache_resource(show_spinner=False)
def build_index(
    collection_name: str,
    files: tuple[tuple[str, bytes], ...],
    embedding_provider: str,
    openai_api_key: str,
    openai_embed_model: str,
    openai_chat_model: str,
    local_embed_model: str,
    chunk_size: int,
    chunk_overlap: int,
):
    documents: list[Document] = []
    for file_name, file_bytes in files:
        documents.extend(read_pdf_pages(file_bytes, file_name))

    if not documents:
        raise ValueError("No selectable text was found. Try a text-based PDF rather than a scanned PDF.")

    Settings.embed_model = get_embedding_model(
        embedding_provider,
        openai_api_key,
        openai_embed_model,
        local_embed_model,
    )
    Settings.llm = get_llm(openai_api_key, openai_chat_model)
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing_collections = [
        collection.name if hasattr(collection, "name") else str(collection)
        for collection in chroma_client.list_collections()
    ]
    if collection_name in existing_collections:
        chroma_client.delete_collection(collection_name)

    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    return index, len(documents)


def source_label(node, fallback_number: int) -> str:
    label = node.metadata.get("source_label")
    return str(label) if label else f"Source {fallback_number}"


def retrieved_rows(source_nodes) -> list[dict[str, str]]:
    rows = []
    for index, node in enumerate(source_nodes, start=1):
        rows.append(
            {
                "number": str(index),
                "source": source_label(node.node, index),
                "score": f"{node.score:.3f}" if node.score is not None else "n/a",
                "text": node.node.get_content().strip(),
            }
        )
    return rows


def retrieval_only_answer(rows: list[dict[str, str]]) -> str:
    lines = [
        "No OpenAI API key is configured for answer generation, so ResearchGPT is showing the most relevant evidence instead."
    ]
    for row in rows:
        excerpt = row["text"].replace("\n", " ")[:550]
        lines.append(f"[{row['number']}] {row['source']}: {excerpt}")
    return "\n\n".join(lines)


def render_sidebar() -> RagSettings:
    st.sidebar.header("Settings")

    openai_api_key = st.sidebar.text_input(
        "OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )
    embedding_provider = st.sidebar.radio("Embedding provider", ["OpenAI", "Local Hugging Face"])
    openai_embed_model = st.sidebar.text_input(
        "OpenAI embedding model",
        value=os.getenv("OPENAI_EMBED_MODEL", DEFAULT_OPENAI_EMBED_MODEL),
    )
    openai_chat_model = st.sidebar.text_input(
        "OpenAI chat model",
        value=os.getenv("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL),
    )
    local_embed_model = st.sidebar.text_input(
        "Local embedding model",
        value=os.getenv("LOCAL_EMBED_MODEL", DEFAULT_LOCAL_EMBED_MODEL),
    )
    chunk_size = st.sidebar.slider("Chunk size", min_value=256, max_value=1536, value=768, step=128)
    chunk_overlap = st.sidebar.slider("Chunk overlap", min_value=0, max_value=300, value=120, step=20)
    top_k = st.sidebar.slider("Retrieved chunks", min_value=2, max_value=8, value=4)

    return RagSettings(
        embedding_provider=embedding_provider,
        openai_api_key=openai_api_key,
        openai_embed_model=openai_embed_model,
        openai_chat_model=openai_chat_model,
        local_embed_model=local_embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    settings = render_sidebar()

    uploaded_files = st.file_uploader(
        "Upload research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more PDFs to build a searchable RAG index.")
        return

    files = tuple((uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in uploaded_files)
    collection_name = make_collection_name(files, settings)

    with st.spinner("Loading pages, chunking text, embedding content, and creating the vector index..."):
        try:
            index, page_count = build_index(
                collection_name,
                files,
                settings.embedding_provider,
                settings.openai_api_key,
                settings.openai_embed_model,
                settings.openai_chat_model,
                settings.local_embed_model,
                settings.chunk_size,
                settings.chunk_overlap,
            )
        except Exception as exc:
            st.error(str(exc))
            return

    st.success(f"Indexed {len(files)} PDF file(s) across {page_count} text-bearing page(s).")

    query = st.text_input(
        "Ask a question about the uploaded paper",
        placeholder="What is the main contribution, and what evidence supports it?",
    )
    if not query:
        return

    with st.spinner("Retrieving relevant chunks and preparing the answer..."):
        if settings.openai_api_key:
            query_engine = index.as_query_engine(
                similarity_top_k=settings.top_k,
                response_mode="compact",
                text_qa_template=CITATION_PROMPT,
            )
            response = query_engine.query(query)
            answer = str(response)
            rows = retrieved_rows(response.source_nodes)
        else:
            retriever = index.as_retriever(similarity_top_k=settings.top_k)
            rows = retrieved_rows(retriever.retrieve(query))
            answer = retrieval_only_answer(rows)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for row in rows:
        st.markdown(f"**[{row['number']}] {row['source']}** - similarity `{row['score']}`")

    with st.expander("Show retrieved chunks"):
        for row in rows:
            st.markdown(f"**Chunk {row['number']}: {row['source']}**")
            st.write(row["text"])


if __name__ == "__main__":
    main()
