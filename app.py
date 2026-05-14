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
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


load_dotenv()

APP_TITLE = "ResearchGPT"
APP_SUBTITLE = "RAG Based Research Paper Assistant"
CHROMA_DIR = Path(".chroma")

DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

GENERATION_PROVIDERS = ["Google Gemini", "Groq", "OpenAI", "Retrieval only"]
EMBEDDING_PROVIDERS = ["Local Hugging Face", "OpenAI"]


@dataclass(frozen=True)
class RagSettings:
    embedding_provider: str
    answer_provider: str
    openai_api_key: str
    gemini_api_key: str
    groq_api_key: str
    openai_embed_model: str
    openai_chat_model: str
    gemini_model: str
    groq_model: str
    local_embed_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 4rem;
            max-width: 1180px;
        }
        .rg-hero {
            border: 1px solid #e2e8f0;
            background: #ffffff;
            border-radius: 10px;
            padding: 1.35rem 1.45rem;
            margin-bottom: 1.2rem;
        }
        .rg-eyebrow {
            color: #2563eb;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .rg-hero h1 {
            color: #0f172a;
            font-size: 2.35rem;
            line-height: 1.1;
            margin: 0 0 0.45rem 0;
        }
        .rg-hero p {
            color: #475569;
            font-size: 1rem;
            margin: 0;
            max-width: 780px;
        }
        .rg-step {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            background: #ffffff;
            padding: 1rem;
            min-height: 116px;
        }
        .rg-step strong {
            color: #0f172a;
            display: block;
            margin-bottom: 0.35rem;
        }
        .rg-step span {
            color: #64748b;
            font-size: 0.94rem;
        }
        .rg-source {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.55rem;
            background: #ffffff;
        }
        .rg-muted {
            color: #64748b;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
            raise ValueError("Add an OpenAI API key or switch embeddings to Local Hugging Face.")
        return OpenAIEmbedding(model=openai_embed_model, api_key=openai_api_key)

    return HuggingFaceEmbedding(model_name=local_embed_model)


@st.cache_resource(show_spinner=False)
def build_index(
    collection_name: str,
    files: tuple[tuple[str, bytes], ...],
    embedding_provider: str,
    openai_api_key: str,
    openai_embed_model: str,
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
    Settings.llm = MockLLM(max_tokens=512)
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


def build_context(rows: list[dict[str, str]]) -> str:
    context_blocks = []
    for row in rows:
        context_blocks.append(
            f"[{row['number']}] Source: {row['source']}\n"
            f"Similarity: {row['score']}\n"
            f"Text:\n{row['text']}"
        )
    return "\n\n---\n\n".join(context_blocks)


def answer_prompt(query: str, rows: list[dict[str, str]]) -> str:
    return (
        "You are ResearchGPT, a careful research-paper assistant.\n"
        "Answer only from the provided context. Cite evidence inline using the bracketed "
        "source numbers already shown in the context, such as [1] or [2]. If the context "
        "does not contain enough evidence, say that the uploaded paper does not provide "
        "enough information.\n\n"
        f"Context:\n{build_context(rows)}\n\n"
        f"Question: {query}\n\n"
        "Answer with concise reasoning and citations."
    )


def retrieval_only_answer(rows: list[dict[str, str]]) -> str:
    lines = [
        "ResearchGPT is in retrieval-only mode, so it is showing the most relevant evidence instead of generating a summary."
    ]
    for row in rows:
        excerpt = row["text"].replace("\n", " ")[:600]
        lines.append(f"[{row['number']}] {row['source']}: {excerpt}")
    return "\n\n".join(lines)


def generate_with_gemini(prompt: str, api_key: str, model: str) -> str:
    if not api_key:
        raise ValueError("Add a Gemini API key in the sidebar or choose Retrieval only.")

    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or "Gemini returned an empty response."


def generate_with_groq(prompt: str, api_key: str, model: str) -> str:
    if not api_key:
        raise ValueError("Add a Groq API key in the sidebar or choose Retrieval only.")

    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You answer research questions using only provided context."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or "Groq returned an empty response."


def generate_with_openai(prompt: str, api_key: str, model: str) -> str:
    if not api_key:
        raise ValueError("Add an OpenAI API key in the sidebar or choose another answer provider.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You answer research questions using only provided context."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or "OpenAI returned an empty response."


def generate_answer(query: str, rows: list[dict[str, str]], settings: RagSettings) -> str:
    if settings.answer_provider == "Retrieval only":
        return retrieval_only_answer(rows)

    prompt = answer_prompt(query, rows)
    if settings.answer_provider == "Google Gemini":
        return generate_with_gemini(prompt, settings.gemini_api_key, settings.gemini_model)
    if settings.answer_provider == "Groq":
        return generate_with_groq(prompt, settings.groq_api_key, settings.groq_model)
    return generate_with_openai(prompt, settings.openai_api_key, settings.openai_chat_model)


def render_sidebar() -> RagSettings:
    st.sidebar.header("Settings")

    embedding_provider = st.sidebar.radio(
        "Embedding provider",
        EMBEDDING_PROVIDERS,
        help="Local Hugging Face embeddings avoid paid embedding APIs.",
    )
    answer_provider = st.sidebar.radio(
        "Answer provider",
        GENERATION_PROVIDERS,
        help="Gemini and Groq are good low/no-cost alternatives to OpenAI for this project.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("API keys")
    gemini_api_key = st.sidebar.text_input(
        "Gemini API key",
        value=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
        type="password",
    )
    groq_api_key = st.sidebar.text_input(
        "Groq API key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
    )
    openai_api_key = st.sidebar.text_input(
        "OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Models")
    gemini_model = st.sidebar.text_input(
        "Gemini model",
        value=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    )
    groq_model = st.sidebar.text_input(
        "Groq model",
        value=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
    )
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

    st.sidebar.divider()
    st.sidebar.subheader("Retrieval")
    chunk_size = st.sidebar.slider("Chunk size", min_value=256, max_value=1536, value=768, step=128)
    chunk_overlap = st.sidebar.slider("Chunk overlap", min_value=0, max_value=300, value=120, step=20)
    top_k = st.sidebar.slider("Retrieved chunks", min_value=2, max_value=8, value=4)

    return RagSettings(
        embedding_provider=embedding_provider,
        answer_provider=answer_provider,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        groq_api_key=groq_api_key,
        openai_embed_model=openai_embed_model,
        openai_chat_model=openai_chat_model,
        gemini_model=gemini_model,
        groq_model=groq_model,
        local_embed_model=local_embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
    )


def render_intro(settings: RagSettings) -> None:
    st.markdown(
        f"""
        <div class="rg-hero">
            <div class="rg-eyebrow">Research paper Q&A with citations</div>
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}. Upload PDFs, retrieve the most relevant chunks, and generate answers with page-level source references.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            '<div class="rg-step"><strong>1. Upload</strong><span>Drop one or more text-based PDFs. Each page keeps its source metadata.</span></div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            '<div class="rg-step"><strong>2. Retrieve</strong><span>Local vectors in ChromaDB find the chunks most related to your question.</span></div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f'<div class="rg-step"><strong>3. Answer</strong><span>Current answer provider: {settings.answer_provider}. Sources stay visible for checking.</span></div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_styles()

    settings = render_sidebar()
    render_intro(settings)

    st.markdown("### Document Workspace")
    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        uploaded_files = st.file_uploader(
            "Upload research PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

    if not uploaded_files:
        with right_col:
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
                settings.local_embed_model,
                settings.chunk_size,
                settings.chunk_overlap,
            )
        except Exception as exc:
            st.error(str(exc))
            return

    with right_col:
        st.metric("PDFs indexed", len(files))
        st.metric("Text-bearing pages", page_count)
        st.metric("Retrieved chunks", settings.top_k)
        st.caption(f"Embeddings: {settings.embedding_provider} | Answers: {settings.answer_provider}")

    st.markdown("### Ask The Paper")
    query = st.text_area(
        "Question",
        placeholder="What is the main contribution, and what evidence supports it?",
        height=92,
    )

    ask_clicked = st.button("Generate answer", type="primary", use_container_width=False)
    if not ask_clicked or not query.strip():
        st.caption("Ask a focused question after the index is ready.")
        return

    with st.spinner("Retrieving relevant chunks and preparing the answer..."):
        try:
            retriever = index.as_retriever(similarity_top_k=settings.top_k)
            rows = retrieved_rows(retriever.retrieve(query))
            answer = generate_answer(query, rows, settings)
        except Exception as exc:
            st.error(str(exc))
            return

    answer_col, source_col = st.columns([1.35, 0.85], gap="large")
    with answer_col:
        st.markdown("### Answer")
        st.write(answer)

    with source_col:
        st.markdown("### Sources")
        for row in rows:
            st.markdown(
                f"""
                <div class="rg-source">
                    <strong>[{row['number']}] {row['source']}</strong><br>
                    <span class="rg-muted">Similarity {row['score']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Show retrieved chunks"):
        for row in rows:
            st.markdown(f"**Chunk {row['number']}: {row['source']}**")
            st.write(row["text"])


if __name__ == "__main__":
    main()
