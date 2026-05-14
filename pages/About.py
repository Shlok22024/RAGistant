import streamlit as st


st.set_page_config(page_title="About ResearchGPT", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.4rem;
        max-width: 1080px;
    }
    .about-hero {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        background: #ffffff;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
    }
    .about-hero h1 {
        color: #0f172a;
        margin: 0 0 0.4rem 0;
    }
    .about-hero p {
        color: #475569;
        margin: 0;
        max-width: 760px;
    }
    .info-block {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background: #ffffff;
        padding: 1rem;
        height: 100%;
    }
    .info-block h3 {
        color: #0f172a;
        margin-top: 0;
    }
    .info-block p, .info-block li {
        color: #475569;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="about-hero">
        <h1>About ResearchGPT</h1>
        <p>
            ResearchGPT is a portfolio RAG application for asking questions over uploaded
            research papers and checking the source chunks behind each answer.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")

with left:
    st.markdown(
        """
        <div class="info-block">
            <h3>Developer</h3>
            <p><strong>Shlok Goud</strong></p>
            <p>
                This project demonstrates practical applied AI skills: document parsing,
                chunking, vector search, prompt design, source-grounded answer generation,
                and product-facing UI design.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="info-block">
            <h3>Design Goal</h3>
            <p>
                The app is intentionally built as a working research tool first, not a
                landing page. The main page keeps the user focused on one flow:
                upload, retrieve, answer, and inspect citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### How The App Works")

steps = st.columns(4)
with steps[0]:
    st.markdown("**Load**")
    st.write("PyMuPDF extracts text from uploaded PDFs page by page.")
with steps[1]:
    st.markdown("**Index**")
    st.write("LlamaIndex chunks the text and stores vectors in ChromaDB.")
with steps[2]:
    st.markdown("**Retrieve**")
    st.write("The question is matched against the most relevant document chunks.")
with steps[3]:
    st.markdown("**Answer**")
    st.write("Gemini, Groq, OpenAI, or retrieval-only mode produces an evidence-based response.")

st.markdown("### Provider Recommendation")
st.write(
    "For low/no-cost demos, use Local Hugging Face embeddings plus Google Gemini or Groq for answer generation. "
    "This keeps embedding costs at zero and avoids relying on an OpenAI quota."
)

st.markdown("### Limitations")
st.write(
    "Scanned PDFs still need OCR, complex tables may need specialized parsing, and answers should be checked against the retrieved chunks."
)
