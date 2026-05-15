import streamlit as st


st.set_page_config(page_title="About ResearchGPT", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(180deg, #f6f8f2 0%, #f8fafc 48%, #eef4f1 100%);
        color: #10201b;
    }
    [data-testid="stSidebar"] {
        background: #edf3ef;
        border-right: 1px solid #d6e0d8;
    }
    .block-container {
        padding-top: 2.4rem;
        max-width: 1080px;
    }
    .about-hero {
        border: 1px solid #c9d8cf;
        border-radius: 10px;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(237, 245, 240, 0.94)),
            repeating-linear-gradient(90deg, transparent 0, transparent 31px, rgba(36, 93, 81, 0.06) 32px);
        padding: 1.65rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 26px rgba(15, 46, 36, 0.06);
    }
    .about-hero h1 {
        color: #10201b;
        margin: 0 0 0.4rem 0;
    }
    .about-hero p {
        color: #4c6259;
        margin: 0;
        max-width: 760px;
    }
    .info-block {
        border: 1px solid #d6e0d8;
        border-radius: 8px;
        background: #ffffff;
        padding: 1rem;
        height: 100%;
        box-shadow: 0 8px 18px rgba(15, 46, 36, 0.04);
    }
    .info-block h3 {
        color: #16352b;
        margin-top: 0;
    }
    .info-block p, .info-block li {
        color: #5f7169;
    }
    .process-step {
        border-top: 3px solid #9bb7a5;
        background: rgba(255, 255, 255, 0.88);
        border-radius: 8px;
        padding: 0.85rem;
        min-height: 128px;
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
    st.markdown('<div class="process-step"><strong>Load</strong><p>PyMuPDF extracts text from uploaded PDFs page by page.</p></div>', unsafe_allow_html=True)
with steps[1]:
    st.markdown('<div class="process-step"><strong>Index</strong><p>LlamaIndex chunks the text and stores vectors in ChromaDB.</p></div>', unsafe_allow_html=True)
with steps[2]:
    st.markdown('<div class="process-step"><strong>Retrieve</strong><p>The question is matched against the most relevant document chunks.</p></div>', unsafe_allow_html=True)
with steps[3]:
    st.markdown('<div class="process-step"><strong>Answer</strong><p>Gemini, Groq, OpenAI, or retrieval-only mode produces an evidence-based response.</p></div>', unsafe_allow_html=True)

st.markdown("### Provider Recommendation")
st.write(
    "For low/no-cost demos, use Local Hugging Face embeddings plus Google Gemini or Groq for answer generation. "
    "This keeps embedding costs at zero and avoids relying on an OpenAI quota."
)

st.markdown("### Limitations")
st.write(
    "Scanned PDFs still need OCR, complex tables may need specialized parsing, and answers should be checked against the retrieved chunks."
)
