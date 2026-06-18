import os
import shutil
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.pdf_loader import PDFProcessingError, load_uploaded_pdfs
from utils.rag_pipeline import RAGPipeline, RAGPipelineError
from utils.vector_store import VectorStoreError, create_or_update_vector_store, load_vector_store


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = DATA_DIR / "faiss_index"

UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


st.set_page_config(
    page_title="AI College Information Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_custom_styles() -> None:
    """Apply small CSS refinements for a clean Streamlit chat experience."""
    dark_mode = st.session_state.get("dark_mode", False)
    sidebar_background = "#111827" if dark_mode else "#f7f9fc"
    page_text = "#f8fafc" if dark_mode else "#172033"
    muted_text = "#cbd5e1" if dark_mode else "#5c667a"
    card_background = "#1f2937" if dark_mode else "#ffffff"
    border_color = "#334155" if dark_mode else "#e5eaf2"
    source_background = "#1e3a5f" if dark_mode else "#eef4ff"
    source_border = "#2f5d8c" if dark_mode else "#d9e7ff"
    source_text = "#dbeafe" if dark_mode else "#1d4f91"

    st.markdown(
        f"""
        <style>
            .main .block-container {{
                max-width: 980px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }}
            [data-testid="stSidebar"] {{
                background: {sidebar_background};
                border-right: 1px solid {border_color};
            }}
            .app-title {{
                font-size: 2.1rem;
                font-weight: 750;
                margin-bottom: 0.2rem;
                color: {page_text};
            }}
            .app-subtitle {{
                color: {muted_text};
                margin-bottom: 1.4rem;
                font-size: 1rem;
            }}
            .source-pill {{
                display: inline-block;
                background: {source_background};
                border: 1px solid {source_border};
                color: {source_text};
                border-radius: 999px;
                padding: 0.18rem 0.65rem;
                margin: 0.15rem 0.25rem 0.15rem 0;
                font-size: 0.82rem;
            }}
            .metric-box {{
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 0.8rem;
                background: {card_background};
            }}
            .stButton > button {{
                border-radius: 8px;
                border: 1px solid #d8deea;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Create Streamlit session variables used by the chatbot."""
    defaults = {
        "messages": [],
        "uploaded_documents": [],
        "vector_store_ready": FAISS_DIR.exists(),
        "last_processed_files": set(),
        "dark_mode": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat() -> None:
    """Clear chat messages while keeping uploaded documents and vector data."""
    st.session_state.messages = []
    st.toast("Chat history cleared.")


def clear_knowledge_base() -> None:
    """Remove local uploads, FAISS data, document list, and current chat history."""
    if FAISS_DIR.exists():
        shutil.rmtree(FAISS_DIR)

    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.name != ".gitkeep" and file_path.is_file():
            file_path.unlink()

    st.session_state.messages = []
    st.session_state.uploaded_documents = []
    st.session_state.last_processed_files = set()
    st.session_state.vector_store_ready = False
    st.toast("Local knowledge base cleared.")


def render_sources(sources: list[dict]) -> None:
    """Render source metadata below an assistant answer."""
    if not sources:
        return

    unique_sources = []
    seen = set()
    for source in sources:
        label = f"{source.get('source', 'Unknown')} - page {source.get('page', 'N/A')}"
        if label not in seen:
            seen.add(label)
            unique_sources.append(label)

    source_html = "".join(f"<span class='source-pill'>{item}</span>" for item in unique_sources)
    st.markdown(source_html, unsafe_allow_html=True)


def build_chat_history_export() -> str:
    """Return the current chat history in a simple markdown format."""
    lines = ["# AI-Powered College Information Assistant Chat History", ""]
    for message in st.session_state.messages:
        role = "Student" if message["role"] == "user" else "Assistant"
        lines.append(f"## {role}")
        lines.append(message["content"])
        if message.get("sources"):
            lines.append("")
            lines.append("Sources:")
            for source in message["sources"]:
                lines.append(f"- {source.get('source', 'Unknown')} page {source.get('page', 'N/A')}")
        lines.append("")
    return "\n".join(lines)


def process_uploaded_files(uploaded_files) -> None:
    """Save uploaded PDFs, extract text, chunk content, and persist FAISS index."""
    if not uploaded_files:
        return

    current_file_signature = {(file.name, getattr(file, "size", 0)) for file in uploaded_files}
    current_file_names = {file.name for file in uploaded_files}
    if current_file_signature == st.session_state.last_processed_files and st.session_state.vector_store_ready:
        return

    with st.spinner("Reading PDFs and building the college knowledge base..."):
        try:
            documents = load_uploaded_pdfs(uploaded_files, UPLOAD_DIR)
            if not documents:
                st.warning("No readable text was found in the uploaded PDFs.")
                return

            create_or_update_vector_store(documents=documents, persist_path=FAISS_DIR)
            st.session_state.uploaded_documents = sorted(current_file_names)
            st.session_state.last_processed_files = current_file_signature
            st.session_state.vector_store_ready = True
            st.success(f"Processed {len(current_file_names)} PDF document(s).")
        except (PDFProcessingError, VectorStoreError) as exc:
            st.session_state.vector_store_ready = False
            st.error(str(exc))


def answer_question(question: str) -> None:
    """Run retrieval and Gemini generation, then append the result to chat state."""
    try:
        vector_store = load_vector_store(FAISS_DIR)
        pipeline = RAGPipeline(vector_store=vector_store)

        with st.spinner("Searching documents and drafting an answer..."):
            result = pipeline.answer(question, chat_history=st.session_state.messages)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "created_at": time.time(),
            }
        )
    except (VectorStoreError, RAGPipelineError) as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"I could not generate an answer yet. {exc}",
                "sources": [],
                "created_at": time.time(),
            }
        )


def main() -> None:
    initialize_session_state()
    apply_custom_styles()

    with st.sidebar:
        st.header("Document Library")
        st.caption("Upload prospectuses, regulations, hostel manuals, notices, and placement brochures.")

        st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="You can upload multiple college documents at once.",
        )

        process_uploaded_files(uploaded_files)

        st.divider()
        st.subheader("Uploaded Documents")
        if st.session_state.uploaded_documents:
            for document_name in st.session_state.uploaded_documents:
                st.write(f"📄 {document_name}")
        else:
            st.info("No PDFs uploaded in this session.")

        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            reset_chat()

        if st.session_state.messages:
            st.download_button(
                "Download Chat History",
                data=build_chat_history_export(),
                file_name="college_assistant_chat_history.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with st.expander("Admin dashboard"):
            st.caption("Manage the local document index for this project demo.")
            st.write(f"Indexed documents: {len(st.session_state.uploaded_documents)}")
            if st.button("Clear Knowledge Base", use_container_width=True):
                clear_knowledge_base()
                st.rerun()

        st.divider()
        st.caption("Gemini API status")
        if os.getenv("GOOGLE_API_KEY"):
            st.success("API key loaded")
        else:
            st.error("Add GOOGLE_API_KEY to .env")

    st.markdown("<div class='app-title'>AI-Powered College Information Assistant</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-subtitle'>Ask questions from uploaded college PDFs and get source-backed answers.</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.vector_store_ready:
        st.info("Upload one or more PDF documents from the sidebar to start chatting.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    prompt = st.chat_input("Ask about admissions, fees, hostel rules, placements, exams, or notices...")

    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": [], "created_at": time.time()}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.vector_store_ready:
            warning = "Please upload and process at least one PDF before asking questions."
            st.session_state.messages.append(
                {"role": "assistant", "content": warning, "sources": [], "created_at": time.time()}
            )
            with st.chat_message("assistant"):
                st.warning(warning)
            return

        with st.chat_message("assistant"):
            answer_question(prompt)
            latest_message = st.session_state.messages[-1]
            st.markdown(latest_message["content"])
            render_sources(latest_message.get("sources", []))


if __name__ == "__main__":
    main()
