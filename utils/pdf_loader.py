from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from pypdf import PdfReader


class PDFProcessingError(Exception):
    """Raised when uploaded PDFs cannot be saved or parsed."""


def save_uploaded_file(uploaded_file, upload_dir: Path) -> Path:
    """Persist a Streamlit uploaded file and return its saved path."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    destination = upload_dir / safe_name

    try:
        destination.write_bytes(uploaded_file.getbuffer())
    except OSError as exc:
        raise PDFProcessingError(f"Could not save {safe_name}: {exc}") from exc

    return destination


def extract_text_from_pdf(pdf_path: Path) -> list[Document]:
    """Extract page-wise text from a PDF and preserve source metadata."""
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise PDFProcessingError(f"Could not open {pdf_path.name}. Please upload a valid PDF.") from exc

    documents: list[Document] = []
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        cleaned_text = " ".join(text.split())
        if cleaned_text:
            documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata={"source": pdf_path.name, "page": page_index, "path": str(pdf_path)},
                )
            )

    return documents


def load_uploaded_pdfs(uploaded_files: Iterable, upload_dir: Path) -> list[Document]:
    """Save and extract text from multiple uploaded PDF files."""
    all_documents: list[Document] = []

    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith(".pdf"):
            continue

        saved_path = save_uploaded_file(uploaded_file, upload_dir)
        all_documents.extend(extract_text_from_pdf(saved_path))

    return all_documents
