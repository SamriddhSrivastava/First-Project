from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 180


class VectorStoreError(Exception):
    """Raised when embeddings or the FAISS vector database fail."""


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Create the local sentence-transformer embedding model used by FAISS."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """Split extracted PDF pages into retrieval-friendly overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def create_or_update_vector_store(documents: list[Document], persist_path: Path) -> FAISS:
    """Build a local FAISS index from documents and save it to disk."""
    if not documents:
        raise VectorStoreError("No document text was available for indexing.")

    try:
        chunks = split_documents(documents)
        embeddings = get_embedding_model()
        vector_store = FAISS.from_documents(chunks, embeddings)
        persist_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(persist_path))
        return vector_store
    except Exception as exc:
        raise VectorStoreError(
            "Could not create the FAISS vector database. Check dependencies and document text."
        ) from exc


def load_vector_store(persist_path: Path) -> FAISS:
    """Load the persisted FAISS index from disk."""
    index_file = persist_path / "index.faiss"
    metadata_file = persist_path / "index.pkl"
    if not index_file.exists() or not metadata_file.exists():
        raise VectorStoreError("Vector database not found. Please upload PDFs first.")

    try:
        return FAISS.load_local(
            str(persist_path),
            get_embedding_model(),
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        raise VectorStoreError("Could not load the saved FAISS vector database.") from exc
