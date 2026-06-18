import os

from langchain_google_genai import ChatGoogleGenerativeAI


DEFAULT_MODEL = "gemini-1.5-flash"
TOP_K = 5


class RAGPipelineError(Exception):
    """Raised when retrieval or answer generation fails."""


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline using FAISS and Gemini."""

    def __init__(self, vector_store):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RAGPipelineError("Missing GOOGLE_API_KEY in your .env file.")

        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
            google_api_key=api_key,
            temperature=0.2,
            convert_system_message_to_human=True,
        )

    def retrieve(self, question: str):
        """Return the most relevant chunks for the user question."""
        return self.vector_store.similarity_search(question, k=TOP_K)

    def answer(self, question: str, chat_history: list[dict] | None = None) -> dict:
        """Generate a grounded answer with source metadata."""
        try:
            docs = self.retrieve(question)
            if not docs:
                return {
                    "answer": "I could not find relevant information in the uploaded college documents.",
                    "sources": [],
                }

            context = self._format_context(docs)
            history = self._format_history(chat_history or [])
            prompt = self._build_prompt(question, context, history)
            response = self.llm.invoke(prompt)

            return {
                "answer": response.content,
                "sources": [doc.metadata for doc in docs],
            }
        except RAGPipelineError:
            raise
        except Exception as exc:
            raise RAGPipelineError("Gemini could not generate a response. Please try again.") from exc

    @staticmethod
    def _format_context(docs) -> str:
        """Create a source-labeled context block for the model."""
        blocks = []
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown document")
            page = doc.metadata.get("page", "N/A")
            blocks.append(f"[Source {index}: {source}, page {page}]\n{doc.page_content}")
        return "\n\n".join(blocks)

    @staticmethod
    def _format_history(messages: list[dict], limit: int = 6) -> str:
        """Keep the latest chat turns to support follow-up questions."""
        recent_messages = messages[-limit:]
        formatted = []
        for message in recent_messages:
            role = "Student" if message.get("role") == "user" else "Assistant"
            formatted.append(f"{role}: {message.get('content', '')}")
        return "\n".join(formatted)

    @staticmethod
    def _build_prompt(question: str, context: str, history: str) -> str:
        """Build a strict prompt that keeps answers grounded in documents."""
        return f"""
You are an AI-Powered College Information Assistant for students.
Answer only from the provided college document context. If the answer is not present, say that the uploaded documents do not contain enough information.

Guidelines:
- Be clear, concise, and student-friendly.
- Mention exact rules, dates, fees, eligibility, or procedures when available.
- Do not invent facts.
- End with a short "Sources used" line naming the relevant document and page numbers.

Recent conversation:
{history or "No previous conversation."}

Retrieved college document context:
{context}

Student question:
{question}

Answer:
""".strip()
