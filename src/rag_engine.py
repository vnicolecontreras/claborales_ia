import os
import re
from collections import Counter
from pathlib import Path

from langchain_openai import ChatOpenAI
from pypdf import PdfReader


class RagEngine:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(self.pdf_path)

        self.document_name = self.pdf_path.name
        self.raw_text = self._read_pdf()
        self.chunks = self._split_text(self.raw_text)
        if not self.chunks:
            raise ValueError("No se pudo extraer texto del PDF.")

        self.chunk_tokens = [self._tokenize(chunk) for chunk in self.chunks]
        self.llm = self._build_llm()

    def _read_pdf(self) -> str:
        reader = PdfReader(str(self.pdf_path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    def _split_text(self, text: str) -> list[str]:
        clean_text = re.sub(r"\r", "\n", text)
        pieces = re.split(r"(?=P\.\d{3})", clean_text)
        chunks = []

        for piece in pieces:
            piece = re.sub(r"\n{2,}", "\n", piece).strip()
            if len(piece) >= 120:
                chunks.append(piece)

        if chunks:
            return chunks

        fallback = []
        step = 1800
        for i in range(0, len(clean_text), step):
            part = clean_text[i:i + step].strip()
            if part:
                fallback.append(part)
        return fallback

    def _tokenize(self, text: str) -> Counter:
        words = re.findall(r"\w+", text.lower())
        return Counter(words)

    def _score(self, question: str, chunk_counter: Counter) -> float:
        question_words = re.findall(r"\w+", question.lower())
        if not question_words:
            return 0.0

        score = 0
        for word in question_words:
            score += chunk_counter.get(word, 0)
        return float(score)

    def _build_llm(self):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("GITHUB_BASE_URL")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if not api_key:
            return None

        kwargs = {
            "api_key": api_key,
            "model": model,
            "temperature": 0,
        }

        if base_url:
            kwargs["base_url"] = base_url

        return ChatOpenAI(**kwargs)

    def _retrieve(self, question: str, top_k: int = 3) -> list[dict]:
        scored = []
        for index, tokens in enumerate(self.chunk_tokens):
            score = self._score(question, tokens)
            scored.append((index, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[:top_k]

        results = []
        for index, score in best:
            chunk = self.chunks[index]
            title = chunk.splitlines()[0][:120]
            results.append(
                {
                    "title": title if title else f"Fragmento {index + 1}",
                    "text": chunk,
                    "score": float(score),
                }
            )
        return results
    
    def _extract_legal_source(self, text: str) -> str:
        for line in text.splitlines():
            if "FUENTE:" in line.upper():
                return line.strip()
        return "Fuente legal no identificada de forma explicita en el fragmento."

    def _build_prompt(self, question: str, contexts: list[dict]) -> str:
        context_text = "\n\n".join(
            [f"[Fragmento {i + 1}]\n{item['text']}" for i, item in enumerate(contexts)]
        )
        return (
            "Eres un asistente de consultas laborales chilenas.\n"
            "Responde usando exclusivamente la informacion entregada en el contexto.\n"
            "Si el contexto no contiene la respuesta, di claramente que no puedes responder con certeza.\n"
            "Si encuentras fuente legal dentro del contexto, mencionarla al final.\n\n"
            f"Pregunta: {question}\n\n"
            f"Contexto:\n{context_text}"
        )

    def ask(self, question: str, top_k: int = 3) -> dict:
        contexts = self._retrieve(question, top_k=top_k)
        prompt = self._build_prompt(question, contexts)
        has_evidence = bool(contexts) and contexts[0]["score"] > 0

        if not has_evidence:
            answer = (
                "No encontre evidencia suficiente en el documento para responder con certeza. "
                "Intenta reformular la pregunta o usar una consulta mas especifica."
            )
            mode = "no-evidence"
        elif self.llm is not None:
            response = self.llm.invoke(prompt)
            answer = response.content
            mode = "llm"
        else:
            source_title = contexts[0]["title"]
            first = contexts[0]["text"]
            answer = (
                "No hay credenciales de modelo configuradas. "
                "Te dejo el fragmento mas parecido encontrado en el documento.\n\n"
                f"Fuente principal recuperada: {source_title}\n\n"
                f"{first[:1500]}"
            )
            mode = "retrieval-only"

        return {
            "answer": answer,
            "sources": contexts,
            "prompt": prompt,
            "mode": mode,
            "top_source": contexts[0]["title"] if contexts else "Sin fuente",
            "legal_source": self._extract_legal_source(contexts[0]["text"]) if contexts else "Sin fuente legal",
        }
        