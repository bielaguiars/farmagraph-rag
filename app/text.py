import hashlib
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


@dataclass
class Chunk:
    indice: int
    texto: str
    secao: str = ""


def extract_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Formato não suportado: {extension}. Use TXT, PDF ou DOCX.")
    if extension == ".pdf":
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
    elif extension == ".docx":
        from docx import Document

        text = "\n".join(p.text for p in Document(BytesIO(content)).paragraphs)
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("O arquivo TXT deve estar em UTF-8.") from exc
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise ValueError("Não foi possível extrair texto (PDF digitalizado precisa de OCR).")
    return text


def document_id(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]


def document_title(text: str, filename: str) -> str:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first[:160] or Path(filename).stem


def document_type(text: str) -> str:
    # prontuário primeiro: "AMBULATORIAL" contém a substring "BULA"
    head = text[:200].upper()
    if "PRONTUÁRIO" in head or "PRONTUARIO" in head:
        return "prontuario"
    if re.search(r"\bBULA\b", head):
        return "bula"
    return "documento"


SECTION_HEADER = re.compile(r"^[A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\n.!?:]{2,60}$")


def split_sections(text: str) -> list[Chunk]:
    """Bulas seguem seções fixas (RDC ANVISA 47/2009): um trecho por seção, com o título."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = [("Identificação", [])]
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if SECTION_HEADER.match(stripped) and not stripped.isupper() and len(stripped.split()) <= 6:
            sections.append((stripped, []))
        else:
            sections[-1][1].append(stripped)
    chunks = []
    for title, body in sections:
        if not body:
            continue
        chunks.append(Chunk(indice=len(chunks), texto=f"{title}\n" + " ".join(body), secao=title))
    return chunks


def document_subject(text: str, doc_type: str) -> str:
    """Assunto principal (ajuda o LLM a resolver 'este medicamento' / 'o paciente')."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if doc_type == "bula" and lines:
        match = re.search(r"[—\-–]\s*(.+)$", lines[0])
        if match:
            return match.group(1).strip().lower()
    if doc_type == "prontuario":
        for line in lines[1:]:
            match = re.match(r"([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+)+)", line)
            if match and not line.upper().startswith("PACIENTE FICT"):
                return match.group(1)
    return ""


def split_into_chunks(text: str, max_chars: int = 900) -> list[Chunk]:
    """Divide por parágrafos, agrupando-os até max_chars (seções da bula ficam inteiras)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[A-ZÁÉÍÓÚ][^\n]{0,40}\n)", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        else:
            sentences = [paragraph]
        for piece in sentences:
            candidate = f"{current}\n{piece}".strip() if current else piece
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return [Chunk(indice=i, texto=c) for i, c in enumerate(chunks)]
