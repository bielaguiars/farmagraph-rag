"""Pipeline de ingestão: texto -> trechos -> NER/RE (LLM) -> resolução -> Neo4j (MERGE) -> algoritmos.

Uso em linha de comando:
    python -m app.ingest            # ingere a pasta corpus/
    python -m app.ingest --reset    # apaga o grafo e reingere tudo
"""

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from app.extraction import extract_from_chunk
from app.resolution import EntityResolver, normalize
from app.services import Services, build_services
from app.text import (
    document_id,
    document_subject,
    document_title,
    document_type,
    extract_text,
    split_into_chunks,
    split_sections,
)

Progress = Callable[[str, float], None]


def _entity_row(services: Services, label: str, canonical: str) -> dict:
    brands = services.resolver.brand_names.get(canonical, []) if label == "Medicamento" else []
    aliases = [canonical, *brands]
    return {
        "id": normalize(canonical),
        "nome": canonical,
        "nomes_comerciais": brands,
        "aliases_texto": " ".join(aliases),
    }


def ingest_document(services: Services, filename: str, content: bytes, *, force: bool = False,
                    progress: Progress | None = None) -> dict:
    started = time.time()
    text = extract_text(filename, content)
    doc_id = document_id(text)
    if not force and services.graph.document_exists(doc_id):
        return {"arquivo": filename, "status": "já ingerido", "documento_id": doc_id}

    doc_type = document_type(text)
    title = document_title(text, filename)
    subject = document_subject(text, doc_type)
    subject_label = {"bula": "Medicamento", "prontuario": "Paciente"}.get(doc_type, "")
    chunks = split_sections(text) if doc_type == "bula" else split_into_chunks(text, services.settings.chunk_size)

    entities: set[tuple[str, str]] = set()
    relations = []
    rejected: list[dict] = []
    resolutions: list[dict] = []
    corrected = 0
    chunk_entities: dict[int, set[tuple[str, str]]] = {}
    raw_outputs = []

    if subject_label and subject:
        resolved = services.resolver.resolve(subject, subject_label)
        entities.add((subject_label, resolved.canonical))
        subject = resolved.canonical

    for i, chunk in enumerate(chunks):
        if progress:
            progress(f"{filename}: LLM extraindo entidades e relações do trecho {i + 1}/{len(chunks)}",
                     0.05 + 0.7 * i / len(chunks))
        try:
            result, raw = extract_from_chunk(services.llm, services.resolver, chunk.texto, title, subject,
                                             subject_label, doc_type, chunk.secao)
        except RuntimeError as exc:
            rejected.append({"trecho": chunk.indice, "motivo": f"falha na extração: {exc}"})
            continue
        raw_outputs.append(raw)
        found = set(result.entidades)
        if subject_label and subject:
            found.add((subject_label, subject))
        chunk_entities[chunk.indice] = found
        entities |= found
        relations.extend((chunk.indice, rel) for rel in result.relacoes)
        rejected.extend({"trecho": chunk.indice, **r} for r in result.rejeitadas)
        resolutions.extend(result.resolucoes)
        corrected += result.corrigidas
        for label, name in found:
            services.resolver.register(label, name)

    if progress:
        progress(f"{filename}: gerando embeddings dos trechos", 0.8)
    vectors = services.llm.embed_documents([f"{title}\n{c.texto}" for c in chunks])

    if progress:
        progress(f"{filename}: gravando no Neo4j", 0.9)
    chunk_rows = [
        {"id": f"{doc_id}:{c.indice}", "indice": c.indice, "texto": c.texto, "embedding": v}
        for c, v in zip(chunks, vectors)
    ]
    services.graph.upsert_document(
        {"id": doc_id, "titulo": title, "tipo": doc_type, "arquivo": filename, "assunto": subject},
        chunk_rows,
    )
    by_label: dict[str, list[dict]] = {}
    for label, canonical in entities:
        by_label.setdefault(label, []).append(_entity_row(services, label, canonical))
    for label, rows in by_label.items():
        services.graph.merge_entities(label, rows)
    for indice, rel in relations:
        services.graph.merge_relation(rel.tipo, rel.origem, rel.destino, rel.propriedades, f"{doc_id}:{indice}")
    for c in chunks:
        mentioned = set(chunk_entities.get(c.indice, set()))
        mentioned |= {(r.label, r.canonical) for r in services.resolver.find_in_text(c.texto)}
        services.graph.merge_mentions(f"{doc_id}:{c.indice}", list(mentioned))
    services.graph.mark_document_ok(doc_id)

    return {
        "arquivo": filename,
        "status": "ok",
        "documento_id": doc_id,
        "tipo": doc_type,
        "trechos": len(chunks),
        "entidades": len(entities),
        "relacoes_aceitas": len(relations),
        "relacoes_brutas_llm": sum(len(r.get("relacoes", [])) for r in raw_outputs),
        "relacoes_rejeitadas": len([r for r in rejected if "relacao" in r or "origem" in r]),
        "motivos_rejeicao": dict(Counter(r.get("motivo", "?") for r in rejected)),
        "resolucoes": resolutions,
        "direcoes_corrigidas": corrected,
        "segundos": round(time.time() - started, 1),
    }


def ingest_folder(services: Services, folder: Path, *, force: bool = False, progress: Progress | None = None) -> list[dict]:
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".txt", ".pdf", ".docx"})
    reports = []
    for n, path in enumerate(files):
        def scoped(message: str, fraction: float, n=n) -> None:
            if progress:
                progress(message, (n + fraction) / max(len(files), 1))
        reports.append(ingest_document(services, path.name, path.read_bytes(), force=force, progress=scoped))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingere documentos no grafo FarmaGraph")
    parser.add_argument("pasta", nargs="?", default=None)
    parser.add_argument("--reset", action="store_true", help="apaga o grafo antes de ingerir")
    args = parser.parse_args()

    services = build_services()
    folder = Path(args.pasta) if args.pasta else services.settings.corpus_dir
    if args.reset:
        print("Apagando o grafo...")
        services.graph.reset()
        # o resolvedor foi carregado com os nomes do grafo antigo: recomeça só do vocabulário
        services.resolver = EntityResolver.from_vocabulary(services.settings.data_dir / "vocabulario.json")
    print("Criando constraints e índices...")
    services.graph.setup_schema(services.settings.embedding_dimension)
    services.refresh_resolver()

    started = time.time()
    reports = ingest_folder(services, folder, force=args.reset,
                            progress=lambda m, f: print(f"[{f:5.0%}] {m}", flush=True))
    print("Calculando PageRank e comunidades (Louvain)...")
    algo = services.graph.compute_algorithms()
    print(json.dumps(algo, ensure_ascii=False))
    out = services.settings.data_dir / "ultima_ingestao.json"
    out.write_text(json.dumps({"relatorios": reports, "algoritmos": algo}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(services.graph.stats(), ensure_ascii=False, indent=2))
    print(f"Concluído em {time.time() - started:.0f}s. Relatório: {out}")
    services.graph.close()


if __name__ == "__main__":
    main()
