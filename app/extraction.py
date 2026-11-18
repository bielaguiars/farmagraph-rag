"""Pipeline Text-to-Graph: NER + RE com LLM local, validados por restrições semânticas.

Mitigação de alucinação na extração (cada etapa é contabilizada no relatório de ingestão):
  1. saída estruturada (JSON Schema) com enums de rótulos e relações permitidos;
  2. extração guiada por seção: em bulas (seções fixas pela RDC ANVISA 47/2009) cada seção
     só pode gerar a relação correspondente (Indicações -> TRATA, Interações -> INTERAGE_COM...);
  3. relações permitidas por tipo de documento (prontuário só gera USA e TEM_CONDICAO);
  4. domínio/contradomínio do esquema (ex.: TRATA só liga Medicamento -> Condicao),
     com correção automática de direção invertida;
  5. ancoragem textual: as duas pontas precisam estar escritas no trecho (ou ser o assunto);
  6. em bulas, toda relação precisa envolver o medicamento da bula.
"""

import re
from dataclasses import dataclass, field

from app.ollama_client import OllamaClient
from app.resolution import EntityResolver, normalize
from app.schema import ENTITY_LABELS, RELATION_SCHEMA

DOC_TYPE_RELATIONS = {
    "bula": ["PERTENCE_A_CLASSE", "TRATA", "CONTRAINDICADO_EM", "PODE_CAUSAR", "INTERAGE_COM"],
    "prontuario": ["USA", "TEM_CONDICAO"],
    "documento": list(RELATION_SCHEMA),
}

# título da seção (normalizado) -> relação esperada. "contraindica" antes de "indica".
SECTION_RELATION = [
    ("contraindica", "CONTRAINDICADO_EM"),
    ("indica", "TRATA"),
    ("interac", "INTERAGE_COM"),
    ("reac", "PODE_CAUSAR"),
    ("advers", "PODE_CAUSAR"),
    ("efeitos", "PODE_CAUSAR"),
    ("classe", "PERTENCE_A_CLASSE"),
]

GUIDANCE = {
    "TRATA": "cada doença ou condição para a qual o medicamento é indicado (destino = nome da condição).",
    "CONTRAINDICADO_EM": "cada condição em que o medicamento não deve ser usado (destino = nome da condição, ex.: gravidez).",
    "PODE_CAUSAR": "cada reação adversa citada (destino = nome da reação).",
    "INTERAGE_COM": "um item para CADA medicamento citado que interage com ele (destino = nome do outro medicamento, "
                    "como escrito no texto, inclusive nome comercial), com gravidade ('alta' ou 'moderada') e efeito.",
    "PERTENCE_A_CLASSE": "a classe terapêutica do medicamento, exatamente como escrita no texto (destino = nome da classe).",
    "USA": "cada medicamento em uso pelo paciente (origem = paciente, destino = nome do medicamento sem dose).",
    "TEM_CONDICAO": "cada doença, diagnóstico ou condição do paciente (origem = paciente, destino = condição).",
}

GRAVIDADE = {"alta": "alta", "grave": "alta", "severa": "alta", "moderada": "moderada", "media": "moderada"}
SUBJECT_PRONOUNS = {"este medicamento", "o medicamento", "o farmaco", "a paciente", "o paciente", "paciente"}

# "pacientes com histórico de angioedema" -> "angioedema"
QUALIFIER_PREFIX = re.compile(
    r"^(?:em\s+)?(?:pacientes?|pessoas?|indiv[ií]duos?|mulheres|homens|casos?)\s+(?:com|que\s+t[eê]m|de)\s+"
    r"(?:hist[oó]rico\s+de\s+)?|^hist[oó]rico\s+de\s+",
    re.IGNORECASE,
)
LIST_SEPARATOR = re.compile(r"\s*,\s*|\s+e\s+|\s+ou\s+", re.IGNORECASE)
ADJECTIVE_ENDINGS = ("ica", "icas", "ico", "icos", "al", "ais", "ar", "ares", "osa", "oso", "aria", "ario", "iva", "ivo")


def split_entity_list(name: str) -> list[str]:
    """Separa listas que o LLM devolve como uma entidade só ("dor e febre" -> ["dor", "febre"]).

    Não separa quando a parte seguinte começa por adjetivo ("insuficiência renal e hepática")."""
    name = QUALIFIER_PREFIX.sub("", name.strip()).strip()
    parts = [p.strip() for p in LIST_SEPARATOR.split(name) if p and p.strip()]
    if len(parts) < 2:
        return [name]
    if any(normalize(p).split()[:1] and normalize(p).split()[0].endswith(ADJECTIVE_ENDINGS) for p in parts[1:]):
        return [name]
    return parts


def section_relation(section: str) -> str | None:
    key = normalize(section)
    for prefix, relation in SECTION_RELATION:
        if prefix in key:
            return relation
    return None


MAX_RELATIONS_PER_CHUNK = 10  # limita a lista: modelos pequenos às vezes repetem itens até estourar os tokens


def build_schema(relations: list[str]) -> dict:
    short = {"type": "string", "maxLength": 80}
    return {
        "type": "object",
        "properties": {
            "relacoes": {
                "type": "array",
                "maxItems": MAX_RELATIONS_PER_CHUNK,
                "items": {
                    "type": "object",
                    "properties": {
                        "origem": short,
                        "relacao": {"type": "string", "enum": relations},
                        "destino": short,
                        "gravidade": {"type": "string", "enum": ["alta", "moderada", ""]},
                        "efeito": short,
                    },
                    "required": ["origem", "relacao", "destino"],
                },
            },
        },
        "required": ["relacoes"],
    }


def build_prompt(chunk_text: str, title: str, subject: str, relations: list[str]) -> str:
    lines = [
        "Você extrai fatos de um documento farmacológico em português para um grafo de conhecimento.",
        f"Documento: {title}",
        f"Assunto principal: {subject or 'não informado'} (\"este medicamento\" ou \"o paciente\" refere-se a ele).",
        "Extraia SOMENTE fatos escritos no trecho, sem inventar e sem usar conhecimento externo. "
        "Use nomes curtos como aparecem no texto, sem dose. Se não houver o fato pedido, devolva lista vazia.",
        "Relações a extrair:",
    ]
    for relation in relations:
        src, dst, _ = RELATION_SCHEMA[relation]
        lines.append(f"- {relation} ({src} -> {dst}): {GUIDANCE[relation]}")
    lines.append('Formato: {"relacoes":[{"origem":"...","relacao":"...","destino":"...","gravidade":"","efeito":""}]}')
    lines.append(f'Trecho:\n"""{chunk_text}"""')
    return "\n".join(lines)


@dataclass
class ExtractedRelation:
    tipo: str
    origem: tuple[str, str]  # (rótulo, nome canônico)
    destino: tuple[str, str]
    propriedades: dict = field(default_factory=dict)


@dataclass
class ExtractionResult:
    entidades: list[tuple[str, str]] = field(default_factory=list)
    relacoes: list[ExtractedRelation] = field(default_factory=list)
    rejeitadas: list[dict] = field(default_factory=list)
    resolucoes: list[dict] = field(default_factory=list)
    corrigidas: int = 0
    brutas: int = 0


def _anchored(surface: str, canonical: str, label: str, chunk_norm: str, subject_norm: str,
              resolver: EntityResolver) -> bool:
    variants = {normalize(surface), normalize(canonical)}
    if label == "Medicamento":
        variants |= {normalize(b) for b in resolver.brand_names.get(canonical, [])}
    variants.discard("")
    if subject_norm and subject_norm in variants:
        return True
    return any(v in chunk_norm for v in variants)


def _text_spelling(canonical: str, chunk_text: str) -> str | None:
    import re

    target = normalize(canonical).split()
    if not target:
        return None
    words = re.findall(r"[\wÀ-ÿ-]+", chunk_text)
    for i in range(len(words) - len(target) + 1):
        window = words[i:i + len(target)]
        if [normalize(w) for w in window] == target:
            return " ".join(window).lower()
    return None


def _label_of(name: str, resolver: EntityResolver, subject: str, subject_label: str) -> str | None:
    if subject and normalize(name) in {normalize(subject), *SUBJECT_PRONOUNS}:
        return subject_label or None
    if resolver.is_medication(name):
        return "Medicamento"
    if resolver.resolve(name, "Paciente").method != "novo":
        return "Paciente"
    return None


def validate_extraction(raw: dict, chunk_text: str, subject: str, subject_label: str,
                        resolver: EntityResolver, allowed: list[str], doc_type: str) -> ExtractionResult:
    result = ExtractionResult()
    chunk_norm = normalize(chunk_text)
    subject_norm = normalize(subject)
    seen: set[tuple[str, str]] = set()
    logged: set[tuple[str, str]] = set()

    def resolve(surface: str, label: str) -> tuple[str, str] | None:
        if subject and normalize(surface) in SUBJECT_PRONOUNS:
            surface = subject
        res = resolver.resolve(surface, label)
        if not _anchored(surface, res.canonical, label, chunk_norm, subject_norm, resolver):
            return None
        if res.method == "novo" and label in {"Condicao", "EfeitoAdverso", "ClasseTerapeutica"}:
            spelled = _text_spelling(res.canonical, chunk_text)  # usa a grafia do próprio texto
            if spelled:
                res.canonical = spelled
        if normalize(surface) != normalize(res.canonical) and (surface, res.canonical) not in logged:
            logged.add((surface, res.canonical))
            result.resolucoes.append({"superficie": surface, "canonico": res.canonical, "rotulo": label, "metodo": res.method})
        key = (label, res.canonical)
        if key not in seen:
            seen.add(key)
            result.entidades.append(key)
        return key

    items = [i for i in raw.get("relacoes", []) if isinstance(i, dict)]
    result.brutas = len(items)
    for item in items:
        tipo = str(item.get("relacao", "")).strip().upper()
        origem = str(item.get("origem", "")).strip()
        destino = str(item.get("destino", "")).strip()
        if tipo not in RELATION_SCHEMA or tipo not in allowed or not origem or not destino:
            result.rejeitadas.append({**item, "motivo": "relação não permitida para este documento/seção"})
            continue
        src_label, dst_label, _ = RELATION_SCHEMA[tipo]
        # correção de direção: se as pontas vieram trocadas em relação ao domínio/contradomínio
        o_label = _label_of(origem, resolver, subject, subject_label)
        d_label = _label_of(destino, resolver, subject, subject_label)
        if src_label != dst_label and o_label == dst_label and d_label in {src_label, None} and o_label is not None:
            origem, destino = destino, origem
            o_label, d_label = d_label, o_label
            result.corrigidas += 1
        # em bula/prontuário, a origem das relações é sempre o assunto do documento
        if o_label != src_label and subject and subject_label == src_label and doc_type in {"bula", "prontuario"}:
            origem, o_label = subject, src_label
            result.corrigidas += 1
        if src_label in {"Medicamento", "Paciente"} and o_label != src_label:
            result.rejeitadas.append({**item, "motivo": f"origem não é {src_label}"})
            continue
        if dst_label == "Medicamento" and d_label != "Medicamento":
            result.rejeitadas.append({**item, "motivo": "destino não é Medicamento"})
            continue
        if dst_label not in {"Medicamento", "Paciente"} and d_label in {"Medicamento", "Paciente"}:
            result.rejeitadas.append({**item, "motivo": f"destino é {d_label}, esperado {dst_label}"})
            continue
        if normalize(destino) in {normalize(label) for label in ENTITY_LABELS}:
            result.rejeitadas.append({**item, "motivo": "destino é nome de tipo, não entidade"})
            continue
        src = resolve(origem, src_label)
        if src is None:
            result.rejeitadas.append({**item, "motivo": "entidade não ancorada no texto"})
            continue
        destinos = split_entity_list(destino) if dst_label in {"Condicao", "EfeitoAdverso"} else [destino]
        for dest in destinos:
            part = {**item, "destino": dest} if len(destinos) > 1 else item
            dst = resolve(dest, dst_label)
            if dst is None:
                result.rejeitadas.append({**part, "motivo": "entidade não ancorada no texto"})
                continue
            if src == dst:
                result.rejeitadas.append({**part, "motivo": "auto-relação"})
                continue
            if doc_type == "bula" and subject and subject_label == "Medicamento":
                subject_canonical = resolver.resolve(subject, "Medicamento").canonical
                if subject_canonical not in {src[1], dst[1]}:
                    result.rejeitadas.append({**part, "motivo": "relação não envolve o medicamento da bula"})
                    continue
            a, b = src, dst
            props = {}
            if tipo == "INTERAGE_COM":
                gravidade = GRAVIDADE.get(normalize(str(item.get("gravidade", ""))), "")
                efeito = str(item.get("efeito", "")).strip()
                if gravidade:
                    props["gravidade"] = gravidade
                if efeito:
                    props["efeito"] = resolver.resolve(efeito, "EfeitoAdverso").canonical
                if a[1] > b[1]:  # relação simétrica: grava em ordem canônica
                    a, b = b, a
            result.relacoes.append(ExtractedRelation(tipo, a, b, props))
    return result


def extract_from_chunk(llm: OllamaClient, resolver: EntityResolver, chunk_text: str, title: str,
                       subject: str, subject_label: str, doc_type: str, section: str = "") -> tuple[ExtractionResult, dict]:
    allowed = DOC_TYPE_RELATIONS.get(doc_type, DOC_TYPE_RELATIONS["documento"])
    if doc_type == "bula":
        relation = section_relation(section)
        if relation is None:  # seção sem relação de domínio (apresentação, posologia): não chama o LLM
            return ExtractionResult(), {}
        allowed = [relation]
    prompt = build_prompt(chunk_text, title, subject, allowed)
    raw = llm.generate_json(prompt, build_schema(allowed), num_predict=500, num_ctx=4096)
    return validate_extraction(raw, chunk_text, subject, subject_label, resolver, allowed, doc_type), raw
