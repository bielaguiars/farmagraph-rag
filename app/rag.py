"""Geração da resposta (KAG) com citações, verificação de fundamentação e nível de confiança.

Também expõe as linhas de base usadas na avaliação: LLM sem contexto e RAG só vetorial.
"""

import re
import time
from dataclasses import dataclass, field

from app.ollama_client import OllamaClient
from app.resolution import EntityResolver, normalize
from app.retrieval import HybridRetriever, RetrievalResult

DISCLAIMER = "Conteúdo acadêmico; não substitui avaliação médica ou farmacêutica."

ANSWER_PROMPT = """Você é um assistente farmacológico de apoio acadêmico. Responda em português do Brasil
usando SOMENTE o contexto abaixo.

Pergunta: {pergunta}

{contexto}

Instruções:
- {lead}
- Depois de cada afirmação, cite a fonte entre colchetes, ex.: [K1], [G3], [T2].
- Não mencione medicamentos, interações ou doenças que não estejam no contexto.
- Seja direto: no máximo 6 frases, terminando com uma orientação prática (ex.: discutir alternativa com o médico).

Resposta:"""

LEAD_WITH_ALERTS = ("Os ALERTAS DO GRAFO [K#] já respondem à pergunta: são conclusões verificadas por consultas "
                    "Cypher. Comece por eles, dizendo o medicamento em uso, a interação ou contraindicação, a gravidade "
                    "e o efeito.")
LEAD_WITHOUT_ALERTS = ("Responda com os FATOS DO GRAFO [G#] e os TRECHOS [T#]. Só se nada no contexto responder à "
                       "pergunta, diga: \"Não encontrei essa informação na base.\"")
NOT_FOUND = "Não encontrei essa informação na base."

# KAG: o raciocínio já foi feito no grafo; aqui o LLM só verbaliza as conclusões verificadas
VERBALIZE_PROMPT = """Você é um assistente farmacológico de apoio acadêmico. Os ALERTAS abaixo foram verificados no
grafo de conhecimento e respondem à pergunta. Explique-os em português do Brasil, em até 4 frases, sem acrescentar
nenhum medicamento, doença, efeito ou relação que não esteja escrito neles e sem contradizer nenhum alerta.
Cite cada alerta usado, ex.: [K1]. Termine com uma orientação prática.

Pergunta: {pergunta}

ALERTAS:
{alertas}

Resposta:"""


def ensure_alerts(text: str, r: RetrievalResult) -> str:
    """KAG: se o grafo concluiu algo (alertas K) e o LLM não usou, a conclusão verificada entra na resposta."""
    alerts = [a for a in r.alertas if a.get("tipo") != "alternativa"] or r.alertas
    if not alerts or re.search(r"\[K\d+\]", text):
        return text
    lead = " ".join(f"{a['texto']} [{a['ref']}]" for a in alerts[:3])
    rest = "" if NOT_FOUND in text else "\n\n" + text
    return f"Conclusão do grafo: {lead}{rest}"

VECTOR_ONLY_PROMPT = """Você é um assistente farmacológico de apoio acadêmico. Responda em português do Brasil
usando SOMENTE os trechos abaixo e cite-os como [T1], [T2]. Se não bastar, diga que não encontrou.
No máximo 7 frases.

{contexto}

Pergunta: {pergunta}
Resposta:"""

NO_CONTEXT_PROMPT = """Responda em português do Brasil, em no máximo 7 frases, à pergunta farmacológica abaixo.
Pergunta: {pergunta}
Resposta:"""


@dataclass
class Verification:
    citacoes: list[str] = field(default_factory=list)
    citacoes_validas: list[str] = field(default_factory=list)
    citacoes_invalidas: list[str] = field(default_factory=list)
    entidades_resposta: list[str] = field(default_factory=list)
    nao_fundamentadas: list[str] = field(default_factory=list)


@dataclass
class Answer:
    resposta: str
    recuperacao: RetrievalResult | None
    contexto: str
    verificacao: Verification
    confianca: float
    nivel: str
    componentes: dict
    segundos: float


def build_context(r: RetrievalResult, max_chunks: int = 3, chunk_chars: int = 650) -> str:
    sections = []
    if r.alertas:
        sections.append("ALERTAS DO GRAFO (raciocínio estruturado via Cypher):\n" +
                        "\n".join(f"[{a['ref']}] {a['texto']}" for a in r.alertas))
    if r.caminhos:
        sections.append("CAMINHOS MAIS CURTOS ENTRE ENTIDADES DA PERGUNTA:\n" +
                        "\n".join(f"- {p['texto']}" for p in r.caminhos))
    if r.fatos:
        facts = r.fatos[:12] if r.alertas else r.fatos  # com alertas, menos ruído para o LLM pequeno
        sections.append("FATOS DO GRAFO (subgrafo multi-hop):\n" + "\n".join(f"[{f['ref']}] {f['texto']}" for f in facts))
    if r.cypher_llm and r.cypher_llm.get("linhas"):
        rows = r.cypher_llm["linhas"][:15]
        sections.append("RESULTADO DO CYPHER GERADO PELO LLM:\n" +
                        "\n".join(f"[C{i + 1}] " + "; ".join(f"{k}: {v}" for k, v in row.items()) for i, row in enumerate(rows)))
    if r.trechos:
        sections.append("TRECHOS DOS DOCUMENTOS:\n" + "\n".join(
            f"[{t['ref']}] ({t['arquivo']}) {t['texto'][:chunk_chars]}" for t in r.trechos[:max_chunks]))
    return "\n\n".join(sections) or "Nenhum contexto encontrado."


def verify(answer: str, context: str, valid_refs: set[str], resolver: EntityResolver) -> Verification:
    """Checa citações e procura entidades na resposta que não aparecem no contexto (possível alucinação)."""
    v = Verification()
    v.citacoes = sorted(set(re.findall(r"\[([KGTC]\d+)\]", answer)))
    v.citacoes_validas = [c for c in v.citacoes if c in valid_refs]
    v.citacoes_invalidas = [c for c in v.citacoes if c not in valid_refs]
    context_norm = f" {normalize(context)} "
    for res in resolver.find_in_text(answer.replace(DISCLAIMER, "")):
        if res.label not in {"Medicamento", "Condicao", "EfeitoAdverso", "Paciente"}:
            continue
        v.entidades_resposta.append(res.canonical)
        variants = {normalize(res.canonical), normalize(res.surface)}
        if res.label == "Medicamento":
            variants |= {normalize(b) for b in resolver.brand_names.get(res.canonical, [])}
        if not any(f" {x} " in context_norm for x in variants if x):
            v.nao_fundamentadas.append(res.canonical)
    return v


def confidence(r: RetrievalResult, v: Verification) -> tuple[float, str, dict]:
    top_score = max((t["score"] for t in r.trechos), default=0.0)
    vetorial = min(max((top_score - 0.5) / 0.3, 0.0), 1.0)  # cosseno do nomic ~0.5 (fraco) a ~0.8 (forte)
    grafo = 1.0 if (r.alertas or r.caminhos) else 0.7 if r.fatos else 0.0
    total_ent = len(v.entidades_resposta)
    fundamentacao = 1.0 - (len(v.nao_fundamentadas) / total_ent) if total_ent else 0.6
    citacao = (len(v.citacoes_validas) / len(v.citacoes)) if v.citacoes else 0.0
    score = 0.25 * vetorial + 0.25 * grafo + 0.35 * fundamentacao + 0.15 * citacao
    nivel = "Alta" if score >= 0.75 else "Média" if score >= 0.5 else "Baixa"
    comps = {"similaridade vetorial": round(vetorial, 2), "suporte do grafo": round(grafo, 2),
             "entidades fundamentadas": round(fundamentacao, 2), "citações válidas": round(citacao, 2)}
    return round(score, 2), nivel, comps


class GraphRAG:
    def __init__(self, retriever: HybridRetriever, llm: OllamaClient, resolver: EntityResolver):
        self.retriever = retriever
        self.llm = llm
        self.resolver = resolver

    def ask(self, question: str, use_llm_cypher: bool = False, hops: int | None = None) -> Answer:
        started = time.time()
        r = self.retriever.retrieve(question, use_llm_cypher=use_llm_cypher, hops=hops)
        context = build_context(r)
        if not (r.alertas or r.fatos or r.trechos):
            text = NOT_FOUND
        else:
            if r.alertas:
                # KAG: o grafo já concluiu; o LLM só verbaliza os alertas (sem o resto do contexto para confundi-lo)
                ordered = [a for a in r.alertas if a.get("tipo") != "alternativa"] + \
                          [a for a in r.alertas if a.get("tipo") == "alternativa"]
                text = self.llm.generate(VERBALIZE_PROMPT.format(
                    pergunta=question, alertas="\n".join(f"[{a['ref']}] {a['texto']}" for a in ordered[:6])),
                    temperature=0.1, num_predict=350)
            else:
                text = self.llm.generate(ANSWER_PROMPT.format(contexto=context, pergunta=question, lead=LEAD_WITHOUT_ALERTS),
                                         temperature=0.1, num_predict=450, num_ctx=8192)
            text = ensure_alerts(text, r)
        refs = {a["ref"] for a in r.alertas} | {f["ref"] for f in r.fatos} | {t["ref"] for t in r.trechos}
        if r.cypher_llm and r.cypher_llm.get("linhas"):
            refs |= {f"C{i + 1}" for i in range(len(r.cypher_llm["linhas"][:15]))}
        v = verify(text, context, refs, self.resolver)
        score, nivel, comps = confidence(r, v)
        return Answer(text, r, context, v, score, nivel, comps, round(time.time() - started, 1))

    # ------------------------------------------------------- linhas de base
    def ask_vector_only(self, question: str) -> Answer:
        started = time.time()
        embedding = self.llm.embed_query(question)
        from app.retrieval import VECTOR_QUERY

        rows = self.retriever.graph.run(VECTOR_QUERY, {"k": self.retriever.top_k, "embedding": embedding})
        r = RetrievalResult(pergunta=question, trechos=[{**c, "ref": f"T{i + 1}"} for i, c in enumerate(rows)])
        context = "\n".join(f"[{t['ref']}] ({t['arquivo']}) {t['texto'][:650]}" for t in r.trechos[:3])
        text = self.llm.generate(VECTOR_ONLY_PROMPT.format(contexto=context, pergunta=question),
                                 temperature=0.1, num_predict=450)
        v = verify(text, context, {t["ref"] for t in r.trechos}, self.resolver)
        score, nivel, comps = confidence(r, v)
        return Answer(text, r, context, v, score, nivel, comps, round(time.time() - started, 1))

    def ask_no_context(self, question: str) -> Answer:
        started = time.time()
        text = self.llm.generate(NO_CONTEXT_PROMPT.format(pergunta=question), temperature=0.1, num_predict=450)
        v = verify(text, "", set(), self.resolver)
        return Answer(text, None, "", v, 0.0, "—", {}, round(time.time() - started, 1))
