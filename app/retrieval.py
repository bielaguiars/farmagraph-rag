"""Recuperação híbrida (GraphRAG/KAG): vetor + fulltext + travessia Cypher multi-hop + algoritmos.

Fluxo de uma pergunta:
  1. busca vetorial (cosseno) nos trechos            -> índice VECTOR chunk_embedding
  2. ligação de entidades da pergunta                -> dicionário + índice FULLTEXT entidade_nome
  3. expansão multi-hop (1..N saltos) a partir delas -> Cypher de comprimento variável
  4. raciocínio estruturado (KAG) para pacientes     -> Cypher de 2-3 saltos (interações e contraindicações)
  5. caminhos mais curtos entre entidades da pergunta -> shortestPath
  6. (opcional) Text-to-Cypher gerado pelo LLM, validado como somente leitura
  Os fatos são ranqueados por ligação com as sementes, gravidade e PageRank.
"""

import re
from dataclasses import dataclass, field

from app.graph import REL_PATTERN, Neo4jGraph
from app.ollama_client import OllamaClient
from app.resolution import EntityResolver, normalize
from app.schema import schema_description

STOPWORDS = set(
    "a o os as um uma de da do das dos em no na nos nas para por com sem que qual quais quem como "
    "pode podem posso devo deve ser tem ter tomar usar uso usando junto entre e ou se ao aos mais "
    "menos muito sobre isso esse essa este esta há existe existem risco riscos seguro segura quando "
    "remedio remedios medicamento medicamentos paciente pacientes alguma algum dele dela meu minha".split()
)

VECTOR_QUERY = """
CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding) YIELD node, score
MATCH (d:Documento)-[:TEM_TRECHO]->(node)
OPTIONAL MATCH (node)-[:MENCIONA]->(e)
RETURN node.id AS id, d.titulo AS titulo, d.arquivo AS arquivo, node.indice AS indice,
       node.texto AS texto, score,
       collect(DISTINCT {id: elementId(e), label: labels(e)[0], nome: e.nome}) AS entidades
ORDER BY score DESC
"""

LINK_QUERY = """
UNWIND $refs AS ref
MATCH (n) WHERE ref.label IN labels(n) AND n.id = ref.id
RETURN elementId(n) AS id, labels(n)[0] AS label, n.nome AS nome
"""

FULLTEXT_QUERY = """
CALL db.index.fulltext.queryNodes('entidade_nome', $q) YIELD node, score
RETURN elementId(node) AS id, labels(node)[0] AS label, node.nome AS nome, score
ORDER BY score DESC LIMIT 5
"""

EXPANSION_QUERY_TEMPLATE = """
MATCH (s) WHERE elementId(s) IN $seeds
MATCH path = (s)-[:{rels}*1..{hops}]-(n)
WITH path LIMIT 3000
UNWIND relationships(path) AS r
WITH DISTINCT r
WITH r, startNode(r) AS a, endNode(r) AS b
RETURN elementId(a) AS a_id, labels(a)[0] AS a_label, a.nome AS a_nome, a.pagerank AS a_pr, a.comunidade AS a_com,
       elementId(b) AS b_id, labels(b)[0] AS b_label, b.nome AS b_nome, b.pagerank AS b_pr, b.comunidade AS b_com,
       type(r) AS tipo, properties(r) AS props
"""

PATIENT_INTERACTIONS_QUERY = """
MATCH (p:Paciente)-[:USA]->(m:Medicamento)-[i:INTERAGE_COM]-(c:Medicamento)
WHERE elementId(p) IN $pacientes AND (elementId(c) IN $candidatos OR (p)-[:USA]->(c))
  AND (elementId(c) IN $candidatos OR m.nome < c.nome)
RETURN p.nome AS paciente, m.nome AS em_uso, c.nome AS outro, i.gravidade AS gravidade,
       i.efeito AS efeito, i.fontes AS fontes, EXISTS { (p)-[:USA]->(c) } AS ambos_em_uso
ORDER BY CASE i.gravidade WHEN 'alta' THEN 0 ELSE 1 END
"""

PATIENT_CONTRA_QUERY = """
MATCH (p:Paciente) WHERE elementId(p) IN $pacientes
MATCH (c:Medicamento)-[x:CONTRAINDICADO_EM]->(cond:Condicao)
WHERE (elementId(c) IN $candidatos OR (p)-[:USA]->(c))
  AND ((p)-[:TEM_CONDICAO]->(cond) OR elementId(cond) IN $condicoes)
RETURN p.nome AS paciente, c.nome AS medicamento, cond.nome AS condicao, x.fontes AS fontes,
       EXISTS { (p)-[:USA]->(c) } AS em_uso
"""

ALTERNATIVES_QUERY = """
MATCH (alt:Medicamento)-[:TRATA]->(alvo:Condicao) WHERE elementId(alvo) IN $condicoes
OPTIONAL MATCH (p:Paciente)-[:USA]->(m:Medicamento)-[i:INTERAGE_COM]-(alt) WHERE elementId(p) IN $pacientes
OPTIONAL MATCH (alt)-[:CONTRAINDICADO_EM]->(cc:Condicao)
  WHERE elementId(cc) IN $condicoes OR EXISTS { MATCH (p2:Paciente)-[:TEM_CONDICAO]->(cc) WHERE elementId(p2) IN $pacientes }
WITH alt, alvo, collect(DISTINCT CASE WHEN m IS NULL THEN NULL ELSE m.nome + ' (' + coalesce(i.gravidade, '?') + ', ' + coalesce(i.efeito, '?') + ')' END) AS interacoes,
     collect(DISTINCT cc.nome) AS contraindicacoes
RETURN alt.nome AS medicamento, alvo.nome AS trata, interacoes, contraindicacoes,
       size(interacoes) = 0 AND size(contraindicacoes) = 0 AS sem_alerta
ORDER BY sem_alerta DESC, medicamento
"""

SHORTEST_PATH_QUERY = f"""
MATCH (a), (b) WHERE elementId(a) = $a AND elementId(b) = $b
MATCH p = shortestPath((a)-[:{REL_PATTERN}*..6]-(b))
RETURN [n IN nodes(p) | {{id: elementId(n), label: labels(n)[0], nome: n.nome}}] AS nos,
       [r IN relationships(p) | {{tipo: type(r), a: elementId(startNode(r)), b: elementId(endNode(r)),
                                  props: properties(r)}}] AS rels
"""

CYPHER_SCHEMA = {
    "type": "object",
    "properties": {"cypher": {"type": "string"}},
    "required": ["cypher"],
}

CYPHER_PROMPT = """Converta a pergunta em UMA consulta Cypher somente de leitura para o Neo4j.
Esquema:
{schema}
Regras: use apenas MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT. Compare nomes com
toLower(x.nome) = '...' usando os nomes canônicos das entidades reconhecidas. INTERAGE_COM é simétrica:
use -[:INTERAGE_COM]- sem seta. Retorne colunas com nomes legíveis e LIMIT 25.
Exemplos:
P: Quais medicamentos interagem com a varfarina?
{{"cypher": "MATCH (m:Medicamento)-[i:INTERAGE_COM]-(o:Medicamento) WHERE toLower(m.nome) = 'varfarina' RETURN o.nome AS medicamento, i.gravidade AS gravidade, i.efeito AS efeito LIMIT 25"}}
P: Que remédios a paciente Ana usa e para que servem?
{{"cypher": "MATCH (p:Paciente)-[:USA]->(m:Medicamento) WHERE toLower(p.nome) CONTAINS 'ana' OPTIONAL MATCH (m)-[:TRATA]->(c:Condicao) RETURN p.nome AS paciente, m.nome AS medicamento, collect(c.nome) AS trata LIMIT 25"}}
P: Quais medicamentos são contraindicados na gravidez?
{{"cypher": "MATCH (m:Medicamento)-[:CONTRAINDICADO_EM]->(c:Condicao) WHERE toLower(c.nome) = 'gravidez' RETURN m.nome AS medicamento LIMIT 25"}}
P: Quais pacientes usam algum medicamento que pode causar sangramento?
{{"cypher": "MATCH (p:Paciente)-[:USA]->(m:Medicamento)-[:PODE_CAUSAR]->(e:EfeitoAdverso) WHERE toLower(e.nome) = 'sangramento' RETURN p.nome AS paciente, m.nome AS medicamento LIMIT 25"}}
Entidades reconhecidas na pergunta: {entidades}
P: {pergunta}
"""

FORBIDDEN = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD|FOREACH|CALL|apoc|gds|dbms)\b", re.I)
RELATION_WEIGHT = {"INTERAGE_COM": 3, "CONTRAINDICADO_EM": 3, "USA": 2, "TEM_CONDICAO": 2, "TRATA": 1,
                   "PODE_CAUSAR": 1, "PERTENCE_A_CLASSE": 0}


@dataclass
class RetrievalResult:
    pergunta: str
    sementes: list[dict] = field(default_factory=list)
    trechos: list[dict] = field(default_factory=list)
    fatos: list[dict] = field(default_factory=list)
    alertas: list[dict] = field(default_factory=list)
    caminhos: list[dict] = field(default_factory=list)
    cypher_log: list[dict] = field(default_factory=list)
    cypher_llm: dict | None = None
    subgrafo: dict = field(default_factory=lambda: {"nodes": [], "edges": []})


class HybridRetriever:
    def __init__(self, graph: Neo4jGraph, llm: OllamaClient, resolver: EntityResolver,
                 top_k: int = 5, max_hops: int = 2, max_facts: int = 28):
        self.graph = graph
        self.llm = llm
        self.resolver = resolver
        self.top_k = top_k
        self.max_hops = max_hops
        self.max_facts = max_facts

    # ------------------------------------------------------------------ util
    def _log(self, result: RetrievalResult, etapa: str, query: str, params: dict, rows: list) -> list:
        shown = {k: (f"<vetor {len(v)} dimensões>" if k == "embedding" else v) for k, v in params.items()}
        result.cypher_log.append({"etapa": etapa, "cypher": query.strip(), "parametros": shown, "linhas": len(rows)})
        return rows

    def _run(self, result: RetrievalResult, etapa: str, query: str, params: dict) -> list[dict]:
        return self._log(result, etapa, query, params, self.graph.run(query, params))

    # ------------------------------------------------------------- pipeline
    def retrieve(self, question: str, use_llm_cypher: bool = False, hops: int | None = None) -> RetrievalResult:
        result = RetrievalResult(pergunta=question)
        hops = hops or self.max_hops

        # 1. busca vetorial
        embedding = self.llm.embed_query(question)
        chunks = self._run(result, "1. Busca vetorial (índice chunk_embedding, cosseno)", VECTOR_QUERY,
                           {"k": self.top_k, "embedding": embedding})
        result.trechos = [
            {**c, "entidades": [e for e in c["entidades"] if e.get("id")], "ref": f"T{i + 1}"}
            for i, c in enumerate(chunks)
        ]

        # 2. ligação de entidades: dicionário/resolução + fulltext
        seeds: dict[str, dict] = {}
        refs = [{"label": r.label, "id": normalize(r.canonical)} for r in self.resolver.find_in_text(question)]
        refs = _drop_contained(refs)
        if refs:
            for row in self._run(result, "2a. Ligação de entidades (resolução + índice RANGE em id)", LINK_QUERY, {"refs": refs}):
                seeds[row["id"]] = {**row, "origem": "pergunta"}
        lucene = _lucene_query(question, covered={normalize(s["nome"]) for s in seeds.values()})
        if lucene:
            hits = self._run(result, "2b. Ligação por texto aproximado (índice FULLTEXT entidade_nome)",
                             FULLTEXT_QUERY, {"q": lucene})
            if hits:
                top = hits[0]["score"]
                for hit in hits:
                    if hit["score"] >= max(1.0, 0.75 * top) and hit["id"] not in seeds and hit["label"] in {"Medicamento", "Paciente"}:
                        seeds[hit["id"]] = {**{k: hit[k] for k in ("id", "label", "nome")}, "origem": "fulltext"}
        if not seeds:  # sem entidades na pergunta: usa as entidades dos trechos mais similares
            for chunk in result.trechos[:2]:
                for e in chunk["entidades"][:4]:
                    seeds.setdefault(e["id"], {**e, "origem": "trecho"})
        result.sementes = list(seeds.values())
        seed_ids = list(seeds)
        if not seed_ids:
            return result

        # 3. expansão multi-hop
        expansion = EXPANSION_QUERY_TEMPLATE.format(rels=REL_PATTERN, hops=hops)
        edges = self._run(result, f"3. Expansão multi-hop (1..{hops} saltos)", expansion, {"seeds": seed_ids})

        # 4. raciocínio estruturado (KAG)
        pacientes = [s["id"] for s in result.sementes if s["label"] == "Paciente"]
        candidatos = [s["id"] for s in result.sementes if s["label"] == "Medicamento"]
        condicoes = [s["id"] for s in result.sementes if s["label"] == "Condicao"]
        alerts: list[dict] = []
        if pacientes:
            params = {"pacientes": pacientes, "candidatos": candidatos, "condicoes": condicoes}
            for row in self._run(result, "4a. KAG: interações com os medicamentos do paciente", PATIENT_INTERACTIONS_QUERY, params):
                situacao = "ambos já em uso" if row["ambos_em_uso"] else "medicamento citado na pergunta"
                alerts.append({"tipo": "interacao", "texto": (
                    f"{row['paciente']} usa {row['em_uso']}, que INTERAGE_COM {row['outro']} ({situacao}): "
                    f"gravidade {row['gravidade'] or '?'}, efeito {row['efeito'] or '?'}."), "gravidade": row["gravidade"],
                    "fontes": row.get("fontes") or []})
            for row in self._run(result, "4b. KAG: contraindicações x condições do paciente", PATIENT_CONTRA_QUERY, params):
                alerts.append({"tipo": "contraindicacao", "texto": (
                    f"{row['medicamento']} é CONTRAINDICADO_EM {row['condicao']}, condição de {row['paciente']}"
                    f"{' (já em uso)' if row['em_uso'] else ''}."), "gravidade": "alta", "fontes": row.get("fontes") or []})
        if condicoes and (pacientes or len(condicoes) >= 2):
            params = {"pacientes": pacientes, "condicoes": condicoes}
            for row in self._run(result, "4c. KAG: alternativas que tratam a condição, checando riscos", ALTERNATIVES_QUERY, params):
                if row["sem_alerta"]:
                    texto = f"{row['medicamento']} TRATA {row['trata']} e não tem interação nem contraindicação registrada para este caso."
                else:
                    partes = []
                    if row["interacoes"]:
                        partes.append("interage com " + ", ".join(row["interacoes"]))
                    if row["contraindicacoes"]:
                        partes.append("é contraindicado em " + ", ".join(row["contraindicacoes"]))
                    texto = f"{row['medicamento']} TRATA {row['trata']}, mas " + " e ".join(partes) + "."
                alerts.append({"tipo": "alternativa", "texto": texto, "gravidade": "" if row["sem_alerta"] else "alta", "fontes": []})
        result.alertas = [{**a, "ref": f"K{i + 1}"} for i, a in enumerate(alerts[:15])]

        # 5. caminhos mais curtos entre entidades da pergunta
        main = [s for s in result.sementes if s["origem"] != "trecho"][:4]
        pairs = [(a, b) for i, a in enumerate(main) for b in main[i + 1:]
                 if not (a["label"] == "Condicao" and b["label"] == "Condicao")][:3]
        for a, b in pairs:
            rows = self._run(result, f"5. shortestPath {a['nome']} ↔ {b['nome']}", SHORTEST_PATH_QUERY, {"a": a["id"], "b": b["id"]})
            if rows:
                result.caminhos.append({"de": a["nome"], "para": b["nome"], **rows[0], "texto": _path_text(rows[0])})
        # 5b. KAG: dois medicamentos sem ligação direta -> o caminho vira conclusão verificada (evita o LLM
        #     "encurtar" o caminho e inventar uma interação direta)
        if not result.alertas:
            for p in result.caminhos:
                alert = _path_alert(p)
                if alert:
                    result.alertas.append({**alert, "ref": f"K{len(result.alertas) + 1}"})

        # ranqueamento dos fatos do subgrafo
        result.fatos = self._rank_facts(edges, set(seed_ids), result.caminhos)

        # 6. Text-to-Cypher (opcional)
        if use_llm_cypher:
            result.cypher_llm = self.text_to_cypher(question, result.sementes)

        result.subgrafo = self._build_subgraph(result)
        return result

    def _rank_facts(self, edges: list[dict], seeds: set[str], paths: list[dict]) -> list[dict]:
        neighbors: dict[str, set[str]] = {}
        for e in edges:
            for x, y in ((e["a_id"], e["b_id"]), (e["b_id"], e["a_id"])):
                if y in seeds:
                    neighbors.setdefault(x, set()).add(y)
        bridges = {n for n, s in neighbors.items() if len(s) >= 2}
        path_edges = {(r["a"], r["b"], r["tipo"]) for p in paths for r in p["rels"]}
        scored = []
        for e in edges:
            score = 4 * (e["a_id"] in seeds) + 4 * (e["b_id"] in seeds)
            score += RELATION_WEIGHT.get(e["tipo"], 0)
            gravidade = (e.get("props") or {}).get("gravidade")
            score += 2 if gravidade == "alta" else 1 if gravidade == "moderada" else 0
            score += 5 if (e["a_id"] in bridges or e["b_id"] in bridges) else 0
            score += 6 if (e["a_id"], e["b_id"], e["tipo"]) in path_edges else 0
            score += 10 * ((e.get("a_pr") or 0) + (e.get("b_pr") or 0)) / 2
            scored.append((score, e))
        scored.sort(key=lambda x: -x[0])
        facts = []
        for i, (score, e) in enumerate(scored[: self.max_facts]):
            props = {k: v for k, v in (e.get("props") or {}).items() if k in {"gravidade", "efeito", "fontes"}}
            facts.append({**e, "props": props, "score": round(score, 2), "ref": f"G{i + 1}",
                          "texto": _fact_text(e)})
        return facts

    def _build_subgraph(self, result: RetrievalResult) -> dict:
        nodes: dict[str, dict] = {}
        edges: dict[tuple, dict] = {}
        seed_ids = {s["id"] for s in result.sementes}
        for f in result.fatos:
            for side in ("a", "b"):
                nodes.setdefault(f[f"{side}_id"], {"id": f[f"{side}_id"], "label": f[f"{side}_label"], "nome": f[f"{side}_nome"],
                                                   "comunidade": f.get(f"{side}_com"), "pagerank": f.get(f"{side}_pr")})
            edges[(f["a_id"], f["b_id"], f["tipo"])] = {"source": f["a_id"], "target": f["b_id"], "tipo": f["tipo"],
                                                        "props": f["props"], "ref": f["ref"]}
        for p in result.caminhos:
            for n in p["nos"]:
                nodes.setdefault(n["id"], {"id": n["id"], "label": n["label"], "nome": n["nome"]})
            for r in p["rels"]:
                edges.setdefault((r["a"], r["b"], r["tipo"]), {"source": r["a"], "target": r["b"], "tipo": r["tipo"],
                                                              "props": r.get("props") or {}, "caminho": True})
        for s in result.sementes:
            nodes.setdefault(s["id"], {"id": s["id"], "label": s["label"], "nome": s["nome"]})
        for n in nodes.values():
            n["semente"] = n["id"] in seed_ids
        return {"nodes": list(nodes.values()), "edges": list(edges.values())}

    # ------------------------------------------------------------ Text-to-Cypher
    def text_to_cypher(self, question: str, seeds: list[dict]) -> dict:
        entidades = ", ".join(f"{s['nome']} ({s['label']})" for s in seeds) or "nenhuma"
        prompt = CYPHER_PROMPT.format(schema=schema_description(), entidades=entidades, pergunta=question)
        out: dict = {"cypher": "", "linhas": [], "erro": None, "tentativas": []}
        feedback = ""
        for _ in range(2):  # autocorreção: se a validação rejeitar, o LLM recebe o erro e tenta de novo
            try:
                cypher = self.llm.generate_json(prompt + feedback, CYPHER_SCHEMA, num_predict=300).get("cypher", "").strip()
                cypher = undirect_symmetric(cypher)
            except RuntimeError as exc:
                out["erro"] = str(exc)
                return out
            ok, motivo = self.validate_read_only(cypher)
            if ok:
                break
            out["tentativas"].append({"cypher": cypher, "erro": motivo})
            feedback = (f"\nA consulta anterior foi rejeitada pela validação: {motivo}\nConsulta rejeitada: {cypher}\n"
                        "Gere outra consulta, mais simples, que corrija o problema.\n")
        out["cypher"] = cypher
        if not ok:
            out["erro"] = motivo
            return out
        if not re.search(r"\bLIMIT\s+\d+", cypher, re.I):
            cypher = f"{cypher.rstrip(' ;')} LIMIT 25"
            out["cypher"] = cypher
        try:
            out["linhas"] = self.graph.run(cypher)[:25]
        except Exception as exc:  # consulta válida sintaticamente mas falhou ao executar
            out["erro"] = f"erro ao executar: {exc}"
        return out

    def validate_read_only(self, cypher: str) -> tuple[bool, str]:
        if not cypher:
            return False, "LLM não gerou consulta"
        if FORBIDDEN.search(cypher):
            return False, "consulta bloqueada: contém cláusula de escrita ou procedimento"
        try:
            _, summary, _ = self.graph.driver.execute_query(f"EXPLAIN {cypher}", database_=self.graph.database)
        except Exception as exc:
            return False, f"Cypher inválido: {str(exc)[:200]}"
        if summary.query_type != "r":
            return False, f"consulta não é somente leitura (tipo {summary.query_type})"
        return True, ""


SYMMETRIC_REL = re.compile(r"<?-\[([^\]]*:\s*INTERAGE_COM\b[^\]]*)\]->?")


def undirect_symmetric(cypher: str) -> str:
    """INTERAGE_COM é simétrica e gravada numa ordem canônica: com seta, a consulta perderia metade das interações."""
    return SYMMETRIC_REL.sub(r"-[\1]-", cypher)


def _drop_contained(refs: list[dict]) -> list[dict]:
    """Remove 'dor' quando também há 'dor no joelho' etc. (mantém o termo mais específico)."""
    ids = [r["id"] for r in refs]
    return [r for r in refs if not any(r["id"] != o and f" {r['id']} " in f" {o} " for o in ids)]


def _lucene_query(question: str, covered: set[str]) -> str:
    covered_tokens = {t for c in covered for t in c.split()}
    tokens = [t for t in re.findall(r"[\wÀ-ÿ]+", question)
              if len(t) >= 5 and normalize(t) not in STOPWORDS and normalize(t) not in covered_tokens]
    escaped = [re.sub(r'([+\-!(){}\[\]^"~*?:\\/])', r"\\\1", t) for t in tokens]
    return " OR ".join(f"{t}~1" for t in escaped[:8])


def _fact_text(e: dict) -> str:
    props = e.get("props") or {}
    extra = []
    if props.get("gravidade"):
        extra.append(f"gravidade {props['gravidade']}")
    if props.get("efeito"):
        extra.append(f"efeito {props['efeito']}")
    suffix = f" ({', '.join(extra)})" if extra else ""
    return f"{e['a_nome']} [{e['a_label']}] -{e['tipo']}-> {e['b_nome']} [{e['b_label']}]{suffix}"


def _path_alert(path: dict) -> dict | None:
    """Conclusão KAG para um caminho de 2+ saltos entre dois medicamentos."""
    nodes, rels = path.get("nos") or [], path.get("rels") or []
    if len(rels) < 2 or nodes[0]["label"] != "Medicamento" or nodes[-1]["label"] != "Medicamento":
        return None
    names = {n["id"]: n["nome"] for n in nodes}
    detalhes = []
    for r in rels:
        props = r.get("props") or {}
        extra = ", ".join(str(props[k]) for k in ("gravidade", "efeito") if props.get(k))
        detalhes.append(f"{names.get(r['a'], '?')} {r['tipo']} {names.get(r['b'], '?')}" + (f" ({extra})" if extra else ""))
    gravidades = {(r.get("props") or {}).get("gravidade") for r in rels}
    return {
        "tipo": "caminho",
        "texto": (f"Não há relação direta entre {nodes[0]['nome']} e {nodes[-1]['nome']} no grafo; eles se ligam em "
                  f"{len(rels)} saltos: {_path_text(path)}. Detalhes: " + "; ".join(detalhes) + "."),
        "gravidade": "alta" if "alta" in gravidades else "",
        "fontes": [f for r in rels for f in ((r.get("props") or {}).get("fontes") or [])],
    }


def _path_text(path: dict) -> str:
    names = {n["id"]: n["nome"] for n in path["nos"]}
    parts = [path["nos"][0]["nome"]]
    current = path["nos"][0]["id"]
    for r in path["rels"]:
        nxt = r["b"] if r["a"] == current else r["a"]
        arrow = f"-{r['tipo']}->" if r["a"] == current else f"<-{r['tipo']}-"
        parts.append(f"{arrow} {names.get(nxt, '?')}")
        current = nxt
    return " ".join(parts)
