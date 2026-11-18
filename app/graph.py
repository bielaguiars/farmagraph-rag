"""Acesso ao Neo4j: esquema, gravação idempotente (MERGE), algoritmos de grafo e consultas."""

from neo4j import GraphDatabase, RoutingControl

from app.resolution import normalize
from app.schema import (
    CONSTRAINTS,
    DOMAIN_RELATIONS,
    ENTITY_LABELS,
    RELATION_SCHEMA,
    index_statements,
)

REL_PATTERN = "|".join(DOMAIN_RELATIONS)


class Neo4jGraph:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def ping(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def run(self, query: str, params: dict | None = None, write: bool = False) -> list[dict]:
        records, _, _ = self.driver.execute_query(
            query,
            params or {},
            database_=self.database,
            routing_=RoutingControl.WRITE if write else RoutingControl.READ,
        )
        return [record.data() for record in records]

    # ------------------------------------------------------------------ esquema
    def setup_schema(self, embedding_dimension: int) -> list[str]:
        statements = [*CONSTRAINTS, *index_statements(embedding_dimension)]
        for statement in statements:
            self.run(statement, write=True)
        self.run("CALL db.awaitIndexes(120)", write=True)
        return statements

    def schema_info(self) -> dict:
        constraints = self.run("SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties RETURN *")
        indexes = self.run(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state "
            "WHERE type <> 'LOOKUP' RETURN *"
        )
        return {"constraints": constraints, "indexes": indexes}

    def reset(self) -> None:
        # CALL {} IN TRANSACTIONS exige transação implícita (auto-commit): session.run, não execute_query
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 1000 ROWS").consume()

    # ---------------------------------------------------------------- gravação
    def document_exists(self, doc_id: str) -> bool:
        return bool(self.run("MATCH (d:Documento {id: $id, status: 'ok'}) RETURN d.id AS id", {"id": doc_id}))

    def upsert_document(self, doc: dict, chunks: list[dict]) -> None:
        self.run(
            """
            MERGE (d:Documento {id: $doc.id})
            SET d.titulo = $doc.titulo, d.tipo = $doc.tipo, d.arquivo = $doc.arquivo,
                d.assunto = $doc.assunto, d.status = 'processando'
            WITH d
            OPTIONAL MATCH (d)-[:TEM_TRECHO]->(old:Chunk)
            DETACH DELETE old
            """,
            {"doc": doc},
            write=True,
        )
        self.run(
            """
            MATCH (d:Documento {id: $doc_id})
            UNWIND $chunks AS c
            CREATE (k:Chunk {id: c.id, documento_id: $doc_id, indice: c.indice, texto: c.texto,
                             arquivo: $arquivo})
            WITH d, k, c
            CALL db.create.setNodeVectorProperty(k, 'embedding', c.embedding)
            MERGE (d)-[:TEM_TRECHO]->(k)
            """,
            {"doc_id": doc["id"], "arquivo": doc["arquivo"], "chunks": chunks},
            write=True,
        )

    def mark_document_ok(self, doc_id: str) -> None:
        self.run("MATCH (d:Documento {id: $id}) SET d.status = 'ok'", {"id": doc_id}, write=True)

    def merge_entities(self, label: str, entities: list[dict]) -> None:
        if label not in ENTITY_LABELS or not entities:
            return
        self.run(
            f"""
            UNWIND $rows AS row
            MERGE (n:{label} {{id: row.id}})
            ON CREATE SET n.nome = row.nome
            SET n.aliases_texto = row.aliases_texto,
                n.nomes_comerciais = CASE WHEN size(row.nomes_comerciais) > 0
                                          THEN row.nomes_comerciais ELSE n.nomes_comerciais END
            """,
            {"rows": entities},
            write=True,
        )

    def merge_mentions(self, chunk_id: str, entities: list[tuple[str, str]]) -> None:
        for label in ENTITY_LABELS:
            ids = [normalize(name) for lab, name in entities if lab == label]
            if ids:
                self.run(
                    f"""
                    MATCH (k:Chunk {{id: $chunk_id}})
                    UNWIND $ids AS eid
                    MATCH (e:{label} {{id: eid}})
                    MERGE (k)-[:MENCIONA]->(e)
                    """,
                    {"chunk_id": chunk_id, "ids": ids},
                    write=True,
                )

    def merge_relation(self, tipo: str, src: tuple[str, str], dst: tuple[str, str], props: dict,
                       chunk_id: str) -> None:
        if tipo not in RELATION_SCHEMA:
            raise ValueError(f"Relação fora do esquema: {tipo}")
        src_label, dst_label, _ = RELATION_SCHEMA[tipo]
        self.run(
            f"""
            MATCH (a:{src_label} {{id: $src}}), (b:{dst_label} {{id: $dst}})
            MERGE (a)-[r:{tipo}]->(b)
            ON CREATE SET r.fontes = [$chunk], r.extraido_por = 'llm'
            ON MATCH SET r.fontes = CASE WHEN $chunk IN r.fontes THEN r.fontes ELSE r.fontes + $chunk END
            SET r.gravidade = CASE WHEN r.gravidade = 'alta' THEN 'alta'
                                   ELSE coalesce($props.gravidade, r.gravidade) END,
                r.efeito = coalesce(r.efeito, $props.efeito)
            """,
            {"src": normalize(src[1]), "dst": normalize(dst[1]), "props": props, "chunk": chunk_id},
            write=True,
        )

    # -------------------------------------------------------- algoritmos de grafo
    def compute_algorithms(self) -> dict:
        """PageRank no grafo de domínio e Louvain nas interações (GDS, com fallback networkx)."""
        try:
            return self._compute_with_gds()
        except Exception as exc:  # GDS ausente ou falhou: calcula em Python e grava de volta
            info = self._compute_with_networkx()
            info["aviso"] = f"GDS indisponível ({type(exc).__name__}); usado networkx."
            return info

    def _compute_with_gds(self) -> dict:
        for name in ("farma_dominio", "farma_interacoes"):
            self.run("CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName", {"name": name}, write=True)
        rel_projection = {rel: {"orientation": "UNDIRECTED"} for rel in DOMAIN_RELATIONS}
        self.run(
            "CALL gds.graph.project('farma_dominio', $labels, $rels)",
            {"labels": ENTITY_LABELS, "rels": rel_projection},
            write=True,
        )
        self.run(
            "CALL gds.pageRank.write('farma_dominio', {writeProperty: 'pagerank'}) YIELD nodePropertiesWritten RETURN *",
            write=True,
        )
        self.run(
            "CALL gds.graph.project('farma_interacoes', 'Medicamento', {INTERAGE_COM: {orientation: 'UNDIRECTED'}})",
            write=True,
        )
        result = self.run(
            "CALL gds.louvain.write('farma_interacoes', {writeProperty: 'comunidade'}) "
            "YIELD communityCount, modularity RETURN communityCount, modularity",
            write=True,
        )
        for name in ("farma_dominio", "farma_interacoes"):
            self.run("CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName", {"name": name}, write=True)
        return {"motor": "Neo4j GDS", **(result[0] if result else {})}

    def _compute_with_networkx(self) -> dict:
        import networkx as nx

        edges = self.run(
            f"MATCH (a)-[r:{REL_PATTERN}]->(b) RETURN elementId(a) AS a, elementId(b) AS b, type(r) AS t"
        )
        graph = nx.Graph()
        graph.add_edges_from((e["a"], e["b"]) for e in edges)
        pagerank = nx.pagerank(graph) if graph.number_of_nodes() else {}
        self.run(
            "UNWIND $rows AS row MATCH (n) WHERE elementId(n) = row.id SET n.pagerank = row.score",
            {"rows": [{"id": k, "score": v} for k, v in pagerank.items()]},
            write=True,
        )
        inter = nx.Graph()
        inter.add_edges_from((e["a"], e["b"]) for e in edges if e["t"] == "INTERAGE_COM")
        communities = nx.community.louvain_communities(inter, seed=42) if inter.number_of_nodes() else []
        rows = [{"id": node, "c": i} for i, group in enumerate(communities) for node in group]
        self.run(
            "UNWIND $rows AS row MATCH (n) WHERE elementId(n) = row.id SET n.comunidade = row.c",
            {"rows": rows},
            write=True,
        )
        modularity = nx.community.modularity(inter, communities) if communities else 0.0
        return {"motor": "networkx", "communityCount": len(communities), "modularity": modularity}

    # ----------------------------------------------------------------- leitura
    def stats(self) -> dict:
        rows = self.run(
            "MATCH (n) WITH labels(n)[0] AS label, count(*) AS total RETURN label, total ORDER BY label"
        )
        rels = self.run("MATCH ()-[r]->() RETURN type(r) AS tipo, count(*) AS total ORDER BY tipo")
        return {"nos": {r["label"]: r["total"] for r in rows}, "relacoes": {r["tipo"]: r["total"] for r in rels}}

    def entity_names(self) -> dict[str, list[str]]:
        rows = self.run(
            "MATCH (n) WHERE n:Medicamento OR n:Condicao OR n:EfeitoAdverso OR n:ClasseTerapeutica OR n:Paciente "
            "RETURN labels(n)[0] AS label, n.nome AS nome"
        )
        names: dict[str, list[str]] = {}
        for row in rows:
            names.setdefault(row["label"], []).append(row["nome"])
        return names

    def full_graph(self, limit: int = 600) -> dict:
        rows = self.run(
            f"""
            MATCH (a)-[r:{REL_PATTERN}]->(b)
            RETURN elementId(a) AS a_id, labels(a)[0] AS a_label, a.nome AS a_nome,
                   a.comunidade AS a_com, a.pagerank AS a_pr,
                   elementId(b) AS b_id, labels(b)[0] AS b_label, b.nome AS b_nome,
                   b.comunidade AS b_com, b.pagerank AS b_pr,
                   type(r) AS tipo, properties(r) AS props
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return rows_to_subgraph(rows)

    def top_pagerank(self, limit: int = 10) -> list[dict]:
        return self.run(
            "MATCH (n) WHERE n.pagerank IS NOT NULL AND (n:Medicamento OR n:Condicao OR n:EfeitoAdverso) "
            "RETURN n.nome AS entidade, labels(n)[0] AS rotulo, round(n.pagerank, 4) AS pagerank "
            "ORDER BY n.pagerank DESC LIMIT $limit",
            {"limit": limit},
        )

    def communities(self) -> list[dict]:
        return self.run(
            "MATCH (m:Medicamento) WHERE m.comunidade IS NOT NULL "
            "RETURN m.comunidade AS comunidade, collect(m.nome) AS medicamentos, count(*) AS total "
            "ORDER BY total DESC"
        )


def rows_to_subgraph(rows: list[dict]) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for row in rows:
        for side in ("a", "b"):
            node_id = row[f"{side}_id"]
            if node_id not in nodes:
                nodes[node_id] = {
                    "id": node_id,
                    "label": row[f"{side}_label"],
                    "nome": row[f"{side}_nome"],
                    "comunidade": row.get(f"{side}_com"),
                    "pagerank": row.get(f"{side}_pr"),
                }
        props = {k: v for k, v in (row.get("props") or {}).items() if k not in {"extraido_por"}}
        edges.append({"source": row["a_id"], "target": row["b_id"], "tipo": row["tipo"], "props": props})
    return {"nodes": list(nodes.values()), "edges": edges}
