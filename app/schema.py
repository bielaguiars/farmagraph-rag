"""Esquema LPG do domínio farmacológico: rótulos, relações permitidas, constraints e índices.

O esquema é pré-definido (orientado ao domínio) e funciona também como restrição semântica:
uma relação extraída pelo LLM só entra no grafo se respeitar o domínio/contradomínio abaixo.
"""

ENTITY_LABELS = ["Medicamento", "ClasseTerapeutica", "Condicao", "EfeitoAdverso", "Paciente"]

# tipo de relação -> (rótulo de origem, rótulo de destino, descrição)
RELATION_SCHEMA: dict[str, tuple[str, str, str]] = {
    "PERTENCE_A_CLASSE": ("Medicamento", "ClasseTerapeutica", "medicamento pertence a uma classe terapêutica"),
    "TRATA": ("Medicamento", "Condicao", "medicamento é indicado para a condição"),
    "CONTRAINDICADO_EM": ("Medicamento", "Condicao", "medicamento é contraindicado na condição"),
    "PODE_CAUSAR": ("Medicamento", "EfeitoAdverso", "medicamento pode causar a reação adversa"),
    "INTERAGE_COM": ("Medicamento", "Medicamento", "interação medicamentosa entre dois fármacos"),
    "USA": ("Paciente", "Medicamento", "paciente faz uso do medicamento"),
    "TEM_CONDICAO": ("Paciente", "Condicao", "paciente tem a condição/diagnóstico"),
}
DOMAIN_RELATIONS = list(RELATION_SCHEMA)
SYMMETRIC_RELATIONS = {"INTERAGE_COM"}

# Constraints de unicidade (cada uma cria um índice RANGE/B-tree implícito em `id`).
CONSTRAINTS = [
    f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
    for label in [*ENTITY_LABELS, "Documento", "Chunk"]
]


def index_statements(embedding_dimension: int) -> list[str]:
    return [
        # Índices RANGE (B-tree) para filtros e ordenações frequentes
        "CREATE INDEX medicamento_nome IF NOT EXISTS FOR (n:Medicamento) ON (n.nome)",
        "CREATE INDEX paciente_nome IF NOT EXISTS FOR (n:Paciente) ON (n.nome)",
        "CREATE INDEX chunk_documento IF NOT EXISTS FOR (n:Chunk) ON (n.documento_id)",
        "CREATE INDEX interacao_gravidade IF NOT EXISTS FOR ()-[r:INTERAGE_COM]-() ON (r.gravidade)",
        # Fulltext (Lucene) para ligar termos da pergunta a entidades, inclusive nomes comerciais
        "CREATE FULLTEXT INDEX entidade_nome IF NOT EXISTS "
        "FOR (n:Medicamento|ClasseTerapeutica|Condicao|EfeitoAdverso|Paciente) ON EACH [n.nome, n.aliases_texto]",
        "CREATE FULLTEXT INDEX chunk_texto IF NOT EXISTS FOR (n:Chunk) ON EACH [n.texto]",
        # Vector index (cosseno) para busca semântica nos trechos
        "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (n:Chunk) ON (n.embedding) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {embedding_dimension}, "
        "`vector.similarity_function`: 'cosine'}}",
    ]


def schema_description() -> str:
    """Descrição textual do esquema usada nos prompts (extração e Text-to-Cypher)."""
    lines = ["Rótulos de nós: " + ", ".join(ENTITY_LABELS) + ", Documento, Chunk."]
    lines.append("Propriedade de nome: n.nome (Medicamento.nome é o princípio ativo em minúsculas, "
                 "ex.: 'varfarina'; Medicamento.nomes_comerciais é lista, ex.: ['Marevan']).")
    lines.append("Relações:")
    for rel, (src, dst, desc) in RELATION_SCHEMA.items():
        extra = " {gravidade: 'alta'|'moderada', efeito, mecanismo}" if rel == "INTERAGE_COM" else ""
        lines.append(f"  (:{src})-[:{rel}{extra}]->(:{dst})  // {desc}")
    lines.append("  (:Documento)-[:TEM_TRECHO]->(:Chunk)-[:MENCIONA]->(entidade)")
    return "\n".join(lines)
