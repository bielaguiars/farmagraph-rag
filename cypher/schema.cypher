// Esquema LPG do FarmaGraph RAG (gerado de app/schema.py)

// Constraints de unicidade
CREATE CONSTRAINT medicamento_id IF NOT EXISTS FOR (n:Medicamento) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT classeterapeutica_id IF NOT EXISTS FOR (n:ClasseTerapeutica) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT condicao_id IF NOT EXISTS FOR (n:Condicao) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT efeitoadverso_id IF NOT EXISTS FOR (n:EfeitoAdverso) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT paciente_id IF NOT EXISTS FOR (n:Paciente) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT documento_id IF NOT EXISTS FOR (n:Documento) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE;

// Indices RANGE, FULLTEXT e VECTOR
CREATE INDEX medicamento_nome IF NOT EXISTS FOR (n:Medicamento) ON (n.nome);
CREATE INDEX paciente_nome IF NOT EXISTS FOR (n:Paciente) ON (n.nome);
CREATE INDEX chunk_documento IF NOT EXISTS FOR (n:Chunk) ON (n.documento_id);
CREATE INDEX interacao_gravidade IF NOT EXISTS FOR ()-[r:INTERAGE_COM]-() ON (r.gravidade);
CREATE FULLTEXT INDEX entidade_nome IF NOT EXISTS FOR (n:Medicamento|ClasseTerapeutica|Condicao|EfeitoAdverso|Paciente) ON EACH [n.nome, n.aliases_texto];
CREATE FULLTEXT INDEX chunk_texto IF NOT EXISTS FOR (n:Chunk) ON EACH [n.texto];
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (n:Chunk) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}};

/* Modelo
Rótulos de nós: Medicamento, ClasseTerapeutica, Condicao, EfeitoAdverso, Paciente, Documento, Chunk.
Propriedade de nome: n.nome (Medicamento.nome é o princípio ativo em minúsculas, ex.: 'varfarina'; Medicamento.nomes_comerciais é lista, ex.: ['Marevan']).
Relações:
  (:Medicamento)-[:PERTENCE_A_CLASSE]->(:ClasseTerapeutica)  // medicamento pertence a uma classe terapêutica
  (:Medicamento)-[:TRATA]->(:Condicao)  // medicamento é indicado para a condição
  (:Medicamento)-[:CONTRAINDICADO_EM]->(:Condicao)  // medicamento é contraindicado na condição
  (:Medicamento)-[:PODE_CAUSAR]->(:EfeitoAdverso)  // medicamento pode causar a reação adversa
  (:Medicamento)-[:INTERAGE_COM {gravidade: 'alta'|'moderada', efeito, mecanismo}]->(:Medicamento)  // interação medicamentosa entre dois fármacos
  (:Paciente)-[:USA]->(:Medicamento)  // paciente faz uso do medicamento
  (:Paciente)-[:TEM_CONDICAO]->(:Condicao)  // paciente tem a condição/diagnóstico
  (:Documento)-[:TEM_TRECHO]->(:Chunk)-[:MENCIONA]->(entidade)
*/
