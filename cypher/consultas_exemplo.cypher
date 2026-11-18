// Consultas de exemplo para demonstrar no Neo4j Browser (http://localhost:7474)

// 1. Visão do grafo de uma paciente (2 saltos)
MATCH p = (:Paciente {id: 'maria jose ferreira'})-[*1..2]-(x)
WHERE NOT x:Chunk AND NOT x:Documento
RETURN p;

// 2. Multi-hop: remédios em uso por cada paciente que interagem com outro remédio em uso (2-3 saltos)
MATCH (p:Paciente)-[:USA]->(a:Medicamento)-[i:INTERAGE_COM]-(b:Medicamento)<-[:USA]-(p)
WHERE a.nome < b.nome
RETURN p.nome AS paciente, a.nome, b.nome, i.gravidade, i.efeito;

// 3. KAG: o ibuprofeno é seguro para a Maria José?
MATCH (p:Paciente {id: 'maria jose ferreira'})-[:USA]->(m:Medicamento)-[i:INTERAGE_COM]-(c:Medicamento {id: 'ibuprofeno'})
RETURN m.nome AS em_uso, i.gravidade, i.efeito;

// 4. Antitérmicos sem contraindicação na dengue
MATCH (m:Medicamento)-[:TRATA]->(:Condicao {id: 'febre'})
WHERE NOT (m)-[:CONTRAINDICADO_EM]->(:Condicao {id: 'dengue'})
RETURN m.nome;

// 5. Caminho mais curto entre dois medicamentos
MATCH p = shortestPath((a:Medicamento {id: 'fluoxetina'})-[*..6]-(b:Medicamento {id: 'clonazepam'}))
RETURN p;

// 6. PageRank e comunidades (calculados pelo GDS na ingestão)
MATCH (m:Medicamento) RETURN m.nome, round(m.pagerank, 3) AS pagerank, m.comunidade ORDER BY pagerank DESC;

// 7. Proveniência: de que trechos veio uma interação
MATCH (:Medicamento {id: 'varfarina'})-[i:INTERAGE_COM]-(:Medicamento {id: 'ibuprofeno'})
UNWIND i.fontes AS fonte
MATCH (c:Chunk {id: fonte})<-[:TEM_TRECHO]-(d:Documento)
RETURN d.arquivo, c.texto;

// 8. Busca vetorial direta (precisa do embedding da pergunta como parâmetro $e)
// CALL db.index.vector.queryNodes('chunk_embedding', 5, $e) YIELD node, score RETURN node.texto, score;

// 9. Fulltext com tolerância a erro de digitação
CALL db.index.fulltext.queryNodes('entidade_nome', 'warfarina~1') YIELD node, score
RETURN node.nome, labels(node), score;
