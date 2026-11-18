# FarmaGraph RAG — GraphRAG/KAG para interações medicamentosas

Projeto final da disciplina **Tendências em Ciência da Computação** (prof. Ricardo Roberto de Lima).

Sistema de perguntas e respostas sobre medicamentos que combina **busca vetorial** com **travessia de um grafo de conhecimento no Neo4j** para responder perguntas *multi-hop* do tipo:

> "A Maria José pode tomar ibuprofeno para a dor no joelho?"

A resposta exige conectar fatos que estão em documentos diferentes:
prontuário (Maria José **usa** Marevan) → resolução de entidades (Marevan = **varfarina**) → bula (varfarina **INTERAGE_COM** ibuprofeno, gravidade alta, sangramento). Uma busca vetorial pura encontra trechos parecidos com a pergunta, mas não faz esse encadeamento; o LLM sozinho não conhece o prontuário e tende a alucinar.

> ⚠️ Base didática e fictícia (bulas simplificadas e pacientes inventados). Não é orientação médica.

## Arquitetura

```
 documentos (bulas, prontuários)          pergunta do usuário
          │                                       │
 1. Text-to-Graph                         3. Recuperação híbrida
   chunk por seção ─► LLM (JSON Schema)     ├─ índice VECTOR (cosseno) nos trechos
   restrições semânticas                    ├─ ligação de entidades: dicionário + índice FULLTEXT
   resolução de entidades                   ├─ travessia Cypher 1..N saltos
          │                                 ├─ raciocínio KAG (interações × pacientes)
          ▼                                 ├─ shortestPath entre entidades da pergunta
 2. Neo4j (LPG)  ◄──────────────────────────├─ Text-to-Cypher pelo LLM (somente leitura)
   constraints, RANGE, FULLTEXT, VECTOR     └─ ranking por PageRank (GDS)
   PageRank + Louvain (GDS)                         │
                                            4. LLM local gera a resposta citando [K#][G#][T#]
                                               verificação de alucinação + nível de confiança
                                               Streamlit + subgrafo interativo (pyvis)
```

| Pilar do enunciado | Onde está |
|---|---|
| NER/RE com LLM, esquema pré-definido | `app/extraction.py` (saída JSON Schema do Ollama, extração guiada por seção) |
| Resolução de entidades | `app/resolution.py` (normalização, dicionário de nomes comerciais, tokens, similaridade) |
| Modelo LPG, constraints e índices | `app/schema.py`, `cypher/schema.cypher` |
| Busca híbrida, multi-hop, algoritmos | `app/retrieval.py`, `app/graph.py` (PageRank e Louvain via GDS) |
| Redução de alucinação | restrições na extração + contexto do grafo + "Conclusão do grafo" quando o LLM ignora um alerta KAG + verificação pós-resposta (`app/rag.py`) |
| Interface e explicabilidade | `streamlit_app.py`, `app/viz.py` |
| Avaliação | `app/evaluate.py` → `data/avaliacao.json` |
| Testes | `tests/` (`pytest`) |

## Modelo do grafo (LPG)

```
(:Paciente)-[:USA]->(:Medicamento)-[:INTERAGE_COM {gravidade, efeito, fontes}]-(:Medicamento)
(:Paciente)-[:TEM_CONDICAO]->(:Condicao)<-[:TRATA|CONTRAINDICADO_EM]-(:Medicamento)
(:Medicamento)-[:PODE_CAUSAR]->(:EfeitoAdverso)
(:Medicamento)-[:PERTENCE_A_CLASSE]->(:ClasseTerapeutica)
(:Documento)-[:TEM_TRECHO]->(:Chunk {texto, embedding})-[:MENCIONA]->(entidade)
```

- **Constraints de unicidade** em `id` para todos os rótulos (cada uma cria um índice RANGE).
- **RANGE**: `Medicamento.nome`, `Paciente.nome`, `Chunk.documento_id`, `INTERAGE_COM.gravidade`.
- **FULLTEXT**: `entidade_nome` (nome + nomes comerciais) e `chunk_texto`.
- **VECTOR**: `chunk_embedding` (768 dimensões, cosseno, `nomic-embed-text`).
- Toda relação guarda `fontes` (ids dos trechos de origem): proveniência até o texto.

## Como rodar (Windows / PowerShell)

Pré-requisitos: Docker Desktop, [Ollama](https://ollama.com) e Python 3.12.

```powershell
# 1. Neo4j com APOC + Graph Data Science
docker compose up -d

# 2. Modelos locais
ollama pull llama3.2
ollama pull nomic-embed-text

# 3. Ambiente Python
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt

# 4. Ingestão do corpus (cria esquema, extrai o grafo, gera embeddings, PageRank/Louvain)
.\.venv\Scripts\python -m app.ingest --reset

# 5. Interface
.\.venv\Scripts\python -m streamlit run streamlit_app.py
```

Ou simplesmente `.\iniciar.ps1` (sobe Neo4j, Ollama e a interface).

- Interface: <http://localhost:8501>
- Neo4j Browser: <http://localhost:7474> (usuário `neo4j`, senha `farmagraph123` — credencial local de desenvolvimento)
- Avaliação: `.\.venv\Scripts\python -m app.evaluate`
- Artigo (SBC, .docx e .md com os números da avaliação): `.\.venv\Scripts\python docs\gerar_artigo.py`
- Testes: `.\.venv\Scripts\python -m pytest`

Configurações em `.env` (veja `.env.example`).

## Dados

- `corpus/`: 22 bulas simplificadas e 7 prontuários fictícios (texto livre).
- `data/vocabulario.json`: dicionário controlado de princípios ativos, nomes comerciais e sinônimos (usado só na resolução de entidades; não contém relações).
- `data/gabarito.json`: fatos de referência usados **apenas na avaliação** (precisão/recall da extração e fatos esperados nas respostas).

## Bibliotecas e créditos

Neo4j 5 (Community) + APOC + Graph Data Science, driver `neo4j` para Python, Ollama (`llama3.2`, `nomic-embed-text`), Streamlit, pyvis (vis-network), networkx (fallback dos algoritmos), pypdf e python-docx. O exemplo JurisGraphRAG fornecido pelo professor serviu de referência conceitual; este código foi escrito do zero para outro domínio e com Neo4j.

## Limitações

- O LLM local (3B parâmetros) erra parte das extrações; as restrições semânticas bloqueiam a maioria dos erros, mas reduzem o recall.
- O dicionário de nomes comerciais precisa ser mantido (em produção viria de uma base oficial, como a lista DCB/ANVISA).
- Sem autenticação: usar apenas em rede local.
