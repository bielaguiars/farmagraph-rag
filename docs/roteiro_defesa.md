# Roteiro da defesa — FarmaGraph RAG

Tempo sugerido: 12–15 min de apresentação e demonstração, mais as perguntas.

## Antes de entrar na sala (checklist)

1. Docker Desktop aberto. Se ele fechar sozinho com erro de "dockerInference" ou "engine.sock", feche, renomeie as pastas
   `%LOCALAPPDATA%\Docker\run` e `%LOCALAPPDATA%\docker-secrets-engine` (ex.: acrescente `.old`) e abra de novo.
2. Na pasta do projeto: `.\iniciar.ps1`. A interface abre em <http://localhost:8501>.
3. Faça **uma pergunta de aquecimento** antes da apresentação: a primeira chamada carrega o modelo na GPU (~1 min).
4. Deixe o Neo4j Browser aberto em outra aba (<http://localhost:7474>, usuário `neo4j`) com as consultas de
   `cypher/consultas_exemplo.cypher` à mão.
5. Tudo roda **offline**: não depende de internet nem de API paga.

## Estrutura da fala

### 1. Problema (1 min)
- LLMs alucinam e não fazem bem raciocínio multi-etapa. Em saúde isso é perigoso.
- Exemplo: "A Maria José pode tomar ibuprofeno?". O LLM não sabe quem é a Maria José; o RAG vetorial acha a bula do
  ibuprofeno, mas não sabe que ela usa **Marevan** e que Marevan é **varfarina**.
- A resposta certa está espalhada em 3 lugares: prontuário → dicionário de nomes comerciais → bula.

### 2. Domínio e dados (1 min)
- 22 bulas simplificadas (seções fixas, como manda a RDC 47/2009 da ANVISA) e 7 prontuários fictícios.
- Gabarito separado, usado **só para avaliar**.

### 3. Arquitetura — os 4 pilares do enunciado (3 min)
Mostre o diagrama do README.
1. **Text-to-Graph:** o texto é dividido por seção da bula; o llama3.2 local extrai entidades e relações com saída
   em **JSON Schema** (enums de rótulos e relações). Seis travas barram alucinações (ver aba Ingestão).
2. **Neo4j LPG:** 5 rótulos de domínio + Documento/Chunk; 7 relações; constraints de unicidade, índices RANGE,
   FULLTEXT e VECTOR (aba Esquema Neo4j).
3. **Recuperação híbrida:** vetor + fulltext para achar o ponto de partida; Cypher de 1..N saltos; consultas KAG de
   2–3 saltos; shortestPath; PageRank para ranquear o contexto; Text-to-Cypher do LLM validado como só leitura.
4. **Interface:** resposta com citações, subgrafo interativo, Cypher executado, nós consultados e nível de confiança.

### 4. Demonstração (5–6 min)
Use os botões de exemplo da barra lateral: um clique preenche a pergunta e já roda a busca (~25 s cada).
Para comparar com o RAG vetorial, ligue "Comparar com RAG só vetorial" **antes** de clicar no exemplo.

| Pergunta | O que mostrar |
|---|---|
| A Maria José pode tomar ibuprofeno para a dor no joelho? | Alerta K: varfarina × ibuprofeno (alta, sangramento) e losartana × ibuprofeno. No subgrafo: estrela na paciente e no ibuprofeno, aresta vermelha. Abra "Consultas Cypher executadas" e mostre a 4a (KAG). |
| Que remédio para febre é seguro para a Luiza, com suspeita de dengue? | Consulta 4c: paracetamol e dipirona sem alerta; AAS e ibuprofeno contraindicados na dengue. |
| Como a fluoxetina se relaciona com o clonazepam? | shortestPath: fluoxetina → tramadol ← clonazepam (2 saltos). |
| Quais medicamentos interagem com a varfarina? | Text-to-Cypher: mostre a consulta gerada pelo LLM e a validação via EXPLAIN. |
| (ligue "Comparar com RAG só vetorial") A Ana Beatriz pode usar Tramal? | O RAG vetorial não liga a paciente à fluoxetina; o GraphRAG acha a síndrome serotoninérgica. |

Depois, aba **Explorar o grafo**: colorir por comunidade (Louvain) e mostrar o ranking de PageRank.
Aba **Ingestão**: relações barradas por motivo e a tabela de resolução (Marevan → varfarina, Klaricid → claritromicina).

### 5. Avaliação (2 min)
Aba **Avaliação**: precisão/recall da extração por tipo de relação e a tabela LLM × RAG vetorial × GraphRAG.
Diga os números (estão em `data/avaliacao.json` e no artigo).

### 6. Limitações e próximos passos (1 min)
- Modelo de 3B erra parte das extrações; as travas trocam recall por precisão.
- Dicionário de nomes comerciais precisaria vir de base oficial.
- Próximos passos: modelo maior, Leiden + resumos de comunidade (Global Search do GraphRAG), avaliação com RAGAS.

## Perguntas prováveis do professor

**Por que não só RAG vetorial?** Similaridade de cosseno acha trechos parecidos, mas não encadeia fatos de
documentos diferentes. A pergunta da Maria José precisa de 3 saltos: Paciente -USA-> Medicamento -INTERAGE_COM-> Medicamento.

**Onde está o multi-hop?** Na expansão `(s)-[*1..2]-(n)`, nas consultas KAG (`(p)-[:USA]->(m)-[:INTERAGE_COM]-(c)`, e
`(p)-[:TEM_CONDICAO]->(cond)<-[:CONTRAINDICADO_EM]-(c)`) e no `shortestPath`.

**Como vocês reduzem alucinação?** Em três camadas: (1) na extração, JSON Schema + relações permitidas por seção e
por tipo de documento + domínio/contradomínio + ancoragem textual (a entidade tem de estar escrita no trecho);
(2) na geração, o LLM recebe fatos verificados por Cypher e precisa citar [K#]/[G#]/[T#]; se o grafo concluiu um
alerta e o LLM não o citou, a conclusão do grafo entra no início da resposta ("Conclusão do grafo: ..."); (3) depois
da geração, um verificador procura medicamentos/doenças na resposta que não estão no contexto e checa se as
citações existem. No teste, o LLM copiou "Aldactone" do exemplo do prompt numa bula que não falava dele, e a
ancoragem bloqueou.

**Por que às vezes aparece "Conclusão do grafo:" na resposta?** O modelo local tem só 3B parâmetros. Nos testes, ele
chegou a responder "não encontrei" com o alerta da sinvastatina × claritromicina no contexto, e a dizer que a
fluoxetina interage direto com o clonazepam (o grafo mostra que eles só se ligam pelo tramadol). Por isso, quando o
grafo conclui algo (alertas K), o LLM recebe **só os alertas** e apenas os redige: o raciocínio é do grafo (KAG). Se
mesmo assim ele não citar o alerta, a conclusão verificada é inserida e identificada como tal. É transparente: o
usuário vê o que veio do grafo e o que veio do LLM.

**Por que o modelo separa "dor e febre"?** O LLM às vezes devolve uma lista como uma entidade só. Sem separar, o
paracetamol não apareceria como remédio para febre. A separação não acontece quando a segunda parte é adjetivo
("insuficiência renal e hepática").

**Como funciona a resolução de entidades?** Normalização (caixa, acento, dose, artigo), dicionário controlado de
nomes comerciais e sinônimos, correspondência por tokens (paciente "Maria José" → "Maria José Ferreira") e
similaridade aproximada (warfarina → varfarina). Os nós são gravados com `MERGE` no id normalizado, então a mesma
entidade vinda de documentos diferentes vira um nó só.

**Para que servem os índices?** Constraint de unicidade garante o MERGE idempotente e cria índice RANGE em `id`.
FULLTEXT liga termos da pergunta a entidades, com tolerância a erro (`warfarina~1`). VECTOR faz a busca semântica
nos trechos. RANGE em `gravidade` acelera o filtro de interações graves.

**Por que PageRank e Louvain?** PageRank mede centralidade: medicamentos com muitas interações (varfarina) sobem e
entram primeiro no contexto. Louvain agrupa as interações em "clusters de risco" (anticoagulação/sangramento,
potássio/rim/lítio, serotonina, estatina).

**Como é calculada a confiança?** Média ponderada de 4 sinais: similaridade vetorial do melhor trecho (25%),
suporte do grafo — houve alerta ou caminho (25%), fração das entidades da resposta presentes no contexto (35%) e
fração de citações válidas (15%).

**O Text-to-Cypher é seguro?** A consulta gerada passa por uma lista de bloqueio (CREATE, MERGE, DELETE, SET,
CALL...) e por um `EXPLAIN` no Neo4j: só executa se o tipo da consulta for `r` (somente leitura).

**Por que LLM local?** Dados de saúde são sensíveis (LGPD); Ollama roda na máquina, sem enviar nada a terceiros.

**Qual a diferença entre GraphRAG e KAG aqui?** GraphRAG: o subgrafo vira contexto para o LLM. KAG: parte do
raciocínio é feita de forma lógica no próprio grafo (as consultas 4a/4b/4c produzem conclusões verificadas que o
LLM só verbaliza).
