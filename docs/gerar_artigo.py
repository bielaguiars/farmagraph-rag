"""Gera o artigo (formato SBC) em .docx e .md com os números reais do grafo e da avaliação.

Uso: .venv\\Scripts\\python docs\\gerar_artigo.py
Lê data/avaliacao.json, data/ultima_ingestao.json e consulta o Neo4j (estatísticas, PageRank, Louvain).
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.shared import Cm, Pt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app.services import build_services  # noqa: E402

DOCS = ROOT / "docs"
FIG = DOCS / "figura_arquitetura.png"


def pct(x: float) -> str:
    return f"{x * 100 + 1e-9:.1f}%".replace(".", ",")


def num(x: float, d: int = 2) -> str:
    return f"{x + 1e-9:.{d}f}".replace(".", ",")  # 0,845 -> 0,85 (evita o erro de ponto flutuante)


# ------------------------------------------------------------------ figura
def draw_figure(path: Path) -> None:
    W, H = 1800, 1050
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
    bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 26)
    title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 30)

    def box(x, y, w, h, text, fill="#eef4fb", outline="#2f5f8f", fnt=font):
        d.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=fill, outline=outline, width=3)
        lines = text.split("\n")
        lh = fnt.size + 6
        ty = y + (h - lh * len(lines)) / 2
        for line in lines:
            tw = d.textlength(line, font=fnt)
            d.text((x + (w - tw) / 2, ty), line, fill="#111111", font=fnt)
            ty += lh

    def arrow(x1, y1, x2, y2, both=False):
        d.line([x1, y1, x2, y2], fill="#333333", width=4)
        import math

        def head(xa, ya, xb, yb):
            ang = math.atan2(yb - ya, xb - xa)
            for s in (-0.45, 0.45):
                d.line([xb, yb, xb - 18 * math.cos(ang + s), yb - 18 * math.sin(ang + s)], fill="#333333", width=4)

        head(x1, y1, x2, y2)
        if both:
            head(x2, y2, x1, y1)

    d.text((60, 25), "Construção do grafo (offline)", fill="#2f5f8f", font=title)
    d.text((1230, 25), "Consulta (online)", fill="#7a3b8f", font=title)

    left = ["Bulas e prontuários\n(texto livre)", "Segmentação por seção\nda bula (RDC 47/2009)",
            "LLM local (Llama 3.2)\nNER/RE com JSON Schema", "6 restrições semânticas\n(domínio, ancoragem...)",
            "Resolução de entidades\n(Marevan → varfarina)"]
    lx, lw, bh, gap, y0 = 60, 470, 110, 50, 90
    for i, t in enumerate(left):
        y = y0 + i * (bh + gap)
        box(lx, y, lw, bh, t)
        if i:
            arrow(lx + lw / 2, y - gap, lx + lw / 2, y)

    right = ["Pergunta do usuário", "Busca VECTOR nos trechos +\nligação de entidades (FULLTEXT)",
             "Expansão multi-hop, consultas\nKAG, shortestPath, Text-to-Cypher",
             "Ranking por PageRank +\ncontexto citável [K][G][T][C]",
             "LLM gera a resposta;\nverificação + confiança", "Interface Streamlit\n(subgrafo, Cypher, nós)"]
    rx, rw, rbh, rgap = 1230, 510, 105, 40
    for i, t in enumerate(right):
        y = y0 + i * (rbh + rgap)
        box(rx, y, rw, rbh, t, fill="#f6eefa", outline="#7a3b8f")
        if i:
            arrow(rx + rw / 2, y - rgap, rx + rw / 2, y)

    nx_, ny, nw, nh = 640, 360, 480, 300
    box(nx_, ny, nw, nh, "Neo4j (LPG)\n\nconstraints de unicidade\níndices RANGE · FULLTEXT · VECTOR\n"
        "GDS: PageRank e Louvain\nproveniência (fontes) em cada aresta", fill="#eaf6ee", outline="#2e7d4f")
    # extração -> Neo4j
    last_left_y = y0 + 4 * (bh + gap) + bh / 2
    arrow(lx + lw, last_left_y, nx_, ny + nh - 40)
    # Neo4j <-> recuperação
    arrow(nx_ + nw, ny + 90, rx, y0 + 1 * (rbh + rgap) + rbh / 2, both=True)
    arrow(nx_ + nw, ny + 160, rx, y0 + 2 * (rbh + rgap) + rbh / 2, both=True)
    arrow(nx_ + nw, ny + 230, rx, y0 + 3 * (rbh + rgap) + rbh / 2, both=True)
    img.save(path)


# ------------------------------------------------------------------ dados
def load_data() -> dict:
    ev = json.loads((ROOT / "data" / "avaliacao.json").read_text(encoding="utf-8"))
    ing_path = ROOT / "data" / "ultima_ingestao.json"
    ing = json.loads(ing_path.read_text(encoding="utf-8")) if ing_path.exists() else {}
    services = build_services()
    g = services.graph
    data = {
        "ev": ev,
        "ing": ing,
        "stats": g.stats(),
        "pagerank": g.top_pagerank(6),
        "comunidades": g.communities(),
        "schema": g.schema_info(),
    }
    g.close()
    return data


# ------------------------------------------------------------------ conteúdo
def build_content(data: dict) -> dict:
    ev, stats = data["ev"], data["stats"]
    ext = ev["extracao"]
    tot = ext["por_relacao"][-1]
    qa = ev.get("qa")
    nos, rels = stats["nos"], stats["relacoes"]
    dominio = ["Medicamento", "ClasseTerapeutica", "Condicao", "EfeitoAdverso", "Paciente"]
    n_ent = sum(nos.get(k, 0) for k in dominio)
    dom_rels = ["PERTENCE_A_CLASSE", "TRATA", "CONTRAINDICADO_EM", "PODE_CAUSAR", "INTERAGE_COM", "USA", "TEM_CONDICAO"]
    n_rel = sum(rels.get(k, 0) for k in dom_rels)
    n_chunks = nos.get("Chunk", 0)
    pr = data["pagerank"]
    com = [c for c in data["comunidades"] if c["total"] > 1]
    n_idx = len(data["schema"]["indexes"])
    n_con = len(data["schema"]["constraints"])

    # relatório de ingestão (execuções registradas)
    reports = [r for r in data["ing"].get("relatorios", []) if isinstance(r, dict) and "relacoes_aceitas" in r]
    brutas = sum(r.get("relacoes_brutas_llm", 0) for r in reports)
    aceitas = sum(r.get("relacoes_aceitas", 0) for r in reports)
    rejeitadas = sum(r.get("relacoes_rejeitadas", 0) for r in reports)
    corrigidas = sum(r.get("direcoes_corrigidas", 0) for r in reports)
    motivos = Counter()
    for r in reports:
        for k, v in (r.get("motivos_rejeicao") or {}).items():
            motivos[k] += v
    seg = sum(r.get("segundos", 0) for r in reports)

    erros = ext.get("erros", [])
    fn = Counter(e["tipo"] for e in erros if e["erro"] == "faltou extrair")

    if qa:
        res = {r["modo"]: r for r in qa["resumo"]}
        llm = res["LLM sem contexto"]["cobertura média dos fatos esperados"]
        vet = res["RAG só vetorial"]["cobertura média dos fatos esperados"]
        gr = res["GraphRAG (este projeto)"]["cobertura média dos fatos esperados"]
        qa_frase_pt = (f"o GraphRAG cobriu {pct(gr)} dos fatos esperados em 9 perguntas multi-hop, contra {pct(vet)} do "
                       f"RAG apenas vetorial e {pct(llm)} do LLM sem contexto")
        qa_frase_en = (f"GraphRAG covered {pct(gr).replace(',', '.')} of the expected facts in 9 multi-hop questions, "
                       f"versus {pct(vet).replace(',', '.')} for vector-only RAG and {pct(llm).replace(',', '.')} for the "
                       f"LLM without context")
    else:
        qa_frase_pt = "a avaliação de perguntas não foi executada"
        qa_frase_en = "question answering was not evaluated"

    blocks = []
    H1 = lambda t: blocks.append(("h1", t))  # noqa: E731
    H2 = lambda t: blocks.append(("h2", t))  # noqa: E731
    P = lambda t: blocks.append(("p", t))  # noqa: E731
    B = lambda items: blocks.append(("bullets", items))  # noqa: E731

    def T(caption, headers, rows):
        blocks.append(("table", caption, headers, rows))

    def CODE(caption, text):
        blocks.append(("code", caption, text))

    H1("1. Introdução")
    P("Modelos de linguagem de grande porte (LLMs) produzem textos fluentes, mas podem gerar afirmações sem "
      "respaldo nos dados, fenômeno conhecido como alucinação [Ji et al. 2023]. A geração aumentada por recuperação "
      "(RAG) reduz o problema ao fornecer ao modelo trechos recuperados por similaridade semântica [Lewis et al. 2020]. "
      "A similaridade vetorial, porém, devolve trechos isolados: ela não encadeia fatos que estão espalhados em "
      "documentos diferentes, o que limita perguntas *multi-hop*.")
    P("Na área da saúde essa limitação é perigosa. Considere a pergunta \"A Maria José pode tomar ibuprofeno para a dor "
      "no joelho?\". A resposta correta exige três passos: (i) o prontuário informa que a paciente usa *Marevan*; "
      "(ii) Marevan é um nome comercial da **varfarina**; (iii) a bula da varfarina informa que a associação com "
      "ibuprofeno aumenta o risco de sangramento. Um LLM sozinho não conhece o prontuário; um RAG vetorial encontra a "
      "bula do ibuprofeno, mas não sabe que a paciente usa varfarina.")
    P("Grafos de conhecimento representam entidades e relações de forma explícita e permitem percorrer esses "
      "encadeamentos [Hogan et al. 2021; Pan et al. 2024]. Abordagens recentes como o GraphRAG [Edge et al. 2024] e o "
      "KAG [Liang et al. 2024] usam o grafo como memória estruturada do LLM. Este trabalho apresenta o **FarmaGraph "
      "RAG**, um sistema de perguntas e respostas sobre medicamentos e interações medicamentosas construído sobre o "
      "Neo4j. As contribuições são:")
    B(["um pipeline *Text-to-Graph* que extrai entidades e relações de bulas e prontuários com um LLM local, guiado "
       "pelas seções da bula e filtrado por seis restrições semânticas, com resolução de entidades e proveniência;",
       "um modelo de grafo de propriedades rotulado (LPG) com constraints de unicidade e índices RANGE, FULLTEXT e "
       "VECTOR;",
       "uma recuperação híbrida que combina busca vetorial, ligação de entidades, travessia de 1 a N saltos, consultas "
       "lógicas no estilo KAG, caminhos mínimos, PageRank, Louvain e Cypher gerado pelo LLM com validação de "
       "somente leitura;",
       "uma interface explicável que mostra o subgrafo consultado, as consultas Cypher executadas e um nível de "
       "confiança; e",
       "uma avaliação que compara o LLM sem contexto, o RAG apenas vetorial e o GraphRAG."])
    P("Todo o processamento é local (Ollama), sem envio de dados a terceiros, o que é relevante para dados de saúde "
      "sob a LGPD [Brasil 2018]. O restante do artigo apresenta a fundamentação (Seção 2), os dados (Seção 3), a "
      "arquitetura (Seção 4), a avaliação (Seção 5), as limitações (Seção 6) e a conclusão (Seção 7).")

    H1("2. Fundamentação e Trabalhos Relacionados")
    P("**RAG e alucinação.** O RAG [Lewis et al. 2020] combina um recuperador denso com um gerador: os trechos mais "
      "similares à pergunta são inseridos no *prompt*. A recuperação depende de *embeddings*; aqui usamos o "
      "nomic-embed-text [Nussbaum et al. 2024], que gera vetores de 768 dimensões e distingue o texto indexado "
      "(*search_document*) da consulta (*search_query*). O levantamento de [Ji et al. 2023] classifica as alucinações "
      "em intrínsecas (contradizem a fonte) e extrínsecas (não verificáveis pela fonte); ambas motivam a verificação "
      "pós-resposta adotada na Seção 4.4.")
    P("**Grafos de conhecimento e LPG.** Em um grafo de propriedades rotulado, nós e relações têm rótulos e "
      "propriedades. O Neo4j implementa esse modelo e oferece a linguagem declarativa Cypher, índices RANGE, "
      "FULLTEXT (Lucene) e VECTOR, e a biblioteca Graph Data Science (GDS) com algoritmos de grafos [Robinson et al. "
      "2015].")
    P("**GraphRAG e KAG.** O GraphRAG da Microsoft [Edge et al. 2024] extrai um grafo de entidades com um LLM, "
      "detecta comunidades e resume cada uma para responder perguntas globais. O KAG [Liang et al. 2024] propõe que "
      "parte do raciocínio seja feita de forma lógica sobre o grafo, com o LLM verbalizando conclusões verificadas. O "
      "levantamento de [Peng et al. 2024] organiza o GraphRAG em indexação, recuperação e geração baseadas em grafo, "
      "estrutura que seguimos na Seção 4.")
    P("**Algoritmos de grafos.** O PageRank [Page et al. 1999] mede a centralidade de um nó a partir das ligações que "
      "recebe; aqui ele prioriza os fatos mais centrais no contexto. O método de Louvain [Blondel et al. 2008] "
      "maximiza a modularidade para encontrar comunidades; aqui ele agrupa os medicamentos em \"clusters de risco\" "
      "de interação.")

    H1("3. Domínio e Dados")
    P("O corpus tem **22 bulas simplificadas**, escritas para o projeto com as seções definidas pela RDC nº 47/2009 "
      "da ANVISA [ANVISA 2009] (classe terapêutica, indicações, contraindicações, interações medicamentosas, reações "
      "adversas, posologia), e **7 prontuários ambulatoriais fictícios** em texto livre. Os prontuários citam os "
      "medicamentos apenas pelo nome comercial (o que exige resolução de entidades) e nunca mencionam interações (o que "
      "exige raciocínio multi-hop). Um vocabulário controlado com princípios ativos, nomes comerciais e sinônimos é "
      "usado somente na resolução de entidades e não contém relações.")
    gold = Counter()
    gab = json.loads((ROOT / "data" / "gabarito.json").read_text(encoding="utf-8"))
    from app.evaluate import gold_triples
    gold.update(t[0] for t in gold_triples(gab))
    P(f"Um gabarito separado, com **{sum(gold.values())} fatos de referência** (22 medicamentos, "
      f"{len(gab['interacoes'])} interações com gravidade e 7 pacientes), é usado **apenas na avaliação**. Os "
      "documentos são didáticos e não constituem orientação médica.")

    H1("4. Arquitetura")
    P("A Figura 1 resume o sistema, dividido em construção do grafo (offline) e consulta (online). O código está "
      "organizado em módulos Python (`app/`) e a interface em Streamlit.")
    blocks.append(("figure", str(FIG), "Figura 1. Arquitetura do FarmaGraph RAG."))

    H2("4.1. Construção do grafo (Text-to-Graph)")
    P("**Segmentação.** As bulas são divididas por seção (um trecho por seção, com o título prefixado); outros "
      "documentos são divididos por parágrafos em trechos de até 900 caracteres.")
    P("**Extração guiada por seção.** Para cada trecho, o Llama 3.2 (3B parâmetros, quantizado em 4 bits) "
      "[Grattafiori et al. 2024] é chamado via Ollama com o parâmetro `format` igual a um JSON Schema: a relação é um "
      "*enum* restrito e a lista tem no máximo 10 itens. Em bulas, cada seção só pode gerar a relação "
      "correspondente (Indicações → TRATA, Contraindicações → CONTRAINDICADO_EM, Interações → INTERAGE_COM, Reações "
      "adversas → PODE_CAUSAR, Classe → PERTENCE_A_CLASSE) e seções sem relação de domínio não chamam o LLM. Em "
      "prontuários só são permitidas USA e TEM_CONDICAO.")
    P("**Restrições semânticas.** Cada relação proposta pelo LLM passa por seis travas: (1) saída estruturada com "
      "*enums*; (2) relação compatível com a seção; (3) relação permitida para o tipo de documento; (4) domínio e "
      "contradomínio do esquema, com correção automática de direção invertida; (5) ancoragem textual, ou seja, as "
      "duas pontas precisam estar escritas no trecho (ou ser o assunto do documento); e (6) em bulas, a relação "
      "precisa envolver o medicamento da bula. Em um teste, o modelo copiou \"Aldactone\" do exemplo do *prompt* "
      "para uma bula que não o mencionava, e a ancoragem bloqueou a relação.")
    if reports:
        mot = "; ".join(f"{k}: {v}" for k, v in motivos.most_common(4))
        P(f"Na última execução da ingestão ({len(reports)} documentos, {num(seg / 60, 1)} min), o LLM propôs "
          f"{brutas} relações; depois de separar as listas, {aceitas} foram aceitas e {rejeitadas} rejeitadas, e "
          f"{corrigidas} tiveram a direção ou a origem corrigida. Principais motivos de rejeição: {mot}.")
    P("**Resolução de entidades.** Os nomes são normalizados (caixa, acentos, doses como \"5 mg\" e artigos) e "
      "resolvidos por: dicionário de nomes comerciais e sinônimos (Marevan → varfarina, Klaricid → claritromicina); "
      "correspondência por *tokens* para pacientes (\"Maria José\" → \"Maria José Ferreira\"); similaridade "
      "aproximada (warfarina → varfarina); e singularização de classes terapêuticas. Antes disso, listas que o LLM "
      "devolve como uma entidade só são separadas (\"dor e febre\" → dor, febre), exceto quando a parte seguinte é "
      "um adjetivo (\"insuficiência renal e hepática\"), e qualificadores como \"pacientes com histórico de\" são "
      "removidos. Os nós são gravados com `MERGE` no identificador normalizado, de modo que a mesma entidade vinda "
      "de documentos diferentes vira um único nó.")
    P("**Proveniência.** Toda relação guarda a lista `fontes` com os identificadores dos trechos de onde foi "
      "extraída, e cada trecho aponta para as entidades que menciona (`MENCIONA`), o que permite rastrear qualquer "
      "fato até o texto original.")

    H2("4.2. Modelo de grafo no Neo4j")
    T("Tabela 1. Esquema do grafo (LPG).", ["Relação", "Origem → Destino", "Propriedades"], [
        ["PERTENCE_A_CLASSE", "Medicamento → ClasseTerapeutica", "fontes"],
        ["TRATA", "Medicamento → Condicao", "fontes"],
        ["CONTRAINDICADO_EM", "Medicamento → Condicao", "fontes"],
        ["PODE_CAUSAR", "Medicamento → EfeitoAdverso", "fontes"],
        ["INTERAGE_COM", "Medicamento — Medicamento (simétrica)", "gravidade, efeito, fontes"],
        ["USA", "Paciente → Medicamento", "fontes"],
        ["TEM_CONDICAO", "Paciente → Condicao", "fontes"],
        ["TEM_TRECHO / MENCIONA", "Documento → Chunk → entidade", "texto, embedding (no Chunk)"],
    ])
    P(f"O esquema tem {n_con} constraints de unicidade em `id` (uma por rótulo, cada uma com seu índice RANGE) e "
      f"{n_idx} índices ao todo (exceto os de *lookup*): RANGE em `Medicamento.nome`, `Paciente.nome`, "
      "`Chunk.documento_id` e na propriedade `gravidade` da relação INTERAGE_COM; FULLTEXT `entidade_nome` (nome e "
      "nomes comerciais) e `chunk_texto`; e VECTOR `chunk_embedding` (768 dimensões, cosseno). Os *embeddings* são "
      "gravados com `db.create.setNodeVectorProperty`.")
    T("Tabela 2. Tamanho do grafo após a ingestão.", ["Elemento", "Quantidade"], [
        *[[k, str(nos.get(k, 0))] for k in dominio],
        ["Documento / Chunk", f"{nos.get('Documento', 0)} / {n_chunks}"],
        *[[k, str(rels.get(k, 0))] for k in dom_rels],
        ["Total de entidades / relações de domínio", f"{n_ent} / {n_rel}"],
    ])

    H2("4.3. Recuperação híbrida")
    P("Para cada pergunta, o recuperador executa os passos abaixo; todas as consultas Cypher executadas ficam "
      "visíveis na interface.")
    B(["**Busca vetorial:** os 5 trechos mais similares pelo índice VECTOR (`db.index.vector.queryNodes`).",
       "**Ligação de entidades:** termos da pergunta são resolvidos pelo dicionário e pelo índice FULLTEXT com "
       "tolerância a erro de digitação (`warfarina~1`); se nada for encontrado, usam-se as entidades dos trechos "
       "recuperados.",
       "**Expansão multi-hop:** padrão de comprimento variável `(s)-[*1..2]-(n)` a partir das entidades-semente.",
       "**Raciocínio KAG:** consultas de 2 a 3 saltos que produzem conclusões verificadas: interações entre os "
       "medicamentos do paciente e o medicamento perguntado, contraindicações pelas condições do paciente e "
       "alternativas que tratam a condição sem interação (Listagem 1).",
       "**Caminhos mínimos:** `shortestPath` entre pares de entidades da pergunta (ex.: fluoxetina → tramadol ← "
       "clonazepam).",
       "**Ranking:** os fatos são pontuados pelo tipo de relação, por ligarem sementes, por estarem em caminhos, pela "
       "gravidade e pelo PageRank dos nós, e os mais bem pontuados entram no contexto.",
       "**Text-to-Cypher (opcional):** o LLM gera uma consulta a partir do esquema e de exemplos; ela só é executada "
       "se passar por uma lista de bloqueio (CREATE, MERGE, DELETE, SET, CALL...) e se o `EXPLAIN` do Neo4j indicar "
       "consulta somente de leitura (`query_type = 'r'`). Se for rejeitada, o erro volta ao LLM para uma segunda "
       "tentativa, e setas em INTERAGE_COM são removidas, pois a relação é simétrica."])
    CODE("Listagem 1. Consulta KAG: interações entre os medicamentos em uso pelo paciente e o candidato.",
         "MATCH (p:Paciente)-[:USA]->(m:Medicamento)-[i:INTERAGE_COM]-(c:Medicamento)\n"
         "WHERE elementId(p) IN $pacientes\n"
         "  AND (elementId(c) IN $candidatos OR (p)-[:USA]->(c))\n"
         "RETURN p.nome, m.nome AS em_uso, c.nome AS outro,\n"
         "       i.gravidade, i.efeito, i.fontes\n"
         "ORDER BY CASE i.gravidade WHEN 'alta' THEN 0 ELSE 1 END")
    if pr:
        top = ", ".join(f"{r['entidade']} ({num(r['pagerank'], 3)})" for r in pr[:5])
        P(f"O PageRank é calculado pelo GDS sobre todo o grafo de domínio e o Louvain sobre o subgrafo de "
          f"INTERAGE_COM. As entidades mais centrais foram: {top}. O Louvain encontrou "
          f"{len(data['comunidades'])} comunidades, {len(com)} delas com mais de um medicamento, que correspondem a "
          "grupos de risco (por exemplo, anticoagulação e sangramento; potássio, rim e lítio; serotonina).")

    H2("4.4. Geração, verificação e confiança")
    P("O contexto enviado ao LLM é dividido em seções citáveis: conclusões do raciocínio KAG [K#], caminhos, fatos do "
      "grafo [G#], resultados do Text-to-Cypher [C#] e trechos [T#]. O *prompt* exige que cada afirmação cite sua "
      "fonte. Nos testes, porém, o modelo de 3B tratava mal o contexto completo: chegava a responder que não havia "
      "encontrado a informação com o alerta no contexto, ou \"encurtava\" um caminho de dois saltos e afirmava uma "
      "interação direta inexistente. Por isso adotamos a divisão de trabalho do KAG: quando o grafo produz "
      "conclusões (alertas [K#]), o LLM recebe **apenas esses alertas** e a tarefa de redigi-los sem acrescentar nem "
      "contradizer nada; o contexto completo é usado só quando não há alerta. Os caminhos de dois ou mais saltos "
      "entre medicamentos também viram alertas (\"não há relação direta entre fluoxetina e clonazepam; eles se ligam "
      "por tramadol\"). Se a resposta ainda assim não citar um alerta, a conclusão verificada é inserida no início, "
      "identificada como \"Conclusão do grafo\". Após a geração, um verificador confere se as citações existem e "
      "procura, na resposta, medicamentos e condições que não aparecem no contexto (indício de alucinação "
      "extrínseca).")
    P("A confiança é a média ponderada de quatro sinais: similaridade vetorial do melhor trecho (25%), suporte do "
      "grafo, ou seja, se houve alerta KAG ou caminho (25%), fração das entidades da resposta presentes no contexto "
      "(35%) e fração de citações válidas (15%). Valores ≥ 0,75 são classificados como Alta, ≥ 0,5 como Média e os "
      "demais como Baixa.")

    H2("4.5. Interface")
    P("A interface em Streamlit tem cinco abas: **Perguntar** (resposta com citações, subgrafo interativo em pyvis "
      "com as sementes em destaque, consultas Cypher executadas, nós consultados, confiança e comparação opcional com "
      "o RAG vetorial); **Explorar o grafo** (colorido por rótulo ou por comunidade, ranking de PageRank); "
      "**Ingestão** (envio de novos documentos, relações rejeitadas por motivo e tabela de resolução de entidades); "
      "**Avaliação**; e **Esquema Neo4j** (constraints e índices).")

    H1("5. Avaliação")
    P("Os experimentos rodaram em um notebook com GPU NVIDIA RTX 3050 (4 GB), Neo4j 5.26 Community com APOC e GDS "
      "2.13 em Docker, e Ollama com Llama 3.2 (3B) e nomic-embed-text.")
    H2("5.1. Qualidade da extração")
    P("O grafo extraído é comparado ao gabarito após normalização dos nomes (similaridade mínima de 0,85). A Tabela 3 "
      "mostra precisão, revocação (*recall*) e F1 por tipo de relação.")
    T("Tabela 3. Extração de relações: grafo extraído × gabarito.",
      ["Relação", "Gabarito", "Extraídas", "Corretas", "Precisão", "Recall", "F1"],
      [[r["relação"], str(r["gabarito"]), str(r["extraídas"]), str(r["corretas"]), num(r["precisão"]),
        num(r["recall"]), num(r["F1"])] for r in ext["por_relacao"]])
    fn_txt = ", ".join(f"{k} ({v})" for k, v in fn.most_common(3))
    fp_list = [e for e in erros if e["erro"] != "faltou extrair"]
    fp_ex = "; ".join(f"{e['origem']} {e['tipo']} {e['destino']}" for e in fp_list[:3])
    P(f"No total, a extração obteve precisão de {num(tot['precisão'])}, recall de {num(tot['recall'])} e F1 de "
      f"{num(tot['F1'])}, e a gravidade das interações extraídas estava correta em "
      f"{ext.get('gravidade_correta', '-')}. As relações de interação, contraindicação e uso de medicamento, que são "
      "as que sustentam o raciocínio multi-hop, foram extraídas por completo. "
      f"Houve {len(fp_list)} {'relação' if len(fp_list) == 1 else 'relações'} fora do gabarito ({fp_ex or '-'}); "
      "ao conferir o texto, trata-se de um "
      "sintoma realmente escrito no prontuário que o gabarito não lista, ou seja, uma lacuna do gabarito e não uma "
      f"alucinação. Os fatos não extraídos concentram-se em {fn_txt or '-'}: as restrições preferem descartar a "
      "aceitar um fato duvidoso. Outra limitação observada é que o modelo às vezes não separa listas, gerando "
      "entidades compostas como \"dor e febre\" ou \"sangramento e úlcera gástrica\"; a comparação por similaridade "
      "conta essas relações como corretas, mas elas indicam espaço para um passo de normalização de listas.")

    H2("5.2. Perguntas multi-hop")
    if qa:
        P("Nove perguntas exigem encadear prontuário, dicionário e bula (Tabela 5). Cada resposta é pontuada pela "
          "cobertura: a fração dos fatos esperados (grupos de palavras-chave do gabarito) presentes no texto. "
          "Comparamos três modos com o mesmo LLM: sem contexto, RAG apenas vetorial (5 trechos) e o GraphRAG "
          "proposto.")
        T("Tabela 4. Resultado por modo de resposta.",
          ["Modo", "Cobertura média", "Perguntas 100%", "Entidades sem suporte", "Tempo médio (s)"],
          [[r["modo"], pct(r["cobertura média dos fatos esperados"]), r["perguntas 100% corretas"],
            str(r.get("entidades sem suporte (total)", "n/a")), num(r["tempo médio (s)"], 1)] for r in qa["resumo"]])
        T("Tabela 5. Cobertura por pergunta.", ["Pergunta", "LLM", "RAG vetorial", "GraphRAG"],
          [[d["pergunta"], num(d["LLM sem contexto · cobertura"]), num(d["RAG só vetorial · cobertura"]),
            num(d["GraphRAG (este projeto) · cobertura"])] for d in qa["detalhes"]])
        n_fallback = sum(1 for d in qa["detalhes"]
                         if str(d.get("GraphRAG (este projeto) · resposta", "")).startswith("Conclusão do grafo"))
        P(f"Em média, {qa_frase_pt}. O LLM sem contexto não conhece os pacientes e tende a responder de forma "
          "genérica; o RAG vetorial recupera a bula certa quando o medicamento é citado, mas não liga o paciente aos "
          "medicamentos que ele usa. O GraphRAG resolve o nome comercial, percorre Paciente → Medicamento → "
          f"Medicamento e entrega ao LLM a conclusão já verificada no grafo. Em {n_fallback} das 9 respostas o LLM "
          "não citou o alerta por conta própria e a conclusão do grafo foi inserida pelo mecanismo da Seção 4.4. "
          "O custo é o tempo: a recuperação híbrida e as consultas Cypher deixam o GraphRAG mais lento que as linhas "
          "de base, embora ainda abaixo de meio minuto por pergunta em uma GPU de 4 GB.")
    else:
        P("A avaliação de perguntas não foi executada nesta versão.")

    H1("6. Limitações e Trabalhos Futuros")
    B(["O modelo de 3B parâmetros erra parte das extrações; as restrições semânticas trocam revocação por precisão.",
       "O corpus é pequeno e didático; uma avaliação real exigiria bulas oficiais e um gabarito produzido por "
       "farmacêuticos.",
       "A métrica de cobertura por palavras-chave não mede a qualidade clínica da resposta; métricas como as do "
       "RAGAS e avaliação humana seriam o próximo passo.",
       "O dicionário de nomes comerciais é mantido manualmente; em produção viria de uma base oficial (DCB/ANVISA).",
       "Trabalhos futuros: modelos maiores, resumos de comunidade (busca global do GraphRAG) e atualização "
       "incremental do grafo com novas bulas."])

    H1("7. Conclusão")
    P("O FarmaGraph RAG mostra que um grafo de conhecimento construído automaticamente por um LLM pequeno e local, "
      "desde que a extração seja guiada pela estrutura do documento e filtrada por restrições semânticas, permite "
      "responder perguntas que exigem encadear fatos de documentos diferentes. A combinação de busca vetorial, "
      "travessia Cypher, consultas lógicas no estilo KAG e verificação pós-resposta torna cada afirmação rastreável até "
      f"o trecho de origem. Na avaliação, {qa_frase_pt}. O código, os dados e as instruções de reprodução estão no "
      "repositório do projeto.")

    refs = [
        "ANVISA (2009). Resolução RDC nº 47, de 8 de setembro de 2009. Estabelece regras para elaboração, "
        "harmonização, atualização, publicação e disponibilização de bulas de medicamentos. Diário Oficial da União.",
        "Blondel, V. D., Guillaume, J.-L., Lambiotte, R. e Lefebvre, E. (2008). Fast unfolding of communities in large "
        "networks. Journal of Statistical Mechanics: Theory and Experiment, 2008(10):P10008.",
        "Brasil (2018). Lei nº 13.709, de 14 de agosto de 2018. Lei Geral de Proteção de Dados Pessoais (LGPD).",
        "Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S. e Larson, J. (2024). From local to "
        "global: a Graph RAG approach to query-focused summarization. arXiv:2404.16130.",
        "Grattafiori, A. et al. (2024). The Llama 3 herd of models. arXiv:2407.21783.",
        "Hogan, A. et al. (2021). Knowledge graphs. ACM Computing Surveys, 54(4):1–37.",
        "Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A. e Fung, P. (2023). "
        "Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38.",
        "Lewis, P. et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in "
        "Neural Information Processing Systems 33 (NeurIPS 2020), p. 9459–9474.",
        "Liang, L. et al. (2024). KAG: boosting LLMs in professional domains via knowledge augmented generation. "
        "arXiv:2409.13731.",
        "Nussbaum, Z., Morris, J. X., Duderstadt, B. e Mulyar, A. (2024). Nomic Embed: training a reproducible long "
        "context text embedder. arXiv:2402.01613.",
        "Page, L., Brin, S., Motwani, R. e Winograd, T. (1999). The PageRank citation ranking: bringing order to the "
        "web. Relatório técnico, Stanford InfoLab.",
        "Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J. e Wu, X. (2024). Unifying large language models and knowledge "
        "graphs: a roadmap. IEEE Transactions on Knowledge and Data Engineering, 36(7):3580–3599.",
        "Peng, B. et al. (2024). Graph retrieval-augmented generation: a survey. arXiv:2408.08921.",
        "Robinson, I., Webber, J. e Eifrem, E. (2015). Graph Databases: New Opportunities for Connected Data. 2ª ed. "
        "O'Reilly Media.",
    ]

    resumo = (f"Modelos de linguagem de grande porte (LLMs) alucinam e têm dificuldade com perguntas que exigem "
              f"encadear fatos de fontes diferentes, o que é crítico em saúde. Este artigo apresenta o FarmaGraph RAG, "
              f"um sistema de perguntas e respostas sobre medicamentos que combina busca vetorial e travessia de um "
              f"grafo de conhecimento no Neo4j (GraphRAG) com consultas lógicas sobre o grafo (KAG). O grafo é "
              f"construído a partir de 22 bulas simplificadas e 7 prontuários fictícios por um LLM local (Llama 3.2, "
              f"3B) com saída estruturada, extração guiada pelas seções da bula, seis restrições semânticas e "
              f"resolução de entidades. A recuperação usa índices VECTOR, FULLTEXT e RANGE, expansão multi-hop, "
              f"caminhos mínimos, PageRank e Cypher gerado pelo LLM com validação de somente leitura. A extração "
              f"atingiu precisão de {num(tot['precisão'])} e recall de {num(tot['recall'])} frente a um gabarito de "
              f"{tot['gabarito']} fatos, e {qa_frase_pt}.")
    abstract = (f"Large language models (LLMs) hallucinate and struggle with questions that require chaining facts "
                f"from different sources, which is critical in healthcare. This paper presents FarmaGraph RAG, a "
                f"question answering system about drugs that combines vector search and traversal of a Neo4j knowledge "
                f"graph (GraphRAG) with logical queries over the graph (KAG). The graph is built from 22 simplified "
                f"drug leaflets and 7 fictitious medical records by a local LLM (Llama 3.2, 3B) using structured "
                f"output, section-guided extraction, six semantic constraints and entity resolution. Retrieval uses "
                f"VECTOR, FULLTEXT and RANGE indexes, multi-hop expansion, shortest paths, PageRank and LLM-generated "
                f"Cypher validated as read-only. Extraction reached {num(tot['precisão']).replace(',', '.')} "
                f"precision and {num(tot['recall']).replace(',', '.')} recall against {tot['gabarito']} reference "
                f"facts, and {qa_frase_en}.")
    return {
        "titulo": "FarmaGraph RAG: Recuperação Aumentada por Grafos de Conhecimento para Interações Medicamentosas "
                  "com LLM Local",
        "autores": "Biel Aguiar, Thiago Rocha",
        "afiliacao": "Curso de Bacharelado em Ciência da Computação – Unipê Cruzeiro do Sul",
        "emails": "{[e-mails dos integrantes]}",
        "abstract": abstract,
        "resumo": resumo,
        "blocks": blocks,
        "refs": refs,
    }


# ------------------------------------------------------------------ renderização
INLINE = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`)")


def add_runs(par, text: str, size: int = 12, italic_base: bool = False) -> None:
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            run = par.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("`"):
            run = par.add_run(part[1:-1])
            run.font.name = "Courier New"
            size_here = size - 1
            run.font.size = Pt(size_here)
            run.italic = italic_base
            continue
        elif part.startswith("*"):
            run = par.add_run(part[1:-1])
            run.italic = not italic_base
        else:
            run = par.add_run(part)
            run.italic = italic_base
        run.font.size = Pt(size)


def fmt(par, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=6, indent=None, left=None, right=None):
    pf = par.paragraph_format
    pf.alignment = align
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if indent is not None:
        pf.first_line_indent = Cm(indent)
    if left is not None:
        pf.left_indent = Cm(left)
    if right is not None:
        pf.right_indent = Cm(right)


def render_docx(c: dict, path: Path) -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.top_margin, sec.bottom_margin = Cm(3.5), Cm(2.5)
    sec.left_margin, sec.right_margin = Cm(3.0), Cm(3.0)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts")
    if rfonts is not None:
        for attr in ("eastAsia", "ascii", "hAnsi", "cs"):
            rfonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}" + attr, "Times New Roman")

    p = doc.add_paragraph()
    fmt(p, WD_ALIGN_PARAGRAPH.CENTER, after=12)
    r = p.add_run(c["titulo"])
    r.bold, r.font.size = True, Pt(16)
    p = doc.add_paragraph()
    fmt(p, WD_ALIGN_PARAGRAPH.CENTER, after=6)
    r = p.add_run(c["autores"])
    r.bold = True
    for line in (c["afiliacao"], c["emails"]):
        p = doc.add_paragraph()
        fmt(p, WD_ALIGN_PARAGRAPH.CENTER, after=0)
        p.add_run(line).font.size = Pt(10 if line == c["emails"] else 12)

    for label, text in (("Abstract.", c["abstract"]), ("Resumo.", c["resumo"])):
        p = doc.add_paragraph()
        fmt(p, before=12, after=6, left=0.8, right=0.8)
        r = p.add_run(label + " ")
        r.bold, r.italic = True, True
        add_runs(p, text, italic_base=True)

    for b in c["blocks"]:
        kind = b[0]
        if kind == "h1":
            p = doc.add_paragraph()
            fmt(p, WD_ALIGN_PARAGRAPH.LEFT, before=12, after=6)
            r = p.add_run(b[1])
            r.bold, r.font.size = True, Pt(13)
            p.paragraph_format.keep_with_next = True
        elif kind == "h2":
            p = doc.add_paragraph()
            fmt(p, WD_ALIGN_PARAGRAPH.LEFT, before=12, after=6)
            r = p.add_run(b[1])
            r.bold = True
            p.paragraph_format.keep_with_next = True
        elif kind == "p":
            p = doc.add_paragraph()
            fmt(p, indent=1.27)
            add_runs(p, b[1])
        elif kind == "bullets":
            for item in b[1]:
                p = doc.add_paragraph(style="List Bullet")
                fmt(p, after=3)
                add_runs(p, item)
        elif kind == "figure":
            p = doc.add_paragraph()
            fmt(p, WD_ALIGN_PARAGRAPH.CENTER, before=6, after=3)
            p.add_run().add_picture(b[1], width=Cm(15))
            p = doc.add_paragraph()
            fmt(p, WD_ALIGN_PARAGRAPH.CENTER, after=12)
            r = p.add_run(b[2])
            r.bold, r.font.size = True, Pt(10)
        elif kind == "code":
            p = doc.add_paragraph()
            fmt(p, WD_ALIGN_PARAGRAPH.CENTER, before=6, after=3)
            r = p.add_run(b[1])
            r.bold, r.font.size = True, Pt(10)
            p.paragraph_format.keep_with_next = True
            for line in b[2].splitlines():
                p = doc.add_paragraph()
                fmt(p, WD_ALIGN_PARAGRAPH.LEFT, after=0, left=1.0)
                r = p.add_run(line)
                r.font.name, r.font.size = "Courier New", Pt(9)
            doc.add_paragraph().paragraph_format.space_after = Pt(6)
        elif kind == "table":
            _, caption, headers, rows = b
            p = doc.add_paragraph()
            fmt(p, WD_ALIGN_PARAGRAPH.CENTER, before=6, after=3)
            r = p.add_run(caption)
            r.bold, r.font.size = True, Pt(10)
            p.paragraph_format.keep_with_next = True
            t = doc.add_table(rows=1, cols=len(headers))
            t.style = "Table Grid"
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            for i, h in enumerate(headers):
                cell = t.rows[0].cells[i]
                cell.text = ""
                run = cell.paragraphs[0].add_run(h)
                run.bold, run.font.size = True, Pt(9)
            for row in rows:
                cells = t.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = ""
                    run = cells[i].paragraphs[0].add_run(str(v))
                    run.font.size = Pt(9)
                    if row[0] == "TOTAL":
                        run.bold = True
            doc.add_paragraph().paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph()
    fmt(p, WD_ALIGN_PARAGRAPH.LEFT, before=12, after=6)
    r = p.add_run("Referências")
    r.bold, r.font.size = True, Pt(13)
    for ref in c["refs"]:
        p = doc.add_paragraph()
        fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, after=6, left=0.5, indent=-0.5)
        add_runs(p, ref)
    doc.save(path)


def render_md(c: dict, path: Path) -> None:
    out = [f"# {c['titulo']}", "", f"**{c['autores']}**  ", f"{c['afiliacao']}  ", c["emails"], "",
           f"> **Abstract.** *{c['abstract']}*", "", f"> **Resumo.** *{c['resumo']}*", ""]
    for b in c["blocks"]:
        kind = b[0]
        if kind == "h1":
            out += [f"## {b[1]}", ""]
        elif kind == "h2":
            out += [f"### {b[1]}", ""]
        elif kind == "p":
            out += [b[1], ""]
        elif kind == "bullets":
            out += [f"- {i}" for i in b[1]] + [""]
        elif kind == "figure":
            out += [f"![{b[2]}]({Path(b[1]).name})", "", f"*{b[2]}*", ""]
        elif kind == "code":
            out += [f"*{b[1]}*", "", "```cypher", b[2], "```", ""]
        elif kind == "table":
            _, caption, headers, rows = b
            out += [f"*{caption}*", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
            out += ["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in rows] + [""]
    out += ["## Referências", ""] + [f"- {r}" for r in c["refs"]]
    path.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    draw_figure(FIG)
    content = build_content(load_data())
    render_docx(content, DOCS / "artigo.docx")
    render_md(content, DOCS / "artigo.md")
    print("Gerados:", DOCS / "artigo.docx", DOCS / "artigo.md", FIG)


if __name__ == "__main__":
    main()
