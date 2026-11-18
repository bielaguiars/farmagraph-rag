"""Gera o PDF da apresentação no estilo dos trabalhos anteriores (documento A4, consulta + print).

Uso (precisa do Chrome instalado e dos prints em docs/prints/):
    uv run --with playwright python docs/gerar_pdf_apresentacao.py
Saída: docs/FarmaGraph_RAG_apresentacao.pdf (e o .html usado para gerar).
"""

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PRINTS = DOCS / "prints"
TITLE = "FarmaGraph RAG — Projeto Final"

KEYWORDS = (r"OPTIONAL MATCH|MATCH|WHERE|RETURN|WITH|ORDER BY|LIMIT|SKIP|CALL|YIELD|CREATE|CONSTRAINT|FULLTEXT|"
            r"VECTOR|INDEX|IF NOT EXISTS|FOR|REQUIRE|IS UNIQUE|ON EACH|ON|OPTIONS|AS|AND|OR|IN|NOT|DISTINCT|"
            r"UNWIND|MERGE|SET|DESC|EXISTS|CASE|WHEN|THEN|ELSE|END")
TOKEN = re.compile(
    r"(?P<comment>//[^\n]*)|(?P<string>'[^']*')|(?P<label>(?<=[(\[:|])[A-Z][A-Za-z_]+(?=[\s){\]:|*]))|"
    rf"(?P<kw>\b(?:{KEYWORDS})\b)|(?P<fn>\b[a-zA-Z_.]+(?=\())"
)
COLORS = {"comment": "#9B9A97", "string": "#4D7C0F", "label": "#2F6F9F", "kw": "#A3195B", "fn": "#B35C00"}


def highlight(code: str) -> str:
    out, pos = [], 0
    for m in TOKEN.finditer(code):
        out.append(html.escape(code[pos:m.start()]))
        kind = m.lastgroup
        out.append(f'<span style="color:{COLORS[kind]}">{html.escape(m.group())}</span>')
        pos = m.end()
    out.append(html.escape(code[pos:]))
    return "".join(out)


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


# ------------------------------------------------------------------ conteúdo
# blocos: ("h2", t) seção · ("h3", t) item com ▼ · ("p", t) · ("ul", [..]) · ("code", t) · ("img", arquivo, legenda)
C = []
C += [
    ("p", "Tendências em Ciência da Computação · Prof. Ricardo Roberto de Lima · 18/11/2026"),
    ("p", "**Biel Aguiar e Thiago Rocha** · Unipê Cruzeiro do Sul"),
    ("p", "Repositório: github.com/bielaguiars/farmagraph-rag · Tudo roda localmente: Neo4j 5 (Docker) com APOC e "
          "Graph Data Science, Ollama (llama3.2 + nomic-embed-text) e Streamlit."),
    ("h2", "1 - Problema e dados"),
    ("p", "Pergunta-guia: **“A Maria José pode tomar ibuprofeno para a dor no joelho?”** A resposta está espalhada em "
          "três lugares: o prontuário diz que ela usa **Marevan**; Marevan é a marca da **varfarina**; a bula da "
          "varfarina diz que a associação com ibuprofeno tem **risco alto de sangramento**. O LLM sozinho não conhece "
          "o prontuário, e o RAG vetorial acha a bula do ibuprofeno, mas não liga a paciente à varfarina."),
    ("ul", ["22 bulas simplificadas com as seções da RDC nº 47/2009 da ANVISA (indicações, contraindicações, "
            "interações, reações adversas).",
            "7 prontuários fictícios em texto livre: citam só o nome comercial e nunca mencionam interações.",
            "Gabarito com 168 fatos, usado só na avaliação (o sistema nunca lê esse arquivo).",
            "Base didática e fictícia. Não é orientação médica."]),
    ("h2", "2 - Arquitetura"),
    ("img", str(DOCS / "figura_arquitetura.png"), "Construção do grafo (offline) e consulta (online)."),
    ("h2", "3 - Pilar 1: Text-to-Graph (NER/RE com LLM)"),
    ("ul", ["O texto é dividido por seção da bula e cada seção só pode gerar a sua relação (Indicações → TRATA, "
            "Contraindicações → CONTRAINDICADO_EM, Interações → INTERAGE_COM, Reações adversas → PODE_CAUSAR).",
            "O llama3.2 (3B, local) responde em **JSON Schema**, com as relações em enum e no máximo 10 itens.",
            "Seis travas: enums, relação compatível com a seção, relação permitida para o tipo de documento, "
            "domínio/contradomínio com correção de direção, **ancoragem textual** e envolvimento do remédio da bula.",
            "Resolução de entidades: normalização, dicionário de nomes comerciais (Marevan → varfarina), tokens "
            "(“Maria José” → Maria José Ferreira), similaridade (warfarina → varfarina) e separação de listas "
            "(“dor e febre” → dor, febre).",
            "Na última ingestão: 182 relações propostas pelo LLM, 162 aceitas e 23 barradas; 17 delas por entidade "
            "não ancorada no texto."]),
    ("h3", "Relatório da ingestão: relações barradas e resolução de entidades"),
    ("img", "ui_ingestao.png", ""),
    ("h2", "4 - Pilar 2: modelo LPG no Neo4j"),
    ("h3", "Constraints de unicidade e índices RANGE, FULLTEXT e VECTOR"),
    ("code", "// Constraints de unicidade (uma por rótulo)\n"
             "CREATE CONSTRAINT medicamento_id IF NOT EXISTS FOR (n:Medicamento) REQUIRE n.id IS UNIQUE;\n"
             "CREATE CONSTRAINT paciente_id IF NOT EXISTS FOR (n:Paciente) REQUIRE n.id IS UNIQUE;\n"
             "// ... ClasseTerapeutica, Condicao, EfeitoAdverso, Documento, Chunk\n\n"
             "// RANGE\n"
             "CREATE INDEX medicamento_nome IF NOT EXISTS FOR (n:Medicamento) ON (n.nome);\n"
             "CREATE INDEX interacao_gravidade IF NOT EXISTS FOR ()-[r:INTERAGE_COM]-() ON (r.gravidade);\n"
             "// FULLTEXT (nomes e marcas)\n"
             "CREATE FULLTEXT INDEX entidade_nome IF NOT EXISTS\n"
             "FOR (n:Medicamento|ClasseTerapeutica|Condicao|EfeitoAdverso|Paciente) ON EACH [n.nome, n.aliases_texto];\n"
             "// VECTOR (768 dimensões, cosseno)\n"
             "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (n:Chunk) ON (n.embedding)\n"
             "OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}};"),
    ("img", "neo4j_indices.png", ""),
    ("h3", "Esquema do grafo (db.schema.visualization)"),
    ("code", "CALL db.schema.visualization()"),
    ("img", "neo4j_esquema.png", ""),
    ("h3", "Proveniência: toda aresta guarda os trechos de onde saiu"),
    ("code", "MATCH (a:Medicamento {nome: 'varfarina'})-[r:INTERAGE_COM]-(b:Medicamento {nome: 'ibuprofeno'})\n"
             "MATCH (c:Chunk) WHERE c.id IN r.fontes\n"
             "RETURN a.nome AS de, b.nome AS para, r.gravidade AS gravidade, r.efeito AS efeito,\n"
             "       left(c.texto, 90) AS trecho_de_origem"),
    ("img", "neo4j_proveniencia.png", ""),
    ("h2", "5 - Pilar 3: recuperação híbrida e algoritmos de grafo"),
    ("h3", "Busca vetorial nos trechos (índice VECTOR)"),
    ("code", "MATCH (:Documento {arquivo: 'bula_varfarina.txt'})-[:TEM_TRECHO]->(c:Chunk)\n"
             "WITH c ORDER BY c.indice SKIP 4 LIMIT 1\n"
             "CALL db.index.vector.queryNodes('chunk_embedding', 4, c.embedding) YIELD node, score\n"
             "MATCH (d:Documento)-[:TEM_TRECHO]->(node)\n"
             "RETURN d.arquivo AS documento, left(node.texto, 70) AS trecho, round(score, 3) AS similaridade"),
    ("img", "neo4j_vetor.png", ""),
    ("h3", "Ligação de entidades com tolerância a erro (índice FULLTEXT)"),
    ("code", "CALL db.index.fulltext.queryNodes('entidade_nome', 'warfarina~1 OR marevan') YIELD node, score\n"
             "RETURN labels(node)[0] AS rotulo, node.nome AS entidade, node.nomes_comerciais AS marcas,\n"
             "       round(score, 3) AS score LIMIT 5"),
    ("img", "neo4j_fulltext.png", ""),
    ("h3", "KAG: interações entre os remédios da paciente e o remédio perguntado (multi-hop)"),
    ("code", "MATCH (p:Paciente {nome: 'Maria José Ferreira'})-[u:USA]->(m:Medicamento)\n"
             "      -[i:INTERAGE_COM]-(c:Medicamento {nome: 'ibuprofeno'})\n"
             "RETURN p, u, m, i, c"),
    ("img", "neo4j_kag_maria.png", ""),
    ("h3", "KAG: remédios para febre × contraindicações na dengue"),
    ("code", "MATCH (f:Condicao {nome: 'febre'})<-[t:TRATA]-(s:Medicamento)\n"
             "MATCH (d:Condicao {nome: 'dengue'})<-[x:CONTRAINDICADO_EM]-(m:Medicamento)\n"
             "RETURN f, t, s, d, x, m"),
    ("img", "neo4j_kag_dengue.png", ""),
    ("h3", "Menor caminho entre fluoxetina e clonazepam (shortestPath)"),
    ("code", "MATCH (a:Medicamento {nome: 'fluoxetina'}), (b:Medicamento {nome: 'clonazepam'})\n"
             "MATCH p = shortestPath((a)-[:INTERAGE_COM|TRATA|CONTRAINDICADO_EM|PODE_CAUSAR|\n"
             "                           PERTENCE_A_CLASSE|USA|TEM_CONDICAO*..6]-(b))\n"
             "RETURN p"),
    ("img", "neo4j_caminho.png", ""),
    ("h3", "PageRank (GDS): centralidade usada para ranquear o contexto"),
    ("code", "CALL gds.graph.project('farma_dominio',\n"
             "  ['Medicamento','ClasseTerapeutica','Condicao','EfeitoAdverso','Paciente'],\n"
             "  {INTERAGE_COM: {orientation: 'UNDIRECTED'}, TRATA: {orientation: 'UNDIRECTED'}, ...});\n"
             "CALL gds.pageRank.write('farma_dominio', {writeProperty: 'pagerank'});\n\n"
             "MATCH (n) WHERE n.pagerank IS NOT NULL\n"
             "RETURN labels(n)[0] AS rotulo, n.nome AS entidade, round(n.pagerank, 3) AS pagerank\n"
             "ORDER BY pagerank DESC LIMIT 10"),
    ("img", "neo4j_pagerank.png", ""),
    ("h3", "Louvain (GDS): comunidades de risco nas interações"),
    ("code", "CALL gds.graph.project('farma_interacoes', 'Medicamento', {INTERAGE_COM: {orientation: 'UNDIRECTED'}});\n"
             "CALL gds.louvain.write('farma_interacoes', {writeProperty: 'comunidade'});\n\n"
             "MATCH (m:Medicamento)\n"
             "RETURN m.comunidade AS comunidade, collect(m.nome) AS medicamentos, count(*) AS total\n"
             "ORDER BY total DESC"),
    ("img", "neo4j_louvain.png", ""),
    ("h3", "Text-to-Cypher: consulta gerada pelo LLM e validada como somente leitura"),
    ("p", "A consulta gerada passa por uma lista de bloqueio (CREATE, MERGE, DELETE, SET, CALL...) e por um "
          "`EXPLAIN` no Neo4j: só executa se o tipo da consulta for `r` (leitura)."),
    ("img", "ui_varfarina_text2cypher.png", ""),
    ("h2", "6 - Pilar 4: interface e explicabilidade"),
    ("h3", "“A Maria José pode tomar ibuprofeno para a dor no joelho?” — resposta, confiança e alertas"),
    ("img", "ui_maria_resposta.png", ""),
    ("h3", "Subgrafo usado como contexto (estrela = entidade da pergunta, vermelho = interação alta)"),
    ("img", "ui_maria_subgrafo.png", ""),
    ("h3", "Consultas Cypher executadas para responder"),
    ("img", "ui_maria_cypher.png", ""),
    ("h3", "“Como a fluoxetina se relaciona com o clonazepam?” — caminho de 2 saltos"),
    ("img", "ui_fluox_resposta.png", ""),
    ("img", "ui_fluox_subgrafo.png", ""),
    ("h3", "GraphRAG × RAG só vetorial (“A Ana Beatriz pode usar Tramal depois da cirurgia?”)"),
    ("img", "ui_ana_comparacao.png", ""),
    ("h3", "Grafo completo colorido por comunidade (Louvain) e ranking de PageRank"),
    ("img", "ui_explorar.png", ""),
    ("h2", "7 - Redução de alucinação"),
    ("ul", ["**Na extração:** só entra relação cujas duas pontas estão escritas no trecho. No teste, o LLM copiou "
            "“Aldactone” do exemplo do prompt numa bula que não falava dele, e a ancoragem barrou.",
            "**Na geração (KAG):** quando o grafo conclui algo (alertas K), o LLM recebe só esses alertas e apenas "
            "os redige. Se não citar, a resposta começa com “Conclusão do grafo”.",
            "**Depois da resposta:** o verificador confere as citações [K][G][T] e procura medicamentos e doenças "
            "que não estão no contexto. Confiança = 25% similaridade vetorial + 25% suporte do grafo + 35% "
            "entidades fundamentadas + 15% citações válidas."]),
    ("h2", "8 - Avaliação"),
    ("ul", ["**Extração × gabarito (168 fatos):** precisão 0,99, recall 0,85, F1 0,91. Interações, "
            "contraindicações e uso de medicamentos: 100%. Gravidade das interações certa em 22 de 22.",
            "**Perguntas multi-hop (9), mesmo LLM nos três modos:** GraphRAG cobriu **94%** dos fatos esperados "
            "(8 de 9 perguntas 100% certas, 0 entidades sem suporte); RAG só vetorial 17%; LLM sem contexto 15%.",
            "Custo: 23,8 s por pergunta no GraphRAG, contra 15,6 s (vetorial) e 6,8 s (LLM), numa RTX 3050 de 4 GB."]),
    ("img", "ui_avaliacao.png", ""),
    ("h2", "9 - Como rodar"),
    ("code", "docker compose up -d                      # Neo4j 5 + APOC + GDS\n"
             "ollama pull llama3.2; ollama pull nomic-embed-text\n"
             ".\\.venv\\Scripts\\python -m app.ingest --reset   # constrói o grafo\n"
             ".\\.venv\\Scripts\\python -m app.evaluate         # avaliação\n"
             ".\\.venv\\Scripts\\python -m streamlit run streamlit_app.py\n"
             "# ou simplesmente: .\\iniciar.ps1"),
    ("ul", ["Limitações: modelo de 3B erra extrações e redação; corpus pequeno e didático; cobertura por "
            "palavras-chave não mede qualidade clínica.",
            "Próximos passos: bulas oficiais e gabarito revisado por farmacêuticos, modelo maior, RAGAS e resumos "
            "de comunidade (busca global do GraphRAG)."]),
]

CSS = """
@page { size: A4; margin: 22mm 20mm 22mm 20mm; }
body { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; color: #37352F; font-size: 14.5px; line-height: 1.55; }
h1 { font-size: 38px; font-weight: 700; line-height: 1.15; margin: 0 0 18px; letter-spacing: -0.5px; }
h2 { font-size: 26px; font-weight: 700; margin: 34px 0 10px; break-after: avoid; }
h3 { font-size: 16px; font-weight: 600; margin: 22px 0 8px; break-after: avoid; }
h3::before { content: "▼ "; font-size: 12px; color: #37352F; }
p { margin: 6px 0; }
ul { margin: 6px 0 6px 0; padding-left: 26px; }
li { margin: 3px 0; }
code { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace; font-size: 85%;
       background: rgba(135,131,120,0.15); color: #EB5757; padding: 1px 4px; border-radius: 3px; }
pre { background: #F7F6F3; border-radius: 4px; padding: 18px 22px; margin: 8px 0 12px;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace; font-size: 11.5px; line-height: 1.6;
      white-space: pre-wrap; word-break: break-word; break-inside: avoid; color: #37352F; }
figure { margin: 8px 0 18px; break-inside: avoid; }
figure img { width: 100%; border: 1px solid #E9E9E7; border-radius: 4px; display: block; }
figcaption { font-size: 12px; color: #787774; margin-top: 4px; }
.meta { color: #787774; }
.item { break-inside: avoid; page-break-inside: avoid; padding-top: 2px; }
.item figure img { max-height: 205mm; object-fit: contain; }
"""


def build_html() -> str:
    parts = [f"<h1>{html.escape('FarmaGraph RAG: GraphRAG + KAG para interações medicamentosas')}</h1>"]
    in_item = False  # cada item ▼ (título + código + print) vira um bloco que não se divide entre páginas
    for i, block in enumerate(C):
        kind = block[0]
        if kind in {"h2", "h3"} and in_item:
            parts.append("</div>")
            in_item = False
        if kind == "h2":
            parts.append(f"<h2>{inline(block[1])}</h2>")
        elif kind == "h3":
            parts.append(f"<div class='item'><h3>{inline(block[1])}</h3>")
            in_item = True
        elif kind == "p":
            cls = ' class="meta"' if i < 3 else ""
            parts.append(f"<p{cls}>{inline(block[1])}</p>")
        elif kind == "ul":
            parts.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in block[1]) + "</ul>")
        elif kind == "code":
            parts.append(f"<pre>{highlight(block[1])}</pre>")
        elif kind == "img":
            path = Path(block[1]) if Path(block[1]).is_absolute() else PRINTS / block[1]
            if not path.exists():
                print("aviso: print ausente", path.name)
                continue
            cap = f"<figcaption>{inline(block[2])}</figcaption>" if block[2] else ""
            parts.append(f'<figure><img src="{path.as_uri()}">{cap}</figure>')
    if in_item:
        parts.append("</div>")
    return ("<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>FarmaGraph RAG</title>"
            "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap'>"
            f"<style>{CSS}</style></head><body>" + "\n".join(parts) + "</body></html>")


def main() -> None:
    from playwright.sync_api import sync_playwright

    out_html = DOCS / "FarmaGraph_RAG_apresentacao.html"
    out_pdf = DOCS / "FarmaGraph_RAG_apresentacao.pdf"
    out_html.write_text(build_html(), encoding="utf-8")
    footer = ("<div style='font-family:Arial;font-size:8px;color:#9B9A97;width:100%;padding:0 20mm;"
              f"display:flex;justify-content:space-between'><span>{html.escape(TITLE)}</span>"
              "<span class='pageNumber'></span></div>")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page()
        page.goto(out_html.as_uri(), wait_until="networkidle")
        page.pdf(path=str(out_pdf), format="A4", print_background=True, display_header_footer=True,
                 header_template="<div></div>", footer_template=footer,
                 margin={"top": "22mm", "bottom": "22mm", "left": "20mm", "right": "20mm"})
        browser.close()
    print("Gerado:", out_pdf)


if __name__ == "__main__":
    main()
