import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app.ingest import ingest_document, ingest_folder
from app.rag import DISCLAIMER, GraphRAG
from app.retrieval import SHORTEST_PATH_QUERY, HybridRetriever
from app.services import build_services
from app.viz import legend_html, render

st.set_page_config(page_title="FarmaGraph RAG", page_icon="💊", layout="wide")

EXEMPLOS = [
    "A Maria José pode tomar ibuprofeno para a dor no joelho?",
    "Qual o risco de prescrever claritromicina para o José Carlos?",
    "A Ana Beatriz pode usar Tramal depois da cirurgia?",
    "Por que o potássio do Antônio pode estar alto?",
    "Há interações entre os medicamentos do Roberto?",
    "Que remédio para febre é seguro para a Luiza, com suspeita de dengue?",
    "A Francisca pode começar hidroclorotiazida para a pressão?",
    "Quais medicamentos interagem com a varfarina?",
    "Como a fluoxetina se relaciona com o clonazepam?",
]


@st.cache_resource
def get_services():
    services = build_services()
    retriever = HybridRetriever(services.graph, services.llm, services.resolver,
                                top_k=services.settings.vector_top_k, max_hops=services.settings.max_hops,
                                max_facts=services.settings.max_context_facts)
    return services, GraphRAG(retriever, services.llm, services.resolver)


services, rag = get_services()

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.markdown("## 💊 FarmaGraph RAG")
    st.caption("GraphRAG/KAG sobre interações medicamentosas · Neo4j + Ollama local")
    neo_ok = services.graph.ping()
    models = services.llm.models()
    st.markdown(f"{'🟢' if neo_ok else '🔴'} Neo4j · {'🟢' if models else '🔴'} Ollama")
    if neo_ok:
        try:
            stats = services.graph.stats()
            st.caption(" · ".join(f"{k}: {v}" for k, v in stats["nos"].items()))
        except Exception:
            pass
    st.divider()
    st.markdown("**Opções da busca**")
    hops = st.slider("Saltos na travessia", 1, 3, services.settings.max_hops)
    use_llm_cypher = st.toggle("Gerar Cypher com o LLM (Text-to-Cypher)", value=True)
    compare = st.toggle("Comparar com RAG só vetorial", value=False)
    st.divider()
    st.markdown("**Perguntas de exemplo**")
    for i, exemplo in enumerate(EXEMPLOS):
        if st.button(exemplo, key=f"ex{i}", use_container_width=True):
            st.session_state["pergunta"] = exemplo
    st.divider()
    st.caption(DISCLAIMER)

if not neo_ok:
    st.error("Neo4j indisponível. Rode `docker compose up -d` na pasta do projeto.")
    st.stop()

tab_ask, tab_graph, tab_ingest, tab_eval, tab_schema = st.tabs(
    ["🔎 Perguntar", "🕸️ Explorar o grafo", "📄 Ingestão", "📊 Avaliação", "🧱 Esquema Neo4j"]
)

# --------------------------------------------------------------------- perguntar
with tab_ask:
    st.markdown("### Pergunte sobre medicamentos, interações e pacientes")
    question = st.text_area("Pergunta", key="pergunta", height=80,
                            placeholder="Ex.: A Maria José pode tomar ibuprofeno para a dor no joelho?")
    go = st.button("Pesquisar", type="primary")
    if go and question.strip():
        with st.spinner("Busca vetorial → ligação de entidades → travessia multi-hop → Cypher → LLM..."):
            answer = rag.ask(question.strip(), use_llm_cypher=use_llm_cypher, hops=hops)
            baseline = rag.ask_vector_only(question.strip()) if compare else None
        st.session_state["ultima"] = (answer, baseline)

    if "ultima" in st.session_state:
        answer, baseline = st.session_state["ultima"]
        r = answer.recuperacao
        badge = {"Alta": "🟢", "Média": "🟡", "Baixa": "🔴"}.get(answer.nivel, "⚪")
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown("#### Resposta (GraphRAG)")
            st.markdown(answer.resposta)
            st.caption(DISCLAIMER)
        with cols[1]:
            st.metric("Confiança", f"{badge} {answer.nivel}", f"{answer.confianca:.2f}")
            st.caption(" · ".join(f"{k}: {v}" for k, v in answer.componentes.items()))
            st.caption(f"⏱️ {answer.segundos}s · {len(r.fatos)} fatos · {len(r.alertas)} alertas · {len(r.trechos)} trechos")

        v = answer.verificacao
        if v.nao_fundamentadas or v.citacoes_invalidas:
            st.warning(
                "Verificação de alucinação: "
                + (f"entidades sem suporte no contexto: {', '.join(v.nao_fundamentadas)}. " if v.nao_fundamentadas else "")
                + (f"citações inexistentes: {', '.join(v.citacoes_invalidas)}." if v.citacoes_invalidas else "")
            )
        else:
            st.success(f"Verificação de alucinação: {len(v.entidades_resposta)} entidades citadas, todas presentes no "
                       f"contexto recuperado · {len(v.citacoes_validas)} citações válidas.")

        if baseline:
            with st.expander("Comparação: RAG só vetorial (sem grafo)", expanded=True):
                st.markdown(baseline.resposta)
                bv = baseline.verificacao
                st.caption(f"Confiança {baseline.nivel} ({baseline.confianca:.2f}) · entidades sem suporte: "
                           f"{', '.join(bv.nao_fundamentadas) or 'nenhuma'}")

        if r.alertas:
            st.markdown("#### ⚠️ Alertas do grafo (raciocínio Cypher multi-hop)")
            for a in r.alertas:
                icon = "🔴" if a.get("gravidade") == "alta" else "🟠" if a.get("gravidade") else "🟢"
                st.markdown(f"{icon} **[{a['ref']}]** {a['texto']}")

        st.markdown("#### Subgrafo usado como contexto")
        st.markdown(legend_html(), unsafe_allow_html=True)
        if r.subgrafo["nodes"]:
            components.html(render(r.subgrafo, height=540), height=560, scrolling=False)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Entidades reconhecidas na pergunta**")
            st.dataframe(pd.DataFrame(r.sementes)[["nome", "label", "origem"]] if r.sementes else pd.DataFrame(),
                         hide_index=True, use_container_width=True)
            if r.caminhos:
                st.markdown("**Caminhos mais curtos (shortestPath)**")
                for p in r.caminhos:
                    st.code(p["texto"], language=None)
        with c2:
            st.markdown("**Nós consultados no subgrafo**")
            st.dataframe(pd.DataFrame(
                [{"nó": n["nome"], "rótulo": n["label"], "PageRank": round(n.get("pagerank") or 0, 3)} for n in r.subgrafo["nodes"]]
            ).sort_values("PageRank", ascending=False), hide_index=True, use_container_width=True, height=240)

        with st.expander("Fatos do grafo enviados ao LLM (proveniência)"):
            st.dataframe(pd.DataFrame(
                [{"ref": f["ref"], "fato": f["texto"], "score": f["score"], "trechos de origem": len(f["props"].get("fontes", []))}
                 for f in r.fatos]), hide_index=True, use_container_width=True)
        with st.expander("Trechos recuperados pela busca vetorial"):
            for t in r.trechos:
                st.markdown(f"**[{t['ref']}] {t['arquivo']}** · trecho {t['indice']} · similaridade {t['score']:.3f}")
                st.caption(t["texto"])
        with st.expander("Consultas Cypher executadas", expanded=False):
            for log in r.cypher_log:
                st.markdown(f"**{log['etapa']}** — {log['linhas']} linha(s)")
                st.code(log["cypher"], language="cypher")
            if r.cypher_llm is not None:
                st.markdown("**6. Text-to-Cypher (gerado pelo LLM, validado como somente leitura)**")
                st.code(r.cypher_llm.get("cypher") or "(vazio)", language="cypher")
                if r.cypher_llm.get("erro"):
                    st.caption(f"⚠️ {r.cypher_llm['erro']}")
                elif r.cypher_llm.get("linhas"):
                    st.dataframe(pd.DataFrame(r.cypher_llm["linhas"]), hide_index=True, use_container_width=True)
        with st.expander("Contexto completo enviado ao LLM"):
            st.text(answer.contexto)

# ---------------------------------------------------------------- explorar grafo
with tab_graph:
    st.markdown("### Grafo de conhecimento completo")
    color_by = st.radio("Colorir por", ["label", "comunidade"], horizontal=True,
                        format_func=lambda x: "tipo de nó" if x == "label" else "comunidade (Louvain)")
    full = services.graph.full_graph()
    st.markdown(legend_html(), unsafe_allow_html=True)
    components.html(render(full, height=620, color_by=color_by, edge_labels=False), height=640)
    a, b = st.columns(2)
    with a:
        st.markdown("**PageRank — entidades mais centrais**")
        st.dataframe(pd.DataFrame(services.graph.top_pagerank(12)), hide_index=True, use_container_width=True)
    with b:
        st.markdown("**Comunidades de interação (Louvain)**")
        comms = services.graph.communities()
        st.dataframe(pd.DataFrame([{"comunidade": c["comunidade"], "medicamentos": ", ".join(sorted(c["medicamentos"]))}
                                   for c in comms]), hide_index=True, use_container_width=True)
    st.markdown("**Caminho mais curto entre duas entidades**")
    names = services.graph.run("MATCH (n) WHERE n:Medicamento OR n:Paciente OR n:Condicao OR n:EfeitoAdverso "
                               "RETURN elementId(n) AS id, n.nome + ' (' + labels(n)[0] + ')' AS rotulo ORDER BY rotulo")
    options = {n["rotulo"]: n["id"] for n in names}
    if options:
        s1, s2 = st.columns(2)
        origem = s1.selectbox("De", list(options), index=0)
        destino = s2.selectbox("Para", list(options), index=min(1, len(options) - 1))
        if st.button("Calcular shortestPath"):
            rows = services.graph.run(SHORTEST_PATH_QUERY, {"a": options[origem], "b": options[destino]})
            if rows:
                from app.retrieval import _path_text

                st.code(_path_text(rows[0]), language=None)
                sub = {"nodes": [{**n, "semente": n["id"] in (options[origem], options[destino])} for n in rows[0]["nos"]],
                       "edges": [{"source": e["a"], "target": e["b"], "tipo": e["tipo"], "props": e["props"], "caminho": True}
                                 for e in rows[0]["rels"]]}
                components.html(render(sub, height=320), height=340)
            else:
                st.info("Não há caminho entre essas entidades.")
            st.code(SHORTEST_PATH_QUERY.strip(), language="cypher")

# --------------------------------------------------------------------- ingestão
with tab_ingest:
    st.markdown("### Ingestão de documentos (Text-to-Graph)")
    st.caption("Trechos → NER/RE com LLM local (saída em JSON Schema) → restrições semânticas → resolução de "
               "entidades → MERGE no Neo4j → embeddings no índice vetorial → PageRank/Louvain.")
    files = st.file_uploader("Bulas, prontuários ou outros textos (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"],
                             accept_multiple_files=True)
    col1, col2 = st.columns(2)
    run_upload = col1.button("Ingerir arquivos enviados", type="primary", disabled=not files)
    run_corpus = col2.button("Reingerir o corpus de exemplo (corpus/)")
    if run_upload or run_corpus:
        bar = st.progress(0.0, text="Preparando...")
        progress = lambda m, f: bar.progress(min(max(f, 0.0), 1.0), text=m)  # noqa: E731
        services.graph.setup_schema(services.settings.embedding_dimension)
        if run_upload:
            reports = []
            for n, f in enumerate(files):
                reports.append(ingest_document(services, f.name, f.getvalue(), force=True,
                                               progress=lambda m, x, n=n: progress(m, (n + x) / len(files))))
        else:
            reports = ingest_folder(services, services.settings.corpus_dir, force=True, progress=progress)
        bar.progress(1.0, text="Calculando PageRank e Louvain...")
        algo = services.graph.compute_algorithms()
        services.refresh_resolver()
        bar.empty()
        st.success(f"{len(reports)} documento(s) processado(s). Algoritmos: {algo}")
        st.session_state["relatorios"] = reports
    reports = st.session_state.get("relatorios")
    if reports is None and (services.settings.data_dir / "ultima_ingestao.json").exists():
        reports = json.loads((services.settings.data_dir / "ultima_ingestao.json").read_text(encoding="utf-8"))["relatorios"]
        st.caption("Relatório da última ingestão completa (linha de comando):")
    if reports:
        df = pd.DataFrame([{k: r.get(k) for k in ("arquivo", "status", "trechos", "entidades", "relacoes_brutas_llm",
                                                   "relacoes_aceitas", "relacoes_rejeitadas", "direcoes_corrigidas", "segundos")}
                           for r in reports])
        st.dataframe(df, hide_index=True, use_container_width=True)
        motivos: dict[str, int] = {}
        resol = []
        for r in reports:
            for k, v in (r.get("motivos_rejeicao") or {}).items():
                motivos[k] = motivos.get(k, 0) + v
            resol.extend(r.get("resolucoes") or [])
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Relações barradas pelas restrições semânticas**")
            st.dataframe(pd.DataFrame([{"motivo": k, "quantidade": v} for k, v in sorted(motivos.items(), key=lambda x: -x[1])]),
                         hide_index=True, use_container_width=True)
        with m2:
            st.markdown("**Resolução de entidades (nome no texto → nó canônico)**")
            if resol:
                st.dataframe(pd.DataFrame(resol).drop_duplicates(subset=["superficie", "canonico"]),
                             hide_index=True, use_container_width=True, height=260)

# -------------------------------------------------------------------- avaliação
with tab_eval:
    st.markdown("### Avaliação")
    path = services.settings.data_dir / "avaliacao.json"
    if path.exists():
        ev = json.loads(path.read_text(encoding="utf-8"))
        st.markdown("#### 1. Qualidade da extração (grafo extraído × gabarito)")
        st.dataframe(pd.DataFrame(ev["extracao"]["por_relacao"]), hide_index=True, use_container_width=True)
        st.caption(ev["extracao"].get("observacao", ""))
        st.markdown("#### 2. Perguntas: LLM sozinho × RAG vetorial × GraphRAG")
        qa = ev.get("qa")
        if qa:
            st.dataframe(pd.DataFrame(qa["resumo"]), hide_index=True, use_container_width=True)
            with st.expander("Resultado por pergunta"):
                st.dataframe(pd.DataFrame(qa["detalhes"]), hide_index=True, use_container_width=True)
            st.caption(qa.get("metodo", ""))
        else:
            st.info("Avaliação de perguntas ainda não rodada (`python -m app.evaluate` sem `--sem-qa`).")
    else:
        st.info("Rode `python -m app.evaluate` para gerar a avaliação.")

# ----------------------------------------------------------------------- esquema
with tab_schema:
    st.markdown("### Modelo LPG no Neo4j")
    from app.schema import RELATION_SCHEMA

    st.dataframe(pd.DataFrame([{"relação": k, "origem": v[0], "destino": v[1], "significado": v[2]}
                               for k, v in RELATION_SCHEMA.items()]), hide_index=True, use_container_width=True)
    info = services.graph.schema_info()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Constraints de unicidade**")
        st.dataframe(pd.DataFrame(info["constraints"]), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Índices (RANGE, FULLTEXT, VECTOR)**")
        st.dataframe(pd.DataFrame(info["indexes"]), hide_index=True, use_container_width=True)
    st.markdown("Neo4j Browser: <http://localhost:7474> (usuário `neo4j`)")
