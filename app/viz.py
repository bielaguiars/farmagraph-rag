"""Renderização interativa de subgrafos com pyvis (vis-network embutido, funciona offline)."""

import html

from pyvis.network import Network

LABEL_COLORS = {
    "Medicamento": "#2f7ed8",
    "Paciente": "#8e44ad",
    "Condicao": "#27ae60",
    "EfeitoAdverso": "#e67e22",
    "ClasseTerapeutica": "#7f8c8d",
}
COMMUNITY_COLORS = ["#2f7ed8", "#e74c3c", "#27ae60", "#8e44ad", "#f39c12", "#16a085", "#d35400", "#2c3e50"]


def edge_color(edge: dict) -> str:
    if edge["tipo"] == "INTERAGE_COM":
        return "#e74c3c" if (edge.get("props") or {}).get("gravidade") == "alta" else "#f39c12"
    if edge["tipo"] == "CONTRAINDICADO_EM":
        return "#c0392b"
    if edge.get("caminho"):
        return "#9b59b6"
    return "#95a5a6"


def render(subgraph: dict, height: int = 560, color_by: str = "label", edge_labels: bool = True) -> str:
    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#ffffff",
                  font_color="#222222", cdn_resources="in_line")
    if len(subgraph["nodes"]) > 40:  # grafo completo: física mais compacta para caber na tela
        net.barnes_hut(gravity=-2200, central_gravity=0.9, spring_length=80, spring_strength=0.06, damping=0.35)
    else:
        net.barnes_hut(gravity=-6000, central_gravity=0.25, spring_length=140, spring_strength=0.03)
    for node in subgraph["nodes"]:
        if color_by == "comunidade" and node.get("comunidade") is not None:
            color = COMMUNITY_COLORS[int(node["comunidade"]) % len(COMMUNITY_COLORS)]
        else:
            color = LABEL_COLORS.get(node["label"], "#bdc3c7")
        pagerank = node.get("pagerank") or 0
        size = 14 + min(pagerank * 18, 26)
        tooltip = f"{node['label']}: {node['nome']}"
        if node.get("pagerank") is not None:
            tooltip += f"\nPageRank: {node['pagerank']:.3f}"
        if node.get("comunidade") is not None:
            tooltip += f"\nComunidade: {node['comunidade']}"
        net.add_node(
            node["id"], label=node["nome"], title=tooltip, color=color,
            size=size + (10 if node.get("semente") else 0),
            borderWidth=4 if node.get("semente") else 1,
            shape="star" if node.get("semente") else "dot",
        )
    for edge in subgraph["edges"]:
        props = edge.get("props") or {}
        tooltip = edge["tipo"]
        for key in ("gravidade", "efeito"):
            if props.get(key):
                tooltip += f"\n{key}: {props[key]}"
        if props.get("fontes"):
            tooltip += f"\nfontes: {len(props['fontes'])} trecho(s)"
        if edge.get("ref"):
            tooltip += f"\n[{edge['ref']}]"
        extra = {"label": edge["tipo"]} if edge_labels else {}
        net.add_edge(edge["source"], edge["target"], **extra, title=tooltip,
                     color=edge_color(edge), width=3 if edge.get("caminho") or props.get("gravidade") == "alta" else 1.5,
                     arrows="to" if edge["tipo"] != "INTERAGE_COM" else "")
    net.set_edge_smooth("dynamic")
    html_doc = net.generate_html(notebook=False)
    # enquadra o grafo inteiro depois que a física estabiliza (senão a câmera fica no zoom inicial)
    # (a aba do Streamlit pode estar oculta ao carregar: só conta o tempo enquanto o grafo está visível)
    fit = ("<script>(function(){var last='',seen=0,ticks=0;var t=setInterval(function(){ticks++;"
           "if(typeof network==='undefined'){return;}var c=document.getElementById('mynetwork');"
           "if(!c||c.clientWidth===0){return;}var key=c.clientWidth+'x'+c.clientHeight;"
           "if(key!==last||seen<12){network.fit();}last=key;seen++;if(ticks>600){clearInterval(t);}},700);})();</script>")
    return html_doc.replace("</body>", fit + "</body>")


def legend_html() -> str:
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{c};display:inline-block;margin-right:5px"></span>'
        f"{html.escape(label)}</span>"
        for label, c in LABEL_COLORS.items()
    )
    items += ('<span style="margin-left:6px">★ = entidade da pergunta · aresta vermelha = interação alta / '
              'contraindicação · laranja = interação moderada · roxa = caminho mais curto</span>')
    return f'<div style="font-size:0.85rem;opacity:0.85;line-height:1.8">{items}</div>'
