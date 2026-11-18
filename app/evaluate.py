"""Avaliação: (1) grafo extraído × gabarito; (2) perguntas em 3 modos (LLM, RAG vetorial, GraphRAG).

Uso: python -m app.evaluate [--sem-qa]
"""

import argparse
import difflib
import json
import time

from app.rag import GraphRAG
from app.resolution import normalize, singularize
from app.retrieval import HybridRetriever
from app.services import build_services

QUESTIONS = [
    {"pergunta": "A Maria José pode tomar ibuprofeno para a dor no joelho?",
     "esperado": [["sangramento"], ["varfarina", "marevan"], ["losartana", "cozaar", "renal"]]},
    {"pergunta": "Qual o risco de prescrever claritromicina para o José Carlos?",
     "esperado": [["rabdomiolise"], ["sinvastatina", "zocor"]]},
    {"pergunta": "A Ana Beatriz pode usar Tramal depois da cirurgia?",
     "esperado": [["serotoninergica"], ["fluoxetina", "prozac"]]},
    {"pergunta": "Por que o potássio do Antônio pode estar alto?",
     "esperado": [["hipercalemia", "potassio"], ["enalapril", "renitec"], ["espironolactona", "aldactone"]]},
    {"pergunta": "Há interações entre os medicamentos do Roberto?",
     "esperado": [["clopidogrel", "plavix"], ["omeprazol", "losec"], ["levotiroxina", "puran"]]},
    {"pergunta": "Que remédio para febre é seguro para a Luiza, com suspeita de dengue?",
     "esperado": [["paracetamol", "dipirona", "tylenol", "novalgina"], ["acetilsalicilico", "aas", "ibuprofeno"]]},
    {"pergunta": "A Francisca pode começar hidroclorotiazida para a pressão?",
     "esperado": [["litio", "carbolitium"], ["intoxicacao"]]},
    {"pergunta": "Quais medicamentos interagem com a varfarina?",
     "esperado": [["acetilsalicilico", "aas"], ["ibuprofeno"], ["claritromicina"], ["ciprofloxacino"],
                  ["clopidogrel"], ["paracetamol"]]},
    {"pergunta": "Como a fluoxetina se relaciona com o clonazepam?",
     "esperado": [["tramadol"]]},
]


def _key(label: str, name: str) -> str:
    return normalize(singularize(name)) if label == "ClasseTerapeutica" else normalize(name)


def _match(a: str, b: str) -> bool:
    return a == b or (len(a) > 5 and (a in b or b in a)) or difflib.SequenceMatcher(None, a, b).ratio() >= 0.85


def gold_triples(gabarito: dict) -> set[tuple[str, str, str]]:
    gold = set()
    for m in gabarito["medicamentos"]:
        med = normalize(m["nome"])
        gold.add(("PERTENCE_A_CLASSE", med, _key("ClasseTerapeutica", m["classe"])))
        gold |= {("TRATA", med, normalize(c)) for c in m["trata"]}
        gold |= {("CONTRAINDICADO_EM", med, normalize(c)) for c in m["contraindicado_em"]}
        gold |= {("PODE_CAUSAR", med, normalize(e)) for e in m["pode_causar"]}
    for i in gabarito["interacoes"]:
        a, b = sorted([normalize(i["a"]), normalize(i["b"])])
        gold.add(("INTERAGE_COM", a, b))
    for p in gabarito["pacientes"]:
        pac = normalize(p["nome"])
        gold |= {("USA", pac, normalize(m)) for m in p["usa"]}
        gold |= {("TEM_CONDICAO", pac, normalize(c)) for c in p["condicoes"]}
    return gold


def evaluate_extraction(services) -> dict:
    gabarito = json.loads((services.settings.data_dir / "gabarito.json").read_text(encoding="utf-8"))
    gold = gold_triples(gabarito)
    rows = services.graph.run(
        "MATCH (a)-[r]->(b) WHERE type(r) IN $tipos "
        "RETURN type(r) AS tipo, labels(b)[0] AS b_label, a.nome AS a, b.nome AS b, r.gravidade AS gravidade",
        {"tipos": ["PERTENCE_A_CLASSE", "TRATA", "CONTRAINDICADO_EM", "PODE_CAUSAR", "INTERAGE_COM", "USA", "TEM_CONDICAO"]},
    )
    predicted = set()
    for r in rows:
        a, b = normalize(r["a"]), _key(r["b_label"], r["b"])
        if r["tipo"] == "INTERAGE_COM":
            a, b = sorted([a, b])
        predicted.add((r["tipo"], a, b))

    per_type = []
    total_tp = total_pred = total_gold = 0
    errors = []
    for tipo in ["PERTENCE_A_CLASSE", "TRATA", "CONTRAINDICADO_EM", "PODE_CAUSAR", "INTERAGE_COM", "USA", "TEM_CONDICAO"]:
        g = [t for t in gold if t[0] == tipo]
        p = [t for t in predicted if t[0] == tipo]
        matched_gold = set()
        tp = 0
        for pt in p:
            hit = next((gt for gt in g if gt not in matched_gold and _match(pt[1], gt[1]) and _match(pt[2], gt[2])), None)
            if hit:
                tp += 1
                matched_gold.add(hit)
            else:
                errors.append({"tipo": tipo, "origem": pt[1], "destino": pt[2], "erro": "extraída mas não está no gabarito"})
        for gt in g:
            if gt not in matched_gold:
                errors.append({"tipo": tipo, "origem": gt[1], "destino": gt[2], "erro": "faltou extrair"})
        precision = tp / len(p) if p else 0.0
        recall = tp / len(g) if g else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_type.append({"relação": tipo, "gabarito": len(g), "extraídas": len(p), "corretas": tp,
                         "precisão": round(precision, 3), "recall": round(recall, 3), "F1": round(f1, 3)})
        total_tp += tp
        total_pred += len(p)
        total_gold += len(g)
    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_gold if total_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_type.append({"relação": "TOTAL", "gabarito": total_gold, "extraídas": total_pred, "corretas": total_tp,
                     "precisão": round(precision, 3), "recall": round(recall, 3), "F1": round(f1, 3)})

    gravidade = {normalize(i["a"]) + "|" + normalize(i["b"]): i["gravidade"] for i in gabarito["interacoes"]}
    gravidade |= {normalize(i["b"]) + "|" + normalize(i["a"]): i["gravidade"] for i in gabarito["interacoes"]}
    inter = [r for r in rows if r["tipo"] == "INTERAGE_COM"]
    certas = sum(1 for r in inter if gravidade.get(normalize(r["a"]) + "|" + normalize(r["b"])) == r["gravidade"])
    return {
        "por_relacao": per_type,
        "gravidade_correta": f"{certas}/{len(inter)}",
        "erros": errors,
        "observacao": (f"Comparação com o gabarito (data/gabarito.json). Gravidade das interações extraídas correta em "
                       f"{certas} de {len(inter)}. Nomes comparados após normalização e similaridade ≥ 0,85."),
    }


def score_answer(text: str, expected: list[list[str]]) -> float:
    folded = normalize(text)
    return sum(1 for group in expected if any(normalize(x) in folded for x in group)) / len(expected)


def evaluate_qa(services, rag: GraphRAG) -> dict:
    details = []
    modes = {"LLM sem contexto": rag.ask_no_context, "RAG só vetorial": rag.ask_vector_only,
             "GraphRAG (este projeto)": lambda q: rag.ask(q, use_llm_cypher=True)}
    for item in QUESTIONS:
        row = {"pergunta": item["pergunta"]}
        for mode, fn in modes.items():
            t0 = time.time()
            ans = fn(item["pergunta"])
            row[f"{mode} · cobertura"] = round(score_answer(ans.resposta, item["esperado"]), 2)
            if mode != "LLM sem contexto":
                row[f"{mode} · sem suporte"] = len(ans.verificacao.nao_fundamentadas)
            row[f"{mode} · s"] = round(time.time() - t0, 1)
            row[f"{mode} · resposta"] = ans.resposta
            print(f"  [{mode}] {row[f'{mode} · cobertura']:.2f} ({row[f'{mode} · s']}s)", flush=True)
        details.append(row)
    summary = []
    for mode in modes:
        cov = [d[f"{mode} · cobertura"] for d in details]
        entry = {"modo": mode, "cobertura média dos fatos esperados": round(sum(cov) / len(cov), 2),
                 "perguntas 100% corretas": f"{sum(1 for c in cov if c == 1.0)}/{len(cov)}",
                 "tempo médio (s)": round(sum(d[f'{mode} · s'] for d in details) / len(details), 1)}
        if mode != "LLM sem contexto":
            entry["entidades sem suporte (total)"] = sum(d[f"{mode} · sem suporte"] for d in details)
        summary.append(entry)
    return {
        "resumo": summary,
        "detalhes": details,
        "metodo": ("Cobertura = fração dos fatos esperados (gabarito) presentes na resposta. 'Sem suporte' = entidades "
                   "citadas na resposta que não aparecem no contexto recuperado (indício de alucinação)."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sem-qa", action="store_true")
    args = parser.parse_args()
    services = build_services()
    retriever = HybridRetriever(services.graph, services.llm, services.resolver, top_k=services.settings.vector_top_k,
                                max_hops=services.settings.max_hops, max_facts=services.settings.max_context_facts)
    rag = GraphRAG(retriever, services.llm, services.resolver)
    out_path = services.settings.data_dir / "avaliacao.json"
    result = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    print("Avaliando extração...")
    result["extracao"] = evaluate_extraction(services)
    for row in result["extracao"]["por_relacao"]:
        print(f"  {row['relação']:18} P={row['precisão']:.2f} R={row['recall']:.2f} F1={row['F1']:.2f}")
    if not args.sem_qa:
        print("Avaliando perguntas (3 modos)...")
        result["qa"] = evaluate_qa(services, rag)
        for row in result["qa"]["resumo"]:
            print(" ", row)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvo em {out_path}")


if __name__ == "__main__":
    main()
