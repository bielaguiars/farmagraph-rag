from pathlib import Path

import pytest

from app.extraction import section_relation, split_entity_list, validate_extraction
from app.rag import verify
from app.resolution import EntityResolver, normalize, singularize
from app.retrieval import FORBIDDEN, _drop_contained, _lucene_query, _path_text
from app.text import document_subject, document_type, split_into_chunks, split_sections

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def resolver() -> EntityResolver:
    r = EntityResolver.from_vocabulary(ROOT / "data" / "vocabulario.json")
    r.register("Paciente", "Maria José Ferreira")
    r.register("Condicao", "hipertensão arterial")
    r.register("Condicao", "dor")
    return r


# ---------------------------------------------------------------- resolução
def test_normalize_remove_acentos_doses_e_artigos():
    assert normalize("A Varfarina 5 mg/dia") == "varfarina"
    assert normalize("Ácido Acetilsalicílico") == "acido acetilsalicilico"


def test_nome_comercial_resolve_para_principio_ativo(resolver):
    assert resolver.resolve("Marevan", "Medicamento").canonical == "varfarina"
    assert resolver.resolve("AAS 100 mg", "Medicamento").canonical == "ácido acetilsalicílico"
    assert resolver.resolve("ibuprofeno (Advil)", "Medicamento").canonical == "ibuprofeno"


def test_grafia_aproximada_e_sinonimo(resolver):
    assert resolver.resolve("warfarina", "Medicamento").canonical == "varfarina"
    assert resolver.resolve("losartana potássica", "Medicamento").canonical == "losartana"
    assert resolver.resolve("pressão alta", "Condicao").canonical == "hipertensão arterial"


def test_paciente_por_nome_parcial(resolver):
    assert resolver.resolve("Maria José", "Paciente").canonical == "Maria José Ferreira"


def test_singularizacao_de_classes():
    assert singularize("anticoagulantes orais") == "anticoagulante oral"
    assert singularize("inibidores seletivos da recaptação de serotonina") == "inibidor seletivo da recaptação de serotonina"
    assert singularize("bloqueadores dos canais de cálcio") == "bloqueador do canal de cálcio"


def test_encontra_entidades_na_pergunta(resolver):
    found = {(r.label, r.canonical) for r in resolver.find_in_text("A Maria José pode tomar Advil para dor?")}
    assert ("Medicamento", "ibuprofeno") in found
    assert ("Paciente", "Maria José Ferreira") in found
    assert ("Condicao", "dor") in found


# ------------------------------------------------------------------- texto
def test_prontuario_nao_e_confundido_com_bula():
    assert document_type("PRONTUÁRIO AMBULATORIAL (FICTÍCIO)\n...") == "prontuario"
    assert document_type("BULA SIMPLIFICADA — VARFARINA\n...") == "bula"


def test_assunto_do_documento():
    bula = "BULA SIMPLIFICADA — VARFARINA\nTexto"
    assert document_subject(bula, "bula") == "varfarina"
    pront = "PRONTUÁRIO AMBULATORIAL (FICTÍCIO)\nPaciente fictício criado para fins acadêmicos.\n\nMaria José Ferreira, 72 anos, ..."
    assert document_subject(pront, "prontuario") == "Maria José Ferreira"


def test_bula_dividida_por_secao():
    text = (ROOT / "corpus" / "bula_varfarina.txt").read_text(encoding="utf-8")
    sections = [c.secao for c in split_sections(text)]
    assert "Interações medicamentosas" in sections and "Indicações" in sections
    assert section_relation("Interações medicamentosas") == "INTERAGE_COM"
    assert section_relation("Contraindicações") == "CONTRAINDICADO_EM"
    assert section_relation("Indicações") == "TRATA"
    assert section_relation("Posologia e cuidados") is None


def test_chunks_respeitam_tamanho():
    text = "\n\n".join(["Frase de teste com conteúdo. " * 10] * 6)
    chunks = split_into_chunks(text, 400)
    assert len(chunks) > 1 and all(len(c.texto) <= 400 for c in chunks)


# ------------------------------------------------- restrições semânticas
CHUNK = ("Interações medicamentosas\nA associação com Klaricid tem gravidade alta e causa sangramento. "
         "O uso com Advil também é de gravidade alta.")


def test_relacao_valida_e_resolvida(resolver):
    raw = {"relacoes": [{"origem": "varfarina", "relacao": "INTERAGE_COM", "destino": "Klaricid",
                         "gravidade": "alta", "efeito": "sangramento"}]}
    res = validate_extraction(raw, CHUNK, "varfarina", "Medicamento", resolver, ["INTERAGE_COM"], "bula")
    assert len(res.relacoes) == 1
    rel = res.relacoes[0]
    assert {rel.origem[1], rel.destino[1]} == {"varfarina", "claritromicina"}
    assert rel.propriedades["gravidade"] == "alta"


def test_bloqueia_entidade_nao_ancorada_no_texto(resolver):
    raw = {"relacoes": [{"origem": "varfarina", "relacao": "INTERAGE_COM", "destino": "Aldactone"}]}
    res = validate_extraction(raw, CHUNK, "varfarina", "Medicamento", resolver, ["INTERAGE_COM"], "bula")
    assert not res.relacoes and res.rejeitadas[0]["motivo"] == "entidade não ancorada no texto"


def test_bloqueia_relacao_fora_da_secao(resolver):
    raw = {"relacoes": [{"origem": "varfarina", "relacao": "TRATA", "destino": "sangramento"}]}
    res = validate_extraction(raw, CHUNK, "varfarina", "Medicamento", resolver, ["INTERAGE_COM"], "bula")
    assert not res.relacoes


def test_separa_listas_e_remove_qualificadores():
    assert split_entity_list("dor e febre") == ["dor", "febre"]
    assert split_entity_list("sangramento e úlcera gástrica") == ["sangramento", "úlcera gástrica"]
    assert split_entity_list("pacientes com histórico de angioedema") == ["angioedema"]
    assert split_entity_list("pessoas com epilepsia") == ["epilepsia"]
    assert split_entity_list("insuficiência renal e hepática") == ["insuficiência renal e hepática"]


def test_lista_vira_uma_relacao_por_item(resolver):
    text = "Indicações\nAlívio da dor e febre em adultos."
    raw = {"relacoes": [{"origem": "paracetamol", "relacao": "TRATA", "destino": "dor e febre"}]}
    res = validate_extraction(raw, text, "paracetamol", "Medicamento", resolver, ["TRATA"], "bula")
    assert {r.destino[1] for r in res.relacoes} == {"dor", "febre"}


def test_corrige_direcao_invertida(resolver):
    text = "Maria José Ferreira usa Marevan 5 mg/dia."
    raw = {"relacoes": [{"origem": "Marevan", "relacao": "USA", "destino": "Maria José Ferreira"}]}
    res = validate_extraction(raw, text, "Maria José Ferreira", "Paciente", resolver, ["USA", "TEM_CONDICAO"], "prontuario")
    assert res.corrigidas == 1
    assert res.relacoes[0].origem == ("Paciente", "Maria José Ferreira")
    assert res.relacoes[0].destino == ("Medicamento", "varfarina")


# ------------------------------------------------------------ recuperação
def test_cypher_de_escrita_e_bloqueado():
    assert FORBIDDEN.search("MATCH (n) DETACH DELETE n")
    assert FORBIDDEN.search("CALL apoc.cypher.runWrite('x')")
    assert not FORBIDDEN.search("MATCH (m:Medicamento) RETURN m.nome LIMIT 5")


def test_lucene_ignora_stopwords_e_escapa():
    q = _lucene_query("Pode tomar warfarina com (cuidado)?", covered=set())
    assert "warfarina~1" in q and "tomar" not in q


def test_mantem_entidade_mais_especifica():
    refs = [{"label": "Condicao", "id": "dor"}, {"label": "Condicao", "id": "dor no joelho"}]
    assert _drop_contained(refs) == [{"label": "Condicao", "id": "dor no joelho"}]


def test_texto_do_caminho():
    path = {"nos": [{"id": "1", "nome": "fluoxetina"}, {"id": "2", "nome": "tramadol"}, {"id": "3", "nome": "clonazepam"}],
            "rels": [{"a": "1", "b": "2", "tipo": "INTERAGE_COM"}, {"a": "3", "b": "2", "tipo": "INTERAGE_COM"}]}
    assert _path_text(path) == "fluoxetina -INTERAGE_COM-> tramadol <-INTERAGE_COM- clonazepam"


def test_caminho_de_dois_saltos_vira_conclusao_sem_relacao_direta():
    from app.retrieval import _path_alert

    path = {"nos": [{"id": "1", "label": "Medicamento", "nome": "fluoxetina"},
                    {"id": "2", "label": "Medicamento", "nome": "tramadol"},
                    {"id": "3", "label": "Medicamento", "nome": "clonazepam"}],
            "rels": [{"tipo": "INTERAGE_COM", "a": "1", "b": "2", "props": {"gravidade": "alta"}},
                     {"tipo": "INTERAGE_COM", "a": "3", "b": "2", "props": {"gravidade": "alta"}}]}
    alert = _path_alert(path)
    assert alert["tipo"] == "caminho" and "Não há relação direta" in alert["texto"] and "tramadol" in alert["texto"]
    assert _path_alert({**path, "rels": path["rels"][:1], "nos": path["nos"][:2]}) is None


def test_conclusao_do_grafo_entra_quando_llm_ignora_alerta():
    from app.rag import ensure_alerts
    from app.retrieval import RetrievalResult

    r = RetrievalResult(pergunta="x", alertas=[{"tipo": "interacao", "ref": "K1",
                                                "texto": "José usa sinvastatina, que INTERAGE_COM claritromicina."}])
    out = ensure_alerts("Não encontrei essa informação na base.", r)
    assert out.startswith("Conclusão do grafo:") and "[K1]" in out and "Não encontrei" not in out
    assert ensure_alerts("Há risco [K1].", r) == "Há risco [K1]."


def test_verificacao_detecta_entidade_sem_suporte(resolver):
    context = "[K1] Maria José Ferreira usa varfarina, que INTERAGE_COM ibuprofeno: gravidade alta."
    answer = "Evite ibuprofeno pela varfarina [K1]. Use dipirona [G9]."
    v = verify(answer, context, {"K1"}, resolver)
    assert "dipirona" in v.nao_fundamentadas
    assert v.citacoes_invalidas == ["G9"] and v.citacoes_validas == ["K1"]
