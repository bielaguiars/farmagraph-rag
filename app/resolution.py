"""Resolução/canonicalização de entidades (Entity Resolution).

Etapas: (1) normalização (caixa, acentos, doses, artigos); (2) dicionário controlado de
princípios ativos, nomes comerciais e sinônimos clínicos; (3) correspondência por tokens;
(4) similaridade aproximada (difflib) contra nomes canônicos já conhecidos do mesmo rótulo.
"""

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

DOSE_PATTERN = re.compile(
    r"\b\d+([.,]\d+)?\s*(mg|mcg|µg|g|ml|ui)\b(\s*(/|por)\s*(dia|kg))?|\b\d+\s*x\s*/?\s*dia\b|\b\d+x\b",
    re.IGNORECASE,
)
NOISE_WORDS = {"comprimido", "comprimidos", "capsula", "capsulas", "uso", "continuo", "oral", "dose"}
LEADING_ARTICLES = re.compile(r"^(o|a|os|as|um|uma|da|do|de|das|dos)\s+")


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def singularize(phrase: str) -> str:
    """Singularização simples do português, suficiente para nomes de classes terapêuticas."""
    contractions = {"dos": "do", "das": "da", "nos": "no", "nas": "na", "os": "o", "as": "a"}
    out = []
    for word in phrase.split():
        w = word
        if w.lower() in contractions:  # "bloqueador dos canais" -> "bloqueador do canal"
            w = contractions[w.lower()]
        elif len(w) > 4 and w.endswith("ais"):
            w = w[:-3] + "al"
        elif len(w) > 4 and w.endswith("res"):
            w = w[:-2]
        elif len(w) > 4 and w.endswith("ões"):
            w = w[:-3] + "ão"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return " ".join(out)


def normalize(text: str) -> str:
    text = strip_accents(text.casefold())
    text = DOSE_PATTERN.sub(" ", text)
    text = re.sub(r"[^\w\s/-]", " ", text)
    words = [w for w in text.split() if w not in NOISE_WORDS]
    text = " ".join(words)
    return LEADING_ARTICLES.sub("", text).strip()


@dataclass
class Resolution:
    surface: str
    canonical: str
    label: str
    method: str  # exato | dicionario | tokens | aproximado | novo


@dataclass
class EntityResolver:
    medication_aliases: dict[str, str] = field(default_factory=dict)  # normalizado -> canônico
    brand_names: dict[str, list[str]] = field(default_factory=dict)  # canônico -> nomes comerciais
    synonyms: dict[str, dict[str, str]] = field(default_factory=dict)  # rótulo -> {normalizado: canônico}
    known: dict[str, dict[str, str]] = field(default_factory=dict)  # rótulo -> {normalizado: canônico}

    @classmethod
    def from_vocabulary(cls, path: Path) -> "EntityResolver":
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))
        resolver = cls()
        for canonical, brands in vocab.get("medicamentos", {}).items():
            resolver.brand_names[canonical] = list(brands)
            resolver.medication_aliases[normalize(canonical)] = canonical
            for brand in brands:
                resolver.medication_aliases[normalize(brand)] = canonical
        for alias, canonical in vocab.get("sinonimos_medicamento", {}).items():
            resolver.medication_aliases[normalize(alias)] = canonical
        resolver.synonyms = {
            "Condicao": {normalize(k): v for k, v in vocab.get("sinonimos_condicao", {}).items()},
            "EfeitoAdverso": {normalize(k): v for k, v in vocab.get("sinonimos_efeito", {}).items()},
        }
        return resolver

    def register(self, label: str, canonical: str) -> None:
        self.known.setdefault(label, {})[normalize(canonical)] = canonical

    def is_medication(self, surface: str) -> bool:
        return self.resolve(surface, "Medicamento").method != "novo"

    def resolve(self, surface: str, label: str) -> Resolution:
        surface = surface.strip()
        # "ibuprofeno (Advil)" -> tenta a parte de fora e a de dentro dos parênteses
        candidates = [re.sub(r"\(.*?\)", " ", surface)] + re.findall(r"\((.*?)\)", surface)
        best: Resolution | None = None
        for candidate in candidates:
            key = normalize(candidate)
            if not key:
                continue
            result = self._resolve_key(surface, key, label)
            if result.method != "novo":
                return result
            best = best or result
        return best or Resolution(surface, surface, label, "novo")

    def _resolve_key(self, surface: str, key: str, label: str) -> Resolution:
        if label == "Medicamento":
            table = self.medication_aliases
        else:
            table = {**self.known.get(label, {}), **self.synonyms.get(label, {})}
        if key in table:
            method = "exato" if normalize(table[key]) == key else "dicionario"
            return Resolution(surface, table[key], label, method)
        if label == "Paciente":
            tokens = set(key.split())
            for known_key, canonical in self.known.get("Paciente", {}).items():
                known_tokens = set(known_key.split())
                if len(tokens) >= 2 and tokens <= known_tokens:
                    return Resolution(surface, canonical, label, "tokens")
            return Resolution(surface, self._title(surface), label, "novo")
        if label == "Medicamento":
            tokens = key.split()
            for alias, canonical in self.medication_aliases.items():
                if len(alias) >= 4 and alias in tokens:
                    return Resolution(surface, canonical, label, "tokens")
        close = difflib.get_close_matches(key, list(table), n=1, cutoff=0.86 if label == "Medicamento" else 0.9)
        if close:
            return Resolution(surface, table[close[0]], label, "aproximado")
        canonical = DOSE_PATTERN.sub(" ", surface).strip(" .,;:").lower()
        canonical = LEADING_ARTICLES.sub("", canonical) if label != "Paciente" else canonical
        canonical = " ".join(canonical.split())
        if label == "ClasseTerapeutica":  # classes costumam vir no plural no texto
            canonical = singularize(canonical)
            if normalize(canonical) in table:
                return Resolution(surface, table[normalize(canonical)], label, "tokens")
        return Resolution(surface, canonical, label, "novo")

    @staticmethod
    def _title(name: str) -> str:
        small = {"da", "de", "do", "das", "dos", "e"}
        return " ".join(w if w.lower() in small else w[:1].upper() + w[1:] for w in name.split())

    def find_in_text(self, text: str) -> list[Resolution]:
        """Encontra menções de entidades conhecidas em um texto livre (ex.: pergunta do usuário)."""
        folded = f" {normalize(text)} "
        found: dict[tuple[str, str], Resolution] = {}
        tables = {"Medicamento": self.medication_aliases}
        for label, values in self.known.items():
            if label != "Medicamento":
                tables[label] = {**values, **self.synonyms.get(label, {})}
        for label, table in tables.items():
            for alias, canonical in table.items():
                if len(alias) >= 3 and f" {alias} " in folded:
                    found.setdefault((label, canonical), Resolution(alias, canonical, label, "dicionario"))
        for known_key, canonical in self.known.get("Paciente", {}).items():
            parts = known_key.split()
            for size in (3, 2):
                for start in range(0, len(parts) - size + 1):
                    if f" {' '.join(parts[start:start + size])} " in folded:
                        found.setdefault(("Paciente", canonical), Resolution(canonical, canonical, "Paciente", "tokens"))
        # remove condições que são substring de outra mais específica (ex.: "dor" dentro de "dor no joelho")
        return list(found.values())
