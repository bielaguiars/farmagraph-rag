from dataclasses import dataclass

from app.config import Settings, get_settings
from app.graph import Neo4jGraph
from app.ollama_client import OllamaClient
from app.resolution import EntityResolver


@dataclass
class Services:
    settings: Settings
    graph: Neo4jGraph
    llm: OllamaClient
    resolver: EntityResolver

    def refresh_resolver(self) -> None:
        """Carrega no resolvedor os nomes canônicos que já existem no grafo."""
        for label, names in self.graph.entity_names().items():
            for name in names:
                if name:
                    self.resolver.register(label, name)


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    graph = Neo4jGraph(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, settings.neo4j_database)
    llm = OllamaClient(settings.ollama_url, settings.ollama_chat_model, settings.ollama_embed_model, settings.ollama_timeout)
    resolver = EntityResolver.from_vocabulary(settings.data_dir / "vocabulario.json")
    services = Services(settings, graph, llm, resolver)
    try:
        services.refresh_resolver()
    except Exception:
        pass  # banco ainda vazio ou indisponível
    return services
