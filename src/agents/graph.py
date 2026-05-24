from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.auditor import auditor_node
from src.agents.crawler_session import CrawlerSession
from src.agents.explorer import explorer_node
from src.agents.harvester import harvester_node
from src.agents.planner import planner_node
from src.agents.state import AgentState, derive_collection_mode
from src.config.models import WildConfig
from src.similarity.similarity import embed_reference_texts


def _at_cycle_limit(state: AgentState) -> bool:
    return state.get("iteration", 0) >= state.get("max_iterations", 50)


def route_by_phase(state: AgentState) -> str:
    phase = state.get("phase", "plan")
    if len(state.get("samples", [])) >= state.get("target_count", 10):
        return "end"
    if phase == "done":
        return "end"

    if _at_cycle_limit(state):
        if phase == "explore":
            return "end"
        if phase == "harvest" and state.get("fresh_explore", False):
            return "end"

    return {
        "plan": "planner",
        "explore": "explorer",
        "harvest": "harvester",
        "audit": "auditor",
        "done": "end",
    }.get(phase, "end")


def router_node(state: AgentState) -> dict:
    return {}


def _create_graph(
    config: WildConfig,
    crawler_session: CrawlerSession | None = None,
):
    # Graph definition
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("planner", partial(planner_node, wild_config=config))
    graph.add_node("explorer", partial(explorer_node, wild_config=config))
    graph.add_node(
        "harvester",
        partial(
            harvester_node,
            wild_config=config,
            crawler_session=crawler_session,
        ),
    )
    graph.add_node("auditor", partial(auditor_node, wild_config=config))

    # Add edges
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_by_phase,
        {
            "planner": "planner",
            "explorer": "explorer",
            "harvester": "harvester",
            "auditor": "auditor",
            "end": END,
        },
    )
    graph.add_edge("planner", "router")
    graph.add_edge("explorer", "router")
    graph.add_edge("harvester", "router")
    graph.add_edge("auditor", "router")
    return graph.compile()


def get_initial_state(config: WildConfig) -> AgentState:
    coll = config.collection
    return {
        "theme": coll.theme,
        "example_texts": coll.examples,
        "target_count": coll.target_count,
        "min_relevance": coll.min_relevance,
        "similarity_threshold": coll.similarity_threshold,
        "collection_mode": derive_collection_mode(coll.theme, coll.examples),
        "reference_embeddings": [],
        "embedding_model": config.embeddings.model,
        "extracted_themes": [],
        "search_queries": [],
        "current_query_index": 0,
        "pending_urls": [],
        "visited_urls": [],
        "pending_sample": None,
        "pending_scraped": [],
        "samples": [],
        "rejected_count": 0,
        "last_rejection_reason": None,
        "phase": "plan",
        "iteration": 0,
        "max_iterations": coll.max_iterations,
        "fresh_explore": False,
        "error": None,
    }


def _sort_samples(samples: list[dict], collection_mode: str) -> list[dict]:
    if collection_mode in ("examples", "both"):
        return sorted(
            samples, key=lambda s: s.get("similarity_score") or 0.0, reverse=True
        )
    return sorted(samples, key=lambda s: s.get("relevance_score") or 0, reverse=True)


async def run_collection(config: WildConfig) -> dict[str, Any]:
    coll = config.collection
    if not coll.theme and not coll.examples:
        raise ValueError("Config must include either collection.theme or collection.examples")

    reference_embeddings: list[list[float]] = []
    if coll.examples:
        reference_embeddings = await embed_reference_texts(
            coll.examples,
            embedding_model=config.embeddings.model,
        )

    async with CrawlerSession() as crawler_session:
        graph = _create_graph(config, crawler_session=crawler_session)
        initial_state = get_initial_state(config)
        initial_state["reference_embeddings"] = reference_embeddings

        final_state = await graph.ainvoke(initial_state)
    samples = list(final_state.get("samples", []))
    mode = final_state.get("collection_mode", "theme")
    final_state["samples"] = _sort_samples(samples, mode)
    return final_state
