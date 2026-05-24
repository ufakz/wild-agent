from src.agents.state import AgentState
from src.config.llm_factory import search_urls
from src.config.models import WildConfig
from src.search.domain_filter import filter_urls_by_domain


async def explorer_node(state: AgentState, *, wild_config: WildConfig) -> dict:
    """Search for URLs matching the current query."""
    explorer_cfg = wild_config.llms.explorer
    queries = state.get("search_queries", [])
    current_index = state.get("current_query_index", 0)

    if current_index >= len(queries):
        if state.get("pending_urls"):
            return {"phase": "harvest"}
        return {"phase": "done", "error": "No URLs found"}

    current_query = queries[current_index]
    themes = state.get("extracted_themes", [])
    visited = set(state.get("visited_urls", []))
    pending = list(state.get("pending_urls", []))

    try:
        new_urls = await search_urls(current_query, themes, explorer_cfg)
        if not new_urls:
            new_urls = await search_urls(
                f"{current_query} articles blog posts",
                themes,
                explorer_cfg,
            )

        new_urls = filter_urls_by_domain(new_urls, wild_config.harvest)

        seen = set(pending) | visited
        for url in new_urls:
            if url not in seen:
                seen.add(url)
                pending.append(url)

        max_pending = wild_config.collection.max_pending_urls
        pending = pending[:max_pending]
        next_index = current_index + 1

        if pending:
            return {
                "pending_urls": pending,
                "current_query_index": next_index,
                "phase": "harvest",
                "fresh_explore": True,
            }

        return {
            "pending_urls": pending,
            "current_query_index": next_index,
            "phase": "explore",
            "error": "Web search returned no URLs for this query",
        }

    except Exception as e:
        return {
            "current_query_index": current_index + 1,
            "phase": "explore",
            "error": f"Search failed: {str(e)}",
        }
