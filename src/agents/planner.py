from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.state import AgentState
from src.config.llm_factory import create_structured_chat_model
from src.config.models import WildConfig


class PlannerOutput(BaseModel):
    themes: list[str] = Field(description="Extracted themes (3-5)")
    queries: list[str] = Field(description="Search queries to find content (3-5)")


def create_planner(config: WildConfig):
    llm = create_structured_chat_model(config.llms.planner)
    return llm.with_structured_output(PlannerOutput)


EXAMPLES_PLANNER_PROMPT = """You are analyzing text samples to extract their common themes.

Given these example texts, identify:
1. The main themes/topics (3-5 themes)
2. Search queries that would find similar content (3-5 queries)

Be specific and actionable. Focus on what makes these texts unique.

For search queries:
- Target pages that contain the same kind of text as the examples (similar format and length), not meta articles about the topic.
- Prefer queries that surface raw text samples users could excerpt directly.
- Avoid queries that mainly return Wikipedia, academic papers, or critical analysis unless the examples are that style."""


THEME_TO_QUERIES_PROMPT = """You are creating a search strategy for data collection.

Given this theme: "{theme}"

Generate:
1. Related sub-themes (3-5)
2. Specific search queries to find content about this topic (3-5)

Make queries specific enough to find quality content, not too broad."""


async def planner_node(state: AgentState, *, wild_config: WildConfig) -> dict:
    llm = create_planner(wild_config)

    if state.get("example_texts") and not state.get("extracted_themes"):
        examples_text = "\n\n---\n\n".join([
            f"Example {i+1}:\n{text[:500]}..."
            for i, text in enumerate(state["example_texts"][:5])
        ])
        messages = [
            SystemMessage(content=EXAMPLES_PLANNER_PROMPT),
            HumanMessage(content=f"Analyze these examples:\n\n{examples_text}"),
        ]
        result: PlannerOutput = await llm.ainvoke(messages)
        return {
            "extracted_themes": result.themes,
            "search_queries": result.queries,
            "current_query_index": 0,
            "phase": "explore",
        }

    if state.get("theme"):
        messages = [
            SystemMessage(content=THEME_TO_QUERIES_PROMPT.format(theme=state["theme"])),
            HumanMessage(content=f"Create a search strategy for: {state['theme']}"),
        ]
        result: PlannerOutput = await llm.ainvoke(messages)
        return {
            "extracted_themes": result.themes,
            "search_queries": result.queries,
            "current_query_index": 0,
            "phase": "explore",
        }

    return {"error": "No theme or examples provided", "phase": "done"}
