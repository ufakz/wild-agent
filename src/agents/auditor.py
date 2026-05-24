from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agents.state import AgentState, Sample
from src.config.llm_factory import create_structured_chat_model
from src.config.models import WildConfig
from src.similarity.similarity import score_candidate_similarity


class AuditResult(BaseModel):
    relevance_score: int = Field(ge=0, le=10)
    matched_themes: list[str] = Field(description="Themes found in the content")
    reasoning: str = Field(description="Brief explanation of the score")
    is_quality_content: bool = Field(description="Substantive content, not spam/noise")


def create_auditor(config: WildConfig):
    llm = create_structured_chat_model(config.llms.auditor)
    return llm.with_structured_output(AuditResult)


AUDIT_PROMPT = """You are evaluating text content for relevance to a data collection task.

THEMES WE'RE LOOKING FOR:
{themes}

SCORING CRITERIA:
- 9-10: Highly relevant, directly matches themes, quality content
- 7-8: Good relevance, covers related topics
- 5-6: Somewhat relevant, tangentially related
- 3-4: Low relevance, mostly unrelated
- 0-2: Not relevant, spam, or low-quality content

Evaluate this content and provide a score."""


def _theme_labels(state: AgentState) -> list[str]:
    themes = list(state.get("extracted_themes", []))
    user_theme = state.get("theme")
    if user_theme and user_theme not in themes:
        themes.append(user_theme)
    return themes


async def _run_theme_audit(config: WildConfig, state: AgentState, content: str) -> AuditResult:
    llm = create_auditor(config)
    themes = _theme_labels(state)
    messages = [
        SystemMessage(
            content=AUDIT_PROMPT.format(
                themes=", ".join(themes) if themes else "general topics",
            )
        ),
        HumanMessage(content=f"Evaluate this content:\n\n{content[:2000]}"),
    ]
    return await llm.ainvoke(messages)


def _advance_queue(state: AgentState, samples: list[Sample], accepted: bool) -> dict:
    rejected_count = state.get("rejected_count", 0)
    if not accepted:
        rejected_count += 1

    target = state.get("target_count", 10)
    if accepted and len(samples) >= target:
        return {
            "samples": samples,
            "pending_sample": None,
            "pending_scraped": [],
            "rejected_count": rejected_count,
            "phase": "done",
        }

    pending_scraped = list(state.get("pending_scraped", []))
    if pending_scraped:
        return {
            "samples": samples,
            "pending_sample": pending_scraped.pop(0),
            "pending_scraped": pending_scraped,
            "rejected_count": rejected_count,
            "phase": "audit",
        }

    return {
        "samples": samples,
        "pending_sample": None,
        "pending_scraped": [],
        "rejected_count": rejected_count,
        "phase": "harvest",
    }


async def auditor_node(state: AgentState, *, wild_config: WildConfig) -> dict:
    pending_sample = state.get("pending_sample")
    if not pending_sample:
        return {"phase": "harvest"}

    collection_mode = state.get("collection_mode", "theme")
    content = pending_sample["content"]
    samples = list(state.get("samples", []))

    similarity_score: float | None = None
    matched_reference_index: int | None = None
    relevance_score: int | None = None
    matched_themes: list[str] = []
    rejection_reasons: list[str] = []

    try:
        if collection_mode in ("examples", "both"):
            similarity_score, matched_reference_index = await score_candidate_similarity(
                content,
                state.get("reference_embeddings", []),
                embedding_model=state.get("embedding_model", "all-MiniLM-L6-v2"),
            )
            threshold = state.get("similarity_threshold", 0.7)
            if similarity_score < threshold:
                rejection_reasons.append(
                    f"similarity {similarity_score:.3f} < {threshold:.3f}"
                )

        if collection_mode in ("theme", "both"):
            audit = await _run_theme_audit(wild_config, state, content)
            relevance_score = audit.relevance_score
            matched_themes = audit.matched_themes
            min_relevance = state.get("min_relevance", 7)
            if relevance_score < min_relevance:
                rejection_reasons.append(
                    f"relevance {relevance_score}/10 < {min_relevance}/10"
                )
            if not audit.is_quality_content:
                rejection_reasons.append("low quality content")

        if not rejection_reasons:
            samples.append({
                "content": pending_sample["content"],
                "url": pending_sample["url"],
                "title": pending_sample["title"],
                "similarity_score": similarity_score,
                "relevance_score": relevance_score,
                "matched_reference_index": matched_reference_index,
                "themes": matched_themes or _theme_labels(state),
                "scraped_at": pending_sample["scraped_at"],
            })
            return _advance_queue(state, samples, accepted=True)

        return {
            **_advance_queue(state, samples, accepted=False),
            "last_rejection_reason": "; ".join(rejection_reasons),
        }

    except Exception as e:
        return {
            **_advance_queue(state, samples, accepted=False),
            "error": f"Audit failed: {str(e)}",
            "last_rejection_reason": str(e),
        }
