from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class DatasetFilters(BaseModel):
    """Filters and sorting options for structured movie queries."""

    sort_by: Literal["popularity", "vote_average", "release_date"] = "popularity"
    order: Literal["asc", "desc"] = "desc"
    year: Optional[int] = None
    min_vote_count: int = 50
    limit: int = 5


class Route(BaseModel):
    """Routing and entity extraction output from the router classifier (Call #1)."""

    needs_dataset: bool = Field(
        ...,
        description="True if the question asks about movies, cinema, or films in the dataset.",
    )
    dataset_mode: Optional[Literal["lookup", "keyword", "structured"]] = Field(
        default=None,
        description="The retrieval mode: 'lookup' for specific movie titles, 'keyword' for thematic/plot searches, 'structured' for ranking/filtering.",
    )
    movie_titles: list[str] = Field(
        default_factory=list,
        description="Movie titles extracted from the question for lookup.",
    )
    dataset_keywords: Optional[str] = Field(
        default=None,
        description="Keywords or plot description for thematic search in the movie dataset.",
    )
    dataset_filters: Optional[DatasetFilters] = Field(
        default=None,
        description="Filters and sorting parameters if dataset_mode is structured.",
    )
    needs_superhero_api: bool = Field(
        ...,
        description="True if the question asks about superhero powers, stats, alter egos, or comic lore.",
    )
    hero_names: list[str] = Field(
        default_factory=list,
        description="List of superhero or villain names to search in the Superhero API.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of the routing classification.",
    )


class SourceItem(BaseModel):
    """Attribution item specifying where the response data originated."""

    type: Literal["dataset", "superhero_api"] = Field(
        ..., description="Origin of the data: 'dataset' or 'superhero_api'."
    )
    mode: Optional[str] = Field(
        default=None, description="Retrieval mode, e.g., 'lookup', 'keyword', 'structured', 'cached'."
    )
    detail: str = Field(
        ..., description="Specific detail of the item, ID, query, or character name accessed."
    )


class AskRequest(BaseModel):
    """Incoming request payload for POST /ask."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language question to be answered by the chatbot.",
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Question cannot be empty or whitespace only.")
        return trimmed


class AskResponse(BaseModel):
    """Outgoing response payload from POST /ask."""

    answer: str = Field(..., description="The synthesized natural language answer.")
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="List of data sources consulted to produce the answer.",
    )
    routing: Optional[Route] = Field(
        default=None,
        description="Diagnostic routing metadata detailing the sources and extraction parameters.",
    )
