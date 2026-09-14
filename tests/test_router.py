from unittest.mock import AsyncMock, MagicMock
import pytest
from app.models import DatasetFilters, Route
from app.router import build_router_prompt, classify_question


def test_build_router_prompt_contains_schema_and_question():
    prompt = build_router_prompt("What is the plot of Inception?")
    assert "Inception" in prompt
    assert "Movie Dataset" in prompt
    assert "Superhero API" in prompt
    assert "CRITERIA" in prompt


@pytest.mark.asyncio
async def test_classify_question_lookup():
    mock_llm = MagicMock()
    expected_route = Route(
        needs_dataset=True,
        dataset_mode="lookup",
        movie_titles=["Inception"],
        needs_superhero_api=False,
        hero_names=[],
        reasoning="Movie title lookup",
    )
    mock_llm.generate_structured = AsyncMock(return_value=expected_route)

    route = await classify_question("What is the plot of Inception?", mock_llm)
    assert route.needs_dataset is True
    assert route.dataset_mode == "lookup"
    assert route.movie_titles == ["Inception"]
    assert route.needs_superhero_api is False


@pytest.mark.asyncio
async def test_classify_question_structured():
    mock_llm = MagicMock()
    expected_route = Route(
        needs_dataset=True,
        dataset_mode="structured",
        dataset_filters=DatasetFilters(sort_by="popularity", year=2019, limit=5),
        needs_superhero_api=False,
        hero_names=[],
    )
    mock_llm.generate_structured = AsyncMock(return_value=expected_route)

    route = await classify_question("Top 5 popular movies of 2019", mock_llm)
    assert route.needs_dataset is True
    assert route.dataset_mode == "structured"
    assert route.dataset_filters.year == 2019
    assert route.dataset_filters.sort_by == "popularity"


@pytest.mark.asyncio
async def test_classify_question_superhero_only():
    mock_llm = MagicMock()
    expected_route = Route(
        needs_dataset=False,
        dataset_mode=None,
        movie_titles=[],
        needs_superhero_api=True,
        hero_names=["Spider-Man"],
        reasoning="Query about Spider-Man powerstats",
    )
    mock_llm.generate_structured = AsyncMock(return_value=expected_route)

    route = await classify_question("What are Spider-Man's powerstats?", mock_llm)
    assert route.needs_dataset is False
    assert route.needs_superhero_api is True
    assert "Spider-Man" in route.hero_names


@pytest.mark.asyncio
async def test_classify_question_mixed():
    mock_llm = MagicMock()
    expected_route = Route(
        needs_dataset=True,
        dataset_mode="lookup",
        movie_titles=["The Dark Knight"],
        needs_superhero_api=True,
        hero_names=["Batman"],
        reasoning="Mixed query needing movie overview and character stats",
    )
    mock_llm.generate_structured = AsyncMock(return_value=expected_route)

    route = await classify_question(
        "Who directed The Dark Knight, and what are Batman's power stats?", mock_llm
    )
    assert route.needs_dataset is True
    assert route.needs_superhero_api is True
    assert route.movie_titles == ["The Dark Knight"]
    assert route.hero_names == ["Batman"]


@pytest.mark.asyncio
async def test_classify_question_out_of_scope():
    mock_llm = MagicMock()
    expected_route = Route(
        needs_dataset=False,
        dataset_mode=None,
        movie_titles=[],
        needs_superhero_api=False,
        hero_names=[],
        reasoning="Out-of-scope math/cooking question",
    )
    mock_llm.generate_structured = AsyncMock(return_value=expected_route)

    route = await classify_question("What is 2 + 2?", mock_llm)
    assert route.needs_dataset is False
    assert route.needs_superhero_api is False
