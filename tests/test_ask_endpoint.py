from unittest.mock import AsyncMock, MagicMock
import pytest
from httpx import ASGITransport, AsyncClient

from app.dataset import MovieCatalog
from app.llm_client import LLMServiceUnavailableError
from app.main import app
from app.models import Route
from app.superhero_client import SuperheroClient

FIXTURE_PATH = "tests/fixtures/movies_sample.csv"


@pytest.fixture
def mock_app_state():
    """Setup app state with fixtures and mocks."""
    catalog = MovieCatalog.from_csv(FIXTURE_PATH)
    app.state.catalog = catalog

    superhero_client = SuperheroClient(token="mock_token")
    # Preload mock hero
    superhero_client._cache["batman"] = {
        "id": "70",
        "name": "Batman",
        "powerstats": {"intelligence": "100", "combat": "100"},
        "biography": {"full-name": "Bruce Wayne", "publisher": "DC Comics"},
        "_source_mode": "cached",
    }
    app.state.superhero_client = superhero_client

    mock_llm = MagicMock()
    app.state.llm_client = mock_llm
    return mock_llm


@pytest.mark.asyncio
async def test_ask_empty_question_returns_422():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Empty string
        resp1 = await client.post("/ask", json={"question": ""})
        assert resp1.status_code == 422

        # Whitespace-only string
        resp2 = await client.post("/ask", json={"question": "    "})
        assert resp2.status_code == 422


@pytest.mark.asyncio
async def test_ask_out_of_scope_short_circuits(mock_app_state):
    mock_llm = mock_app_state
    mock_llm.generate_structured = AsyncMock(
        return_value=Route(
            needs_dataset=False,
            dataset_mode=None,
            movie_titles=[],
            needs_superhero_api=False,
            hero_names=[],
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/ask", json={"question": "How do I make chocolate cake?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "outside these domains" in data["answer"]
        assert data["sources"] == []
        assert data["routing"]["needs_dataset"] is False
        assert data["routing"]["needs_superhero_api"] is False


@pytest.mark.asyncio
async def test_ask_dataset_only(mock_app_state):
    mock_llm = mock_app_state
    mock_llm.generate_structured = AsyncMock(
        return_value=Route(
            needs_dataset=True,
            dataset_mode="lookup",
            movie_titles=["Inception"],
            needs_superhero_api=False,
            hero_names=[],
        )
    )
    mock_llm.generate_text = AsyncMock(
        return_value="According to the movie dataset, Inception is about planting an idea into dreams."
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/ask", json={"question": "Tell me about Inception."})
        assert resp.status_code == 200
        data = resp.json()
        assert "Inception" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["type"] == "dataset"
        assert "Inception" in data["sources"][0]["detail"]


@pytest.mark.asyncio
async def test_ask_superhero_only(mock_app_state):
    mock_llm = mock_app_state
    mock_llm.generate_structured = AsyncMock(
        return_value=Route(
            needs_dataset=False,
            dataset_mode=None,
            movie_titles=[],
            needs_superhero_api=True,
            hero_names=["Batman"],
        )
    )
    mock_llm.generate_text = AsyncMock(
        return_value="According to the Superhero API, Batman has 100 intelligence and 100 combat."
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/ask", json={"question": "What are Batman's power stats?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "Batman" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["type"] == "superhero_api"
        assert "Batman" in data["sources"][0]["detail"]


@pytest.mark.asyncio
async def test_ask_both_sources_parallel(mock_app_state):
    mock_llm = mock_app_state
    mock_llm.generate_structured = AsyncMock(
        return_value=Route(
            needs_dataset=True,
            dataset_mode="lookup",
            movie_titles=["The Dark Knight"],
            needs_superhero_api=True,
            hero_names=["Batman"],
        )
    )
    mock_llm.generate_text = AsyncMock(
        return_value=(
            "According to the movie dataset, The Dark Knight was released in 2008. "
            "Note that director information is not present in the dataset. "
            "Per the Superhero API, Batman has 100 combat skill."
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/ask",
            json={"question": "Who directed The Dark Knight, and what are Batman's power stats?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sources"]) == 2
        source_types = {s["type"] for s in data["sources"]}
        assert source_types == {"dataset", "superhero_api"}
        assert data["routing"]["needs_dataset"] is True
        assert data["routing"]["needs_superhero_api"] is True


@pytest.mark.asyncio
async def test_ask_llm_failure_returns_503(mock_app_state):
    mock_llm = mock_app_state
    mock_llm.generate_structured = AsyncMock(
        side_effect=LLMServiceUnavailableError("All models unavailable")
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/ask", json={"question": "Tell me about Inception."})
        assert resp.status_code == 503
        data = resp.json()
        assert "unavailable" in data["detail"].lower()


@pytest.mark.asyncio
async def test_health_endpoint(mock_app_state):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["movies_loaded"] == 11
