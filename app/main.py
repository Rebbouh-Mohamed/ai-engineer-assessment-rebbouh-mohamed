import asyncio
from contextlib import asynccontextmanager
import logging
import os
from typing import Any, Optional
import dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import httpx

from app.dataset import MovieCatalog
from app.llm_client import GeminiClient, LLMServiceUnavailableError
from app.models import AskRequest, AskResponse, Route, SourceItem
from app.router import classify_question
from app.superhero_client import SuperheroClient
from app.synthesize import synthesize_answer

dotenv.load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chatbot")

OUT_OF_SCOPE_MESSAGE = (
    "I am specialized in answering questions about our movie dataset and superhero characters. "
    "Your question appears to be outside these domains. Please ask about movies, plots, cinematic ratings, "
    "or superhero powers and lore!"
)


def get_dataset_path() -> str:
    """Resolve the dataset path from environment or common defaults."""
    env_path = os.getenv("MOVIES_CSV_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    if os.path.exists("data/movies.csv"):
        return "data/movies.csv"
    if os.path.exists("movies.csv"):
        return "movies.csv"
    raise FileNotFoundError("Could not locate movies.csv dataset file.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to initialize and clean up shared application state."""
    # 1. Load Movie Catalog once at startup
    dataset_path = get_dataset_path()
    logger.info("Loading movies catalog from: %s", dataset_path)
    app.state.catalog = MovieCatalog.from_csv(dataset_path)

    # 2. Shared async HTTP client for external requests
    app.state.http_client = httpx.AsyncClient()

    # 3. Superhero API client
    hero_token = (
        os.getenv("SUPER_HERO_API_KEY")
        or os.getenv("SUPERHERO_API_TOKEN")
        or "placeholder_token"
    )
    app.state.superhero_client = SuperheroClient(token=hero_token, client=app.state.http_client)

    # 4. LLM Client
    gemini_key = os.getenv("GEMINI_API_KEY")
    primary_model = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash")
    fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    app.state.llm_client = GeminiClient(
        api_key=gemini_key,
        primary_model=primary_model,
        fallback_model=fallback_model,
    )

    yield

    # Clean up HTTP client
    await app.state.http_client.aclose()
    logger.info("Application shutdown completed.")


app = FastAPI(
    title="Superhero & Movies QA Chatbot",
    description="FastAPI chatbot answering questions across a movie catalog and the Superhero API.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(LLMServiceUnavailableError)
async def llm_unavailable_handler(request: Request, exc: LLMServiceUnavailableError):
    logger.error("LLM Service Unavailable: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "LLM service is currently unavailable. Please try again shortly."},
    )


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint exposing catalog size and system readiness."""
    catalog: Optional[MovieCatalog] = getattr(app.state, "catalog", None)
    superhero_client: Optional[SuperheroClient] = getattr(app.state, "superhero_client", None)
    return {
        "status": "healthy",
        "movies_loaded": len(catalog.df) if catalog is not None else 0,
        "superhero_cache_size": len(superhero_client._cache) if superhero_client else 0,
    }


async def fetch_movie_context(
    route: Route,
    catalog: MovieCatalog,
    question: str,
) -> tuple[list[dict[str, Any]], list[SourceItem]]:
    """Execute movie retrieval based on the route mode and construct attribution sources."""
    movies: list[dict[str, Any]] = []
    sources: list[SourceItem] = []

    mode = route.dataset_mode or "lookup"

    if mode == "lookup":
        titles = route.movie_titles if route.movie_titles else [question]
        for title in titles:
            match = catalog.lookup_by_title(title)
            if match:
                movies.append(match)
                score = match.get("_match_score", 100)
                sources.append(
                    SourceItem(
                        type="dataset",
                        mode="lookup",
                        detail=f"movies dataset, title: '{match.get('title')}' (id {match.get('id')}, match_score: {score})",
                    )
                )

    elif mode == "keyword":
        query = route.dataset_keywords or question
        matches = catalog.keyword_search(query, top_k=5)
        movies.extend(matches)
        if matches:
            titles_str = ", ".join(f"'{m.get('title')}'" for m in matches[:3])
            sources.append(
                SourceItem(
                    type="dataset",
                    mode="keyword",
                    detail=f"movies dataset, query: '{query}', top matches: [{titles_str}]",
                )
            )

    elif mode == "structured":
        filters = route.dataset_filters
        sort_by = filters.sort_by if filters else "popularity"
        order = filters.order if filters else "desc"
        year = filters.year if filters else None
        min_vote_count = filters.min_vote_count if filters else 50
        limit = filters.limit if filters else 5

        matches = catalog.structured_query(
            sort_by=sort_by,
            order=order,
            year=year,
            min_vote_count=min_vote_count,
            limit=limit,
        )
        movies.extend(matches)
        detail_desc = f"sort_by='{sort_by}', order='{order}', year={year}, limit={limit}"
        sources.append(
            SourceItem(
                type="dataset",
                mode="structured",
                detail=f"movies dataset, query filters: ({detail_desc}), returned {len(matches)} rows",
            )
        )

    return movies, sources


async def fetch_hero_context(
    hero_names: list[str],
    hero_client: SuperheroClient,
) -> tuple[list[dict[str, Any]], list[SourceItem], list[str]]:
    """Fetch superhero profiles concurrently and build attribution sources."""
    heroes: list[dict[str, Any]] = []
    sources: list[SourceItem] = []
    unfound_heroes: list[str] = []

    if not hero_names:
        return heroes, sources, unfound_heroes

    results = await asyncio.gather(
        *(hero_client.search_superhero(name) for name in hero_names),
        return_exceptions=True,
    )

    for name, res in zip(hero_names, results):
        if isinstance(res, Exception):
            logger.warning("Hero lookup error for '%s': %s", name, res)
            unfound_heroes.append(name)
        elif res:
            heroes.append(res)
            source_mode = res.get("_source_mode", "live_api")
            sources.append(
                SourceItem(
                    type="superhero_api",
                    mode=source_mode,
                    detail=f"superhero database, queried: '{name}', resolved: '{res.get('name')}' (id {res.get('id')})",
                )
            )
        else:
            unfound_heroes.append(name)

    return heroes, sources, unfound_heroes


@app.post("/ask", response_model=AskResponse, tags=["Chatbot"])
async def ask(req: AskRequest):
    """Single question-answering endpoint.

    Routes between the local movie dataset and Superhero API (or both/neither),
    retrieves context in parallel, and synthesizes a grounded answer with explicit sources.
    """
    catalog: MovieCatalog = app.state.catalog
    hero_client: SuperheroClient = app.state.superhero_client
    llm_client: GeminiClient = app.state.llm_client

    # 1. Classification & Query Planning (Call #1)
    route = await classify_question(req.question, llm_client)

    # 2. Out-of-scope short-circuit (Neither)
    if not route.needs_dataset and not route.needs_superhero_api:
        return AskResponse(
            answer=OUT_OF_SCOPE_MESSAGE,
            sources=[],
            routing=route,
        )

    # 3. Parallel context retrieval
    fetch_tasks = []
    if route.needs_dataset:
        fetch_tasks.append(fetch_movie_context(route, catalog, req.question))
    else:
        fetch_tasks.append(asyncio.sleep(0, result=([], [])))

    if route.needs_superhero_api:
        fetch_tasks.append(fetch_hero_context(route.hero_names, hero_client))
    else:
        fetch_tasks.append(asyncio.sleep(0, result=([], [], [])))

    fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    # Process movie fetch results
    movie_data, movie_sources = ([], [])
    if not isinstance(fetch_results[0], Exception):
        movie_data, movie_sources = fetch_results[0]
    else:
        logger.error("Error during movie fetch: %s", fetch_results[0])

    # Process superhero fetch results
    hero_data, hero_sources, unfound_heroes = ([], [], [])
    if not isinstance(fetch_results[1], Exception):
        hero_data, hero_sources, unfound_heroes = fetch_results[1]
    else:
        logger.error("Error during superhero fetch: %s", fetch_results[1])

    # Combine sources
    all_sources = movie_sources + hero_sources

    # 4. Answer Synthesis (Call #2)
    answer = await synthesize_answer(
        question=req.question,
        dataset_movies=movie_data,
        superheroes=hero_data,
        llm_client=llm_client,
        unfound_heroes=unfound_heroes,
    )

    return AskResponse(
        answer=answer,
        sources=all_sources,
        routing=route,
    )
