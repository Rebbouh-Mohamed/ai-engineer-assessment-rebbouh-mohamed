import json
import logging
from typing import Any, Optional
from app.llm_client import GeminiClient

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are an AI assistant answering questions using information retrieved from two specific sources:
1. A Movie Dataset (local catalog of movies with title, overview, release_date, popularity, vote_average, vote_count).
2. A Superhero API (character database with powerstats, biography, appearance, work, connections).

CRITICAL GROUNDING RULES:
1. Base your answer ONLY on the provided Context below.
2. If the user asks for details that are NOT present in the dataset (for example: director, actors/cast, budget, filming location, or box office earnings), EXPLICITLY state that the movie dataset does not contain cast or director information. Do NOT guess or hallucinate cast or crew from your training memory.
3. If a superhero was queried but not found in the Superhero API, explicitly report that the character was not found in the superhero database.
4. Clearly state where each part of the information came from (e.g., 'From the movie dataset...' or 'According to the Superhero API...').
5. Keep your answer clear, accurate, and concise.
"""


def format_movie_context(movies: list[dict[str, Any]]) -> str:
    """Format movie records into clean readable context text."""
    if not movies:
        return "No movie records found."

    chunks = []
    for i, m in enumerate(movies, start=1):
        overview = m.get("overview") or "No overview available."
        release = m.get("release_date") or "Unknown"
        rating = m.get("vote_average", "N/A")
        votes = m.get("vote_count", "N/A")
        pop = m.get("popularity", "N/A")
        chunks.append(
            f"Movie {i}: '{m.get('title')}' (ID: {m.get('id')})\n"
            f"  - Release Date: {release}\n"
            f"  - Vote Average: {rating} (Votes: {votes})\n"
            f"  - Popularity: {pop}\n"
            f"  - Synopsis: {overview}"
        )
    return "\n\n".join(chunks)


def format_hero_context(heroes: list[dict[str, Any]]) -> str:
    """Format superhero records into readable context text."""
    if not heroes:
        return "No superhero records found."

    chunks = []
    for h in heroes:
        name = h.get("name", "Unknown")
        bio = h.get("biography", {})
        stats = h.get("powerstats", {})
        appearance = h.get("appearance", {})
        work = h.get("work", {})
        connections = h.get("connections", {})

        stats_str = ", ".join(f"{k}: {v}" for k, v in stats.items())
        chunks.append(
            f"Superhero: {name} (ID: {h.get('id')})\n"
            f"  - Real Name: {bio.get('full-name', 'N/A')}\n"
            f"  - Publisher: {bio.get('publisher', 'N/A')}\n"
            f"  - Alignment: {bio.get('alignment', 'N/A')}\n"
            f"  - First Appearance: {bio.get('first-appearance', 'N/A')}\n"
            f"  - Powerstats: {stats_str}\n"
            f"  - Gender/Race: {appearance.get('gender', 'N/A')}, {appearance.get('race', 'N/A')}\n"
            f"  - Occupation: {work.get('occupation', 'N/A')}\n"
            f"  - Affiliations: {connections.get('group-affiliation', 'N/A')}"
        )
    return "\n\n".join(chunks)


def build_synthesis_prompt(
    question: str,
    dataset_movies: list[dict[str, Any]],
    superheroes: list[dict[str, Any]],
    unfound_heroes: Optional[list[str]] = None,
) -> str:
    """Construct the final synthesis prompt combining retrieved contexts."""
    movie_section = format_movie_context(dataset_movies)
    hero_section = format_hero_context(superheroes)

    unfound_section = ""
    if unfound_heroes:
        unfound_section = f"\nSuperheroes searched but not found in the Superhero API: {', '.join(unfound_heroes)}"

    return f"""{SYNTHESIS_SYSTEM_PROMPT}

=== CONTEXT START ===
[Source: Movie Dataset]
{movie_section}

[Source: Superhero API]
{hero_section}{unfound_section}
=== CONTEXT END ===

User Question: {question.strip()}

Synthesized Answer:"""


async def synthesize_answer(
    question: str,
    dataset_movies: list[dict[str, Any]],
    superheroes: list[dict[str, Any]],
    llm_client: GeminiClient,
    unfound_heroes: Optional[list[str]] = None,
) -> str:
    """Invoke LLM call #2 to generate the grounded natural language answer."""
    prompt = build_synthesis_prompt(question, dataset_movies, superheroes, unfound_heroes)
    return await llm_client.generate_text(prompt)
