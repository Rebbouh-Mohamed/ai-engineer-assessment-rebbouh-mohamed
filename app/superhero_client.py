import asyncio
import logging
from typing import Any, Optional
import httpx

logger = logging.getLogger(__name__)

SUPERHERO_API_URL = "https://superheroapi.com/api.php/{token}/search/{name}"

# Curated fallback knowledge base for major characters to ensure resilience
# when the free Superhero API encounters network outages, IP blocks, or rate limits.
FALLBACK_SUPERHEROES: dict[str, dict[str, Any]] = {
    "batman": {
        "id": "70",
        "name": "Batman",
        "powerstats": {
            "intelligence": "100",
            "strength": "26",
            "speed": "27",
            "durability": "50",
            "power": "47",
            "combat": "100",
        },
        "biography": {
            "full-name": "Bruce Wayne",
            "alter-egos": "No alter egos found.",
            "aliases": ["Insider", "Matches Malone"],
            "place-of-birth": "Crest Hill, Bristol County; Gotham City",
            "first-appearance": "Detective Comics #27",
            "publisher": "DC Comics",
            "alignment": "good",
        },
        "appearance": {
            "gender": "Male",
            "race": "Human",
            "height": ["6'2", "188 cm"],
            "weight": ["210 lb", "95 kg"],
            "eye-color": "blue",
            "hair-color": "black",
        },
        "work": {
            "occupation": "Businessman",
            "base": "Batcave, Stately Wayne Manor, Gotham City",
        },
        "connections": {
            "group-affiliation": "Justice League, Batman Family",
            "relatives": "Thomas Wayne (father, deceased), Martha Wayne (mother, deceased), Damian Wayne (son)",
        },
    },
    "iron man": {
        "id": "346",
        "name": "Iron Man",
        "powerstats": {
            "intelligence": "100",
            "strength": "85",
            "speed": "58",
            "durability": "85",
            "power": "100",
            "combat": "64",
        },
        "biography": {
            "full-name": "Tony Stark",
            "alter-egos": "No alter egos found.",
            "aliases": ["Shellhead", "Golden Avenger"],
            "place-of-birth": "Long Island, New York",
            "first-appearance": "Tales of Suspense #39",
            "publisher": "Marvel Comics",
            "alignment": "good",
        },
        "appearance": {
            "gender": "Male",
            "race": "Human",
            "height": ["6'6", "198 cm"],
            "weight": ["425 lb", "191 kg"],
            "eye-color": "Blue",
            "hair-color": "Black",
        },
        "work": {
            "occupation": "Inventor, Industrialist",
            "base": "Seattle, Washington",
        },
        "connections": {
            "group-affiliation": "Avengers",
            "relatives": "Howard Anthony Stark (father, deceased), Maria Collins Carbonell Stark (mother, deceased)",
        },
    },
    "spider-man": {
        "id": "620",
        "name": "Spider-Man",
        "powerstats": {
            "intelligence": "90",
            "strength": "55",
            "speed": "67",
            "durability": "75",
            "power": "74",
            "combat": "85",
        },
        "biography": {
            "full-name": "Peter Parker",
            "alter-egos": "No alter egos found.",
            "aliases": ["Spidey", "Web-Slinger", "Wall-Crawler"],
            "place-of-birth": "New York, New York",
            "first-appearance": "Amazing Fantasy #15",
            "publisher": "Marvel Comics",
            "alignment": "good",
        },
        "appearance": {
            "gender": "Male",
            "race": "Human",
            "height": ["5'10", "178 cm"],
            "weight": ["165 lb", "74 kg"],
            "eye-color": "Hazel",
            "hair-color": "Brown",
        },
        "work": {
            "occupation": "Freelance photographer, scientist",
            "base": "New York, New York",
        },
        "connections": {
            "group-affiliation": "Avengers, Secret Defenders",
            "relatives": "Richard Parker (father, deceased), Mary Parker (mother, deceased), May Parker (aunt)",
        },
    },
    "superman": {
        "id": "644",
        "name": "Superman",
        "powerstats": {
            "intelligence": "94",
            "strength": "100",
            "speed": "100",
            "durability": "100",
            "power": "100",
            "combat": "85",
        },
        "biography": {
            "full-name": "Clark Kent / Kal-El",
            "alter-egos": "No alter egos found.",
            "aliases": ["Man of Steel", "The Last Son of Krypton"],
            "place-of-birth": "Krypton",
            "first-appearance": "Action Comics #1",
            "publisher": "DC Comics",
            "alignment": "good",
        },
        "appearance": {
            "gender": "Male",
            "race": "Kryptonian",
            "height": ["6'3", "191 cm"],
            "weight": ["225 lb", "101 kg"],
            "eye-color": "Blue",
            "hair-color": "Black",
        },
        "work": {
            "occupation": "Reporter for the Daily Planet",
            "base": "Metropolis",
        },
        "connections": {
            "group-affiliation": "Justice League",
            "relatives": "Jor-El (father, deceased), Lara (mother, deceased), Jonathan Kent (adoptive father), Martha Kent (adoptive mother)",
        },
    },
    "thor": {
        "id": "659",
        "name": "Thor",
        "powerstats": {
            "intelligence": "69",
            "strength": "100",
            "speed": "83",
            "durability": "100",
            "power": "100",
            "combat": "100",
        },
        "biography": {
            "full-name": "Thor Odinson",
            "alter-egos": "No alter egos found.",
            "aliases": ["God of Thunder"],
            "place-of-birth": "Asgard",
            "first-appearance": "Journey into Mystery #83",
            "publisher": "Marvel Comics",
            "alignment": "good",
        },
        "appearance": {
            "gender": "Male",
            "race": "Asgardian",
            "height": ["6'6", "198 cm"],
            "weight": ["640 lb", "288 kg"],
            "eye-color": "Blue",
            "hair-color": "Blond",
        },
        "work": {
            "occupation": "King of Asgard, Warrior",
            "base": "Asgard",
        },
        "connections": {
            "group-affiliation": "Avengers",
            "relatives": "Odin (father), Frigga (mother), Loki (adoptive brother)",
        },
    },
    "wonder woman": {
        "id": "720",
        "name": "Wonder Woman",
        "powerstats": {
            "intelligence": "88",
            "strength": "100",
            "speed": "79",
            "durability": "100",
            "power": "100",
            "combat": "100",
        },
        "biography": {
            "full-name": "Diana Prince",
            "alter-egos": "No alter egos found.",
            "aliases": ["Princess Diana", "Amazon Princess"],
            "place-of-birth": "Themyscira",
            "first-appearance": "All Star Comics #8",
            "publisher": "DC Comics",
            "alignment": "good",
        },
        "appearance": {
            "gender": "Female",
            "race": "Amazon / Demigod",
            "height": ["6'0", "183 cm"],
            "weight": ["165 lb", "75 kg"],
            "eye-color": "Blue",
            "hair-color": "Black",
        },
        "work": {
            "occupation": "Ambassador, Superhero",
            "base": "Gateway City; Washington, D.C.",
        },
        "connections": {
            "group-affiliation": "Justice League",
            "relatives": "Hippolyta (mother), Zeus (father)",
        },
    },
}


class SuperheroClient:
    """Async client for Superhero API with in-memory caching and graceful failover."""

    def __init__(self, token: str, client: Optional[httpx.AsyncClient] = None):
        self.token = token
        self._client = client
        self._cache: dict[str, dict[str, Any]] = {}

    def _normalize_name(self, name: str) -> str:
        return name.strip().lower().replace("-", " ").replace("_", " ")

    def _get_fallback_hero(self, norm_name: str) -> Optional[dict[str, Any]]:
        # Check against normalized keys
        norm_query = norm_name.replace(" ", "")
        for key, hero in FALLBACK_SUPERHEROES.items():
            norm_key = self._normalize_name(key).replace(" ", "")
            if norm_key == norm_query or norm_key in norm_name or norm_name in norm_key:
                return dict(hero)
        return None

    async def search_superhero(
        self,
        name: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Optional[dict[str, Any]]:
        """Fetch superhero information by name.

        Implements:
        1. In-memory cache lookup by normalized name.
        2. Async HTTP call to Superhero API with 5.0s timeout and 1-attempt retry.
        3. Graceful fallback on network unreachable / rate-limit / timeout.
        """
        norm_name = self._normalize_name(name)
        if not norm_name:
            return None

        # 1. In-memory cache hit
        if norm_name in self._cache:
            logger.debug("Superhero cache hit for '%s'", norm_name)
            cached = dict(self._cache[norm_name])
            cached["_source_mode"] = "cached"
            return cached

        # Use passed client or internal client or ephemeral client
        active_client = client or self._client
        owns_client = False
        if active_client is None:
            active_client = httpx.AsyncClient()
            owns_client = True

        url = SUPERHERO_API_URL.format(token=self.token, name=name.strip())

        try:
            for attempt in range(2):
                try:
                    resp = await active_client.get(url, timeout=5.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("response") == "error" or not data.get("results"):
                            logger.info("Hero '%s' not found in Superhero API.", name)
                            return None

                        hero_data = data["results"][0]
                        hero_data["_source_mode"] = "live_api"
                        self._cache[norm_name] = hero_data
                        return hero_data
                    elif resp.status_code == 404:
                        return None
                    else:
                        logger.warning("Superhero API returned HTTP %d on attempt %d", resp.status_code, attempt + 1)
                except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                    logger.warning("Superhero API attempt %d error for '%s': %s", attempt + 1, name, e)
                    # If network is completely unreachable (e.g. firewall/offline), don't stall
                    if isinstance(e, httpx.ConnectError):
                        break
                    if attempt == 0:
                        await asyncio.sleep(0.5)
                        continue
                    break
        finally:
            if owns_client:
                await active_client.aclose()

        # Fallback to local curated superhero entry if network is unreachable
        fallback = self._get_fallback_hero(norm_name)
        if fallback:
            logger.info("Serving hero '%s' from curated resilient cache.", name)
            fallback["_source_mode"] = "resilient_fallback"
            self._cache[norm_name] = fallback
            return fallback

        return None
