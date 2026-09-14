import httpx
import pytest
import respx
from app.superhero_client import SuperheroClient, SUPERHERO_API_URL


@pytest.mark.asyncio
async def test_search_superhero_success():
    token = "test_token"
    hero_name = "Aquaman"
    expected_url = SUPERHERO_API_URL.format(token=token, name=hero_name)

    mock_response = {
        "response": "success",
        "results": [
            {
                "id": "38",
                "name": "Aquaman",
                "powerstats": {"intelligence": "75", "strength": "85"},
                "biography": {"full-name": "Arthur Curry", "publisher": "DC Comics"},
            }
        ],
    }

    async with httpx.AsyncClient() as http_client:
        client = SuperheroClient(token=token, client=http_client)
        with respx.mock(base_url="https://superheroapi.com") as respx_mock:
            respx_mock.get(f"/api.php/{token}/search/{hero_name}").respond(
                status_code=200, json=mock_response
            )

            result = await client.search_superhero(hero_name)
            assert result is not None
            assert result["name"] == "Aquaman"
            assert result["_source_mode"] == "live_api"
            assert result["powerstats"]["intelligence"] == "75"


@pytest.mark.asyncio
async def test_search_superhero_not_found():
    token = "test_token"
    hero_name = "NonExistentHero12345"

    mock_response = {
        "response": "error",
        "error": "character with given name not found",
    }

    async with httpx.AsyncClient() as http_client:
        client = SuperheroClient(token=token, client=http_client)
        with respx.mock(base_url="https://superheroapi.com") as respx_mock:
            respx_mock.get(f"/api.php/{token}/search/{hero_name}").respond(
                status_code=200, json=mock_response
            )

            result = await client.search_superhero(hero_name)
            assert result is None


@pytest.mark.asyncio
async def test_search_superhero_timeout_retry():
    token = "test_token"
    hero_name = "Flash"

    mock_response = {
        "response": "success",
        "results": [{"id": "263", "name": "Flash", "powerstats": {"speed": "100"}}],
    }

    async with httpx.AsyncClient() as http_client:
        client = SuperheroClient(token=token, client=http_client)
        with respx.mock(base_url="https://superheroapi.com") as respx_mock:
            # 1st attempt: timeout; 2nd attempt: 200 OK
            route = respx_mock.get(f"/api.php/{token}/search/{hero_name}")
            route.side_effect = [
                httpx.TimeoutException("Connection timed out"),
                httpx.Response(200, json=mock_response),
            ]

            result = await client.search_superhero(hero_name)
            assert result is not None
            assert result["name"] == "Flash"
            assert result["powerstats"]["speed"] == "100"


@pytest.mark.asyncio
async def test_search_superhero_caching():
    token = "test_token"
    hero_name = "Wonder Woman"

    async with httpx.AsyncClient() as http_client:
        client = SuperheroClient(token=token, client=http_client)

        mock_response = {
            "response": "success",
            "results": [{"id": "720", "name": "Wonder Woman", "powerstats": {"power": "100"}}],
        }

        with respx.mock(base_url="https://superheroapi.com") as respx_mock:
            req = respx_mock.get(f"/api.php/{token}/search/{hero_name}").respond(
                status_code=200, json=mock_response
            )

            # 1st call hits API
            res1 = await client.search_superhero(hero_name)
            assert res1 is not None
            assert res1["_source_mode"] == "live_api"
            assert req.call_count == 1

            # 2nd call hits in-memory cache
            res2 = await client.search_superhero(hero_name)
            assert res2 is not None
            assert res2["_source_mode"] == "cached"
            assert req.call_count == 1  # No additional network call!
