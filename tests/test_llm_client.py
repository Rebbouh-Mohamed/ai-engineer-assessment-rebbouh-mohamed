from unittest.mock import AsyncMock, MagicMock
import pytest
from pydantic import BaseModel
from app.llm_client import GeminiClient, LLMServiceUnavailableError


class SampleSchema(BaseModel):
    summary: str
    score: int


@pytest.mark.asyncio
async def test_llm_structured_primary_success():
    mock_genai_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"summary": "test", "score": 95}'

    mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    client = GeminiClient(
        api_key="fake_key",
        primary_model="primary-model",
        fallback_model="fallback-model",
        client=mock_genai_client,
    )

    result = await client.generate_structured("prompt", SampleSchema)
    assert isinstance(result, SampleSchema)
    assert result.summary == "test"
    assert result.score == 95
    assert mock_genai_client.aio.models.generate_content.call_count == 1
    # Verify primary model was called
    call_kwargs = mock_genai_client.aio.models.generate_content.call_args[1]
    assert call_kwargs["model"] == "primary-model"


@pytest.mark.asyncio
async def test_llm_structured_fallback_on_primary_failure():
    mock_genai_client = MagicMock()
    fallback_resp = MagicMock()
    fallback_resp.text = '{"summary": "fallback result", "score": 80}'

    # 1st call fails (e.g. Rate limit 429), 2nd call succeeds
    mock_genai_client.aio.models.generate_content = AsyncMock(
        side_effect=[
            RuntimeError("Primary model rate limited"),
            fallback_resp,
        ]
    )

    client = GeminiClient(
        api_key="fake_key",
        primary_model="primary-model",
        fallback_model="fallback-model",
        client=mock_genai_client,
    )

    result = await client.generate_structured("prompt", SampleSchema)
    assert result.summary == "fallback result"
    assert result.score == 80
    assert mock_genai_client.aio.models.generate_content.call_count == 2
    second_call_kwargs = mock_genai_client.aio.models.generate_content.call_args_list[1][1]
    assert second_call_kwargs["model"] == "fallback-model"


@pytest.mark.asyncio
async def test_llm_structured_all_fail_raises_503_error():
    mock_genai_client = MagicMock()
    mock_genai_client.aio.models.generate_content = AsyncMock(
        side_effect=[
            RuntimeError("Primary failed"),
            RuntimeError("Fallback failed"),
        ]
    )

    client = GeminiClient(
        api_key="fake_key",
        primary_model="primary-model",
        fallback_model="fallback-model",
        client=mock_genai_client,
    )

    with pytest.raises(LLMServiceUnavailableError):
        await client.generate_structured("prompt", SampleSchema)


@pytest.mark.asyncio
async def test_llm_text_fallback():
    mock_genai_client = MagicMock()
    fallback_resp = MagicMock()
    fallback_resp.text = "This is synthesized text from the fallback model."

    mock_genai_client.aio.models.generate_content = AsyncMock(
        side_effect=[
            RuntimeError("Primary down"),
            fallback_resp,
        ]
    )

    client = GeminiClient(
        api_key="fake_key",
        primary_model="primary-model",
        fallback_model="fallback-model",
        client=mock_genai_client,
    )

    text = await client.generate_text("generate text prompt")
    assert text == "This is synthesized text from the fallback model."
    assert mock_genai_client.aio.models.generate_content.call_count == 2
