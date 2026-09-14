import logging
import os
from typing import Optional, Type, TypeVar
from google import genai
from google.genai import types
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_PRIMARY_MODEL = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash")
DEFAULT_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")


class LLMServiceUnavailableError(Exception):
    """Raised when all configured LLM providers/models fail."""


class GeminiClient:
    """Wrapper for Google Gemini models supporting primary and fallback failover."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        primary_model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        client: Optional[genai.Client] = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.primary_model = primary_model or DEFAULT_PRIMARY_MODEL
        self.fallback_model = fallback_model or DEFAULT_FALLBACK_MODEL
        self.client = client or genai.Client(api_key=self.api_key)

    @property
    def candidate_models(self) -> list[str]:
        # Deduplicate while preserving priority order
        models = []
        for m in (self.primary_model, self.fallback_model):
            if m and m not in models:
                models.append(m)
        return models

    async def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        """Call Gemini to produce structured JSON matching a Pydantic schema.

        Attempts the primary model first. If rate-limited, timed out, or returning
        server errors, falls back to the secondary model.
        """
        last_err: Optional[Exception] = None

        for model in self.candidate_models:
            try:
                logger.debug("Attempting structured generation with model: %s", model)
                resp = await self.client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1,
                    ),
                )
                if not resp.text:
                    raise ValueError(f"Model {model} returned an empty response.")
                return schema.model_validate_json(resp.text)
            except Exception as e:
                logger.warning(
                    "Model %s failed during structured generation (%s: %s).",
                    model,
                    type(e).__name__,
                    e,
                )
                last_err = e
                continue

        raise LLMServiceUnavailableError(
            f"All configured Gemini models failed. Last error: {last_err}"
        )

    async def generate_text(self, prompt: str) -> str:
        """Generate free-form text response with primary-to-fallback resilience."""
        last_err: Optional[Exception] = None

        for model in self.candidate_models:
            try:
                logger.debug("Attempting text generation with model: %s", model)
                resp = await self.client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                    ),
                )
                if resp.text:
                    return resp.text.strip()
                raise ValueError(f"Model {model} returned an empty text response.")
            except Exception as e:
                logger.warning(
                    "Model %s failed during text generation (%s: %s).",
                    model,
                    type(e).__name__,
                    e,
                )
                last_err = e
                continue

        raise LLMServiceUnavailableError(
            f"All configured Gemini models failed. Last error: {last_err}"
        )
