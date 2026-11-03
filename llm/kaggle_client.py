"""Small environment-configured HTTP client for the Kaggle-hosted Qwen API.

This module only performs inference requests. It does not load a model, add
project data to prompts, or participate in deterministic matching decisions.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from urllib.parse import urlsplit

import requests


_CAPABILITY_FIELDS = frozenset({"item_code", "value", "unit", "method", "score"})
_DEMAND_FIELDS = frozenset(
    {
        "dimension_code",
        "value_state",
        "value",
        "unit",
        "frequency_band",
        "duration_min",
        "essential",
    }
)


def _normalize_section(
    items: Sequence[Mapping[str, Any]],
    *,
    allowed_fields: frozenset[str],
    name: str,
) -> list[dict[str, Any]]:
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise TypeError(f"{name} must be a sequence of objects.")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TypeError(f"{name}[{index}] must be an object.")
        unexpected = set(item) - allowed_fields
        if unexpected:
            fields = ", ".join(sorted(str(field) for field in unexpected))
            raise ValueError(f"{name}[{index}] contains unsupported fields: {fields}.")
        normalized.append({field: item[field] for field in allowed_fields if field in item})
    return normalized


class KaggleLLMError(RuntimeError):
    """Base class for safe, token-free Kaggle API errors."""


class KaggleLLMConfigurationError(KaggleLLMError):
    """Raised when the API URL or bearer token is not configured."""


class KaggleLLMConnectionError(KaggleLLMError):
    """Raised when the API host cannot be reached."""


class KaggleLLMTimeoutError(KaggleLLMError):
    """Raised when an API request exceeds its configured timeout."""


class KaggleLLMResponseError(KaggleLLMError):
    """Raised when an API response is not valid JSON in the expected shape."""


class KaggleLLMHTTPError(KaggleLLMError):
    """Raised for non-success HTTP responses without exposing response bodies."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


class KaggleLLMClient:
    """Use the Kaggle FastAPI service through ordinary HTTP requests."""

    def __init__(
        self,
        *,
        api_url: str | None = None,
        token: str | None = None,
        session: Any | None = None,
        environ: Mapping[str, str] | None = None,
        health_timeout: tuple[float, float] = (5.0, 15.0),
        chat_timeout: tuple[float, float] = (10.0, 300.0),
    ) -> None:
        env = os.environ if environ is None else environ
        raw_url = env.get("KAGGLE_LLM_API_URL") if api_url is None else api_url
        raw_token = env.get("KAGGLE_LLM_API_TOKEN") if token is None else token
        missing = []
        if not raw_url or not raw_url.strip():
            missing.append("KAGGLE_LLM_API_URL")
        if not raw_token or not raw_token.strip():
            missing.append("KAGGLE_LLM_API_TOKEN")
        if missing:
            names = " and ".join(missing)
            raise KaggleLLMConfigurationError(
                f"Kaggle LLM configuration missing: {names} must be configured."
            )

        normalized_url = raw_url.strip().rstrip("/")
        parsed = urlsplit(normalized_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise KaggleLLMConfigurationError(
                "KAGGLE_LLM_API_URL must be an absolute http or https URL."
            )
        self.base_url = normalized_url
        self._token = raw_token.strip()
        self._session = session if session is not None else requests.Session()
        self.health_timeout = health_timeout
        self.chat_timeout = chat_timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _request_json(
        self,
        method: str,
        endpoint: str,
        *,
        timeout: tuple[float, float],
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            if method == "GET":
                response = self._session.get(
                    url, headers=self._headers, timeout=timeout
                )
            else:
                response = self._session.post(
                    url, headers=self._headers, json=json_body, timeout=timeout
                )
        except requests.Timeout:
            raise KaggleLLMTimeoutError(
                "Kaggle LLM API request timed out."
            ) from None
        except requests.ConnectionError:
            raise KaggleLLMConnectionError(
                "Could not connect to Kaggle LLM API. "
                "The Kaggle notebook or Cloudflare Tunnel may be offline."
            ) from None
        except requests.RequestException:
            raise KaggleLLMConnectionError(
                "Kaggle LLM API request failed before receiving a response."
            ) from None

        status = response.status_code
        if status == 401:
            raise KaggleLLMHTTPError(
                status,
                "Kaggle LLM authentication failed (HTTP 401). "
                "Check KAGGLE_LLM_API_TOKEN.",
            )
        if status == 404:
            raise KaggleLLMHTTPError(
                status,
                f"Kaggle LLM endpoint was not found (HTTP 404): /{endpoint.lstrip('/')}.",
            )
        if status >= 500:
            raise KaggleLLMHTTPError(
                status, f"Kaggle LLM service error (HTTP {status})."
            )
        if not 200 <= status < 300:
            raise KaggleLLMHTTPError(
                status, f"Kaggle LLM request failed (HTTP {status})."
            )
        try:
            payload = response.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            raise KaggleLLMResponseError(
                f"Kaggle LLM endpoint /{endpoint.lstrip('/')} returned malformed JSON."
            ) from None
        if not isinstance(payload, dict):
            raise KaggleLLMResponseError(
                f"Kaggle LLM endpoint /{endpoint.lstrip('/')} returned a JSON value, not an object."
            )
        return payload

    def health(self) -> dict[str, Any]:
        """Return the JSON object from GET /health."""
        return self._request_json("GET", "/health", timeout=self.health_timeout)

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        """Send structured chat messages to POST /chat and return answer text."""
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise TypeError("messages must be a sequence of role/content objects.")
        normalized_messages: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(f"messages[{index}] must be an object.")
            role, content = message.get("role"), message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"messages[{index}].role must be system, user, or assistant."
                )
            if not isinstance(content, str):
                raise ValueError(f"messages[{index}].content must be a string.")
            normalized_messages.append({"role": role, "content": content})
        if not normalized_messages:
            raise ValueError("messages must contain at least one chat message.")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer.")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be a finite non-negative number.")
        if not math.isfinite(float(temperature)) or temperature < 0:
            raise ValueError("temperature must be a finite non-negative number.")

        payload = self._request_json(
            "POST",
            "/chat",
            timeout=self.chat_timeout,
            json_body={
                "messages": normalized_messages,
                "max_tokens": max_tokens,
                "temperature": float(temperature),
            },
        )
        answer = payload.get("answer")
        if not isinstance(answer, str):
            raise KaggleLLMResponseError(
                "Kaggle LLM /chat response must contain a string field named answer."
            )
        return answer

    def match(
        self,
        candidate_capabilities: Sequence[Mapping[str, Any]],
        job_demands: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Send only capability and demand sections to the dedicated /match endpoint."""
        payload = {
            "candidate_capabilities": _normalize_section(
                candidate_capabilities,
                allowed_fields=_CAPABILITY_FIELDS,
                name="candidate_capabilities",
            ),
            "job_demands": _normalize_section(
                job_demands,
                allowed_fields=_DEMAND_FIELDS,
                name="job_demands",
            ),
        }
        response = self._request_json(
            "POST",
            "/match",
            timeout=self.chat_timeout,
            json_body=payload,
        )
        if response.get("status") != "ok" or not isinstance(response.get("results"), list):
            raise KaggleLLMResponseError(
                "Kaggle LLM /match response must contain status=ok and a results list."
            )
        for index, result in enumerate(response["results"]):
            if not isinstance(result, Mapping):
                raise KaggleLLMResponseError(
                    f"Kaggle LLM /match result {index} must be an object."
                )
            if result.get("status") not in {"Met", "Met with accommodation", "Gap", "Unknown"}:
                raise KaggleLLMResponseError(
                    f"Kaggle LLM /match result {index} has an invalid status."
                )
            if result.get("confidence") not in {"high", "medium", "low"}:
                raise KaggleLLMResponseError(
                    f"Kaggle LLM /match result {index} has an invalid confidence."
                )
            if not isinstance(result.get("dimension_code"), str):
                raise KaggleLLMResponseError(
                    f"Kaggle LLM /match result {index} is missing dimension_code."
                )
        return response


class ChatBackend(Protocol):
    """Stable chat interface that allows another backend to be injected later."""

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str: ...


def kaggle_health() -> dict[str, Any]:
    """Convenience function that checks GET /health using current environment."""
    return KaggleLLMClient().health()


def ask_kaggle(
    messages: Sequence[Mapping[str, str]],
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> str:
    """Convenience function that chats with the configured Kaggle endpoint."""
    return KaggleLLMClient().chat(
        messages, max_tokens=max_tokens, temperature=temperature
    )


def ask_llm(
    messages: Sequence[Mapping[str, str]],
    max_tokens: int = 800,
    temperature: float = 0.2,
    *,
    backend: ChatBackend | None = None,
) -> str:
    """Backend-neutral interface; defaults to Kaggle and accepts injected adapters."""
    selected_backend = backend if backend is not None else KaggleLLMClient()
    return selected_backend.chat(
        messages, max_tokens=max_tokens, temperature=temperature
    )
