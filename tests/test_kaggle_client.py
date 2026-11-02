"""Mock-only unit tests for the Kaggle HTTP client; no live endpoint required."""

from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

import requests

from llm.kaggle_client import (
    KaggleLLMClient,
    KaggleLLMConfigurationError,
    KaggleLLMConnectionError,
    KaggleLLMHTTPError,
    KaggleLLMResponseError,
    KaggleLLMTimeoutError,
    ask_llm,
)


class KaggleLLMClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = Mock()

    def client(self, env: dict[str, str] | None = None) -> KaggleLLMClient:
        return KaggleLLMClient(
            session=self.session,
            environ=env
            or {
                "KAGGLE_LLM_API_URL": "https://unit.test/tunnel///",
                "KAGGLE_LLM_API_TOKEN": "test-secret-token",
            },
        )

    @staticmethod
    def response(status: int = 200, payload: object | None = None) -> Mock:
        response = Mock()
        response.status_code = status
        response.json.return_value = payload
        return response

    def test_missing_url(self) -> None:
        with self.assertRaisesRegex(KaggleLLMConfigurationError, "KAGGLE_LLM_API_URL"):
            self.client({"KAGGLE_LLM_API_TOKEN": "secret"})

    def test_missing_token(self) -> None:
        with self.assertRaisesRegex(KaggleLLMConfigurationError, "KAGGLE_LLM_API_TOKEN"):
            self.client({"KAGGLE_LLM_API_URL": "https://unit.test"})

    def test_url_normalization(self) -> None:
        client = self.client()
        self.assertEqual(client.base_url, "https://unit.test/tunnel")

    def test_successful_health(self) -> None:
        self.session.get.return_value = self.response(200, {"status": "ok"})
        result = self.client().health()
        self.assertEqual(result, {"status": "ok"})
        self.session.get.assert_called_once_with(
            "https://unit.test/tunnel/health",
            headers={"Authorization": "Bearer test-secret-token"},
            timeout=(5.0, 15.0),
        )

    def test_successful_chat_preserves_messages(self) -> None:
        self.session.post.return_value = self.response(200, {"answer": "hello"})
        messages = [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
        ]
        answer = self.client().chat(messages)
        self.assertEqual(answer, "hello")
        _, kwargs = self.session.post.call_args
        self.assertEqual(kwargs["json"]["messages"], messages)
        self.assertEqual(kwargs["json"]["max_tokens"], 800)
        self.assertEqual(kwargs["json"]["temperature"], 0.2)

    def test_backend_neutral_ask_llm_uses_injected_backend(self) -> None:
        backend = Mock()
        backend.chat.return_value = "adapter response"
        messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}]
        answer = ask_llm(messages, backend=backend)
        self.assertEqual(answer, "adapter response")
        backend.chat.assert_called_once_with(messages, max_tokens=800, temperature=0.2)

    def test_401_message_never_contains_token(self) -> None:
        self.session.post.return_value = self.response(401, {"detail": "private"})
        with self.assertRaises(KaggleLLMHTTPError) as caught:
            self.client().chat([{"role": "user", "content": "hello"}])
        self.assertIn("KAGGLE_LLM_API_TOKEN", str(caught.exception))
        self.assertNotIn("test-secret-token", str(caught.exception))

    def test_404_and_server_error(self) -> None:
        self.session.get.return_value = self.response(404, {})
        with self.assertRaisesRegex(KaggleLLMHTTPError, "HTTP 404"):
            self.client().health()
        self.session.get.return_value = self.response(503, {})
        with self.assertRaisesRegex(KaggleLLMHTTPError, "HTTP 503"):
            self.client().health()

    def test_timeout(self) -> None:
        self.session.get.side_effect = requests.Timeout()
        with self.assertRaises(KaggleLLMTimeoutError):
            self.client().health()

    def test_connection_failure(self) -> None:
        self.session.post.side_effect = requests.ConnectionError()
        with self.assertRaisesRegex(KaggleLLMConnectionError, "Tunnel may be offline"):
            self.client().chat([{"role": "user", "content": "hello"}])

    def test_malformed_json(self) -> None:
        response = self.response(200)
        response.json.side_effect = ValueError("invalid JSON")
        self.session.post.return_value = response
        with self.assertRaises(KaggleLLMResponseError):
            self.client().chat([{"role": "user", "content": "hello"}])

    def test_missing_answer_field(self) -> None:
        self.session.post.return_value = self.response(200, {"result": "hello"})
        with self.assertRaises(KaggleLLMResponseError):
            self.client().chat([{"role": "user", "content": "hello"}])


if __name__ == "__main__":
    unittest.main()
