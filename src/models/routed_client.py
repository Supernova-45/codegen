from __future__ import annotations

from typing import Any

from models.openai_compatible import OpenAICompatibleClient, Usage


class RoutedModelClient:
    """Routes code generation and test generation to different model clients."""

    def __init__(
        self,
        code_client: OpenAICompatibleClient,
        test_client: OpenAICompatibleClient | None = None,
    ) -> None:
        self._code_client = code_client
        self._test_client = test_client

    def clear_trace(self) -> None:
        self._code_client.clear_trace()
        if self._test_client is not None:
            self._test_client.clear_trace()

    def get_trace(self) -> list[dict[str, Any]]:
        code_trace = [
            _with_role(event, role="codegen")
            for event in self._code_client.get_trace()
        ]
        if self._test_client is None:
            return code_trace
        test_trace = [
            _with_role(event, role="testgen")
            for event in self._test_client.get_trace()
        ]
        return sorted(
            code_trace + test_trace,
            key=lambda x: float(x.get("request_started_at_unix_s", 0.0)),
        )

    def generate_code_candidates(
        self,
        prompt: str,
        n: int,
        function_name: str | None = None,
        signature_hint: str | None = None,
        visible_tests: list[str] | None = None,
        constraints: list[tuple[str, bool]] | None = None,
        temperature: float | None = None,
    ) -> tuple[list[str], Usage]:
        return self._code_client.generate_code_candidates(
            prompt=prompt,
            n=n,
            function_name=function_name,
            signature_hint=signature_hint,
            visible_tests=visible_tests,
            constraints=constraints,
            temperature=temperature,
        )

    def generate_candidate_tests(
        self,
        prompt: str,
        function_name: str | None,
        signature_hint: str | None,
        expected_arity: int | None,
        asked_tests: list[str],
        visible_tests: list[str] | None,
        n_tests: int,
    ) -> tuple[list[str], Usage, dict[str, int]]:
        client = self._test_client or self._code_client
        return client.generate_candidate_tests(
            prompt=prompt,
            function_name=function_name,
            signature_hint=signature_hint,
            expected_arity=expected_arity,
            asked_tests=asked_tests,
            visible_tests=visible_tests,
            n_tests=n_tests,
        )


def _with_role(event: dict[str, Any], role: str) -> dict[str, Any]:
    out = dict(event)
    out["model_role"] = role
    return out
