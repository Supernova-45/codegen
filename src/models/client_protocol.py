from __future__ import annotations

from typing import Any, Protocol

from models.openai_compatible import Usage


class ModelClient(Protocol):
    def clear_trace(self) -> None: ...

    def get_trace(self) -> list[dict[str, Any]]: ...

    def generate_code_candidates(
        self,
        prompt: str,
        n: int,
        function_name: str | None = None,
        signature_hint: str | None = None,
        visible_tests: list[str] | None = None,
        constraints: list[tuple[str, bool]] | None = None,
        temperature: float | None = None,
    ) -> tuple[list[str], Usage]: ...

    def generate_candidate_tests(
        self,
        prompt: str,
        function_name: str | None,
        signature_hint: str | None,
        expected_arity: int | None,
        asked_tests: list[str],
        visible_tests: list[str] | None,
        n_tests: int,
    ) -> tuple[list[str], Usage, dict[str, int]]: ...
