from __future__ import annotations

import pytest

from src.cli.pipeline import execute_with_retry


def test_execute_with_retry_retries_retryable_exceptions() -> None:
    attempts = {"count": 0}

    def flaky_task() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")
        return "ok"

    result = execute_with_retry(
        "flaky_task",
        flaky_task,
        retries=3,
        retry_delay_seconds=0,
    )

    assert result == "ok"
    assert attempts["count"] == 3


def test_execute_with_retry_does_not_retry_non_retryable_exceptions() -> None:
    attempts = {"count": 0}

    def invalid_task() -> None:
        attempts["count"] += 1
        raise ValueError("invalid input")

    with pytest.raises(ValueError, match="invalid input"):
        execute_with_retry(
            "invalid_task",
            invalid_task,
            retries=3,
            retry_delay_seconds=0,
        )

    assert attempts["count"] == 1
