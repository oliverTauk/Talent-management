"""
GitHub Models API client for LLM inference.

Uses the GitHub Models endpoint:
    POST https://models.github.ai/inference/chat/completions

Authentication:
    Set env var GITHUB_MODELS_TOKEN to a fine-grained PAT with models:read.

Usage:
    >>> import os
    >>> os.environ["GITHUB_MODELS_TOKEN"] = "ghp_..."
    >>> from hr_analytics.services.github_models_client import GitHubModelsClient
    >>> client = GitHubModelsClient()
    >>> response = client.complete("You are helpful.", "Say hello.")
    >>> print(response)
"""

from __future__ import annotations

import json
import logging
import time

import requests

logger = logging.getLogger(__name__)

_API_URL = "https://models.github.ai/inference/chat/completions"
_DEFAULT_MODEL = "openai/gpt-4.1"
_DEFAULT_MAX_TOKENS = 16384
_REQUEST_TIMEOUT = 180       # seconds
_RETRY_BACKOFF_BASE = 2      # exponential backoff base (seconds)
_MAX_HTTP_RETRIES = 3        # retries on transient HTTP errors (429, 5xx)


class GitHubModelsClient:
    """
    Thin HTTP client for the GitHub Models inference API.

    Parameters
    ----------
    token : str
        GitHub PAT with ``models:read`` permission.
    model : str
        Model identifier, e.g. ``"openai/gpt-4.1"``.
    max_tokens : int
        Maximum tokens for the completion.
    """

    def __init__(
        self,
        token: str,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ):
        if not token:
            raise EnvironmentError(
                "GitHub Models token is empty. "
                "Set the GITHUB_MODELS_TOKEN environment variable to a "
                "fine-grained PAT with models:read permission."
            )
        self._token = token
        self._model = model
        self._max_tokens = max_tokens
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        })

    def complete(self, system: str, user: str) -> str:
        """
        Send a chat-completion request and return the assistant's text.

        Retries automatically on HTTP 429 (rate limit) and 5xx errors
        with exponential backoff.

        Raises
        ------
        RuntimeError
            If all retries are exhausted or a non-retryable error occurs.
        """
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        last_error: Exception | None = None
        for attempt in range(1, _MAX_HTTP_RETRIES + 1):
            try:
                resp = self._session.post(
                    _API_URL,
                    data=json.dumps(payload),
                    timeout=_REQUEST_TIMEOUT,
                )

                # ---------- handle retryable status codes ----------
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = _RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "GitHub Models API returned %d on attempt %d/%d.  "
                        "Retrying in %ds…",
                        resp.status_code, attempt, _MAX_HTTP_RETRIES, wait,
                    )
                    time.sleep(wait)
                    last_error = RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                    continue

                # ---------- non-retryable HTTP errors ----------
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"GitHub Models API error {resp.status_code}: "
                        f"{resp.text[:500]}"
                    )

                # ---------- parse response ----------
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError(
                        "GitHub Models API returned an empty choices array."
                    )
                return choices[0]["message"]["content"]

            except requests.exceptions.RequestException as exc:
                wait = _RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Network error on attempt %d/%d: %s.  Retrying in %ds…",
                    attempt, _MAX_HTTP_RETRIES, exc, wait,
                )
                time.sleep(wait)
                last_error = exc

        raise RuntimeError(
            f"GitHub Models API failed after {_MAX_HTTP_RETRIES} attempts. "
            f"Last error: {last_error}"
        )


# ============================================================
# Self-test
# ============================================================

def _self_test() -> None:  # pragma: no cover
    """Quick smoke test – call with a trivial prompt."""
    import os
    token = os.environ.get("GITHUB_MODELS_TOKEN", "")
    if not token:
        print("Set GITHUB_MODELS_TOKEN to run the self-test.")
        return
    client = GitHubModelsClient(token=token)
    reply = client.complete(
        system="You are a helpful assistant.",
        user='Return exactly: {"status": "ok"}',
    )
    print("Response:", reply)


if __name__ == "__main__":
    _self_test()
