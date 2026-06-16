'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. ...
'''

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union


JsonDict = dict[str, Any]


class OpenAICompatibleModelClient:
    """
    Minimal client for OpenAI-compatible HTTP APIs (models, chat, completions).
    """

    def __init__(
        self,
        model_name: str,
        port: int,
        host: str = "127.0.0.1",
        scheme: str = "http",
        timeout_s: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.port = int(port)
        self.host = host
        self.scheme = scheme
        self.timeout_s = timeout_s
        self.base_url = f"{scheme}://{host}:{self.port}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[JsonDict] = None,
        headers: Optional[MutableMapping[str, str]] = None,
    ) -> tuple[int, Any]:
        url = f"{self.base_url}{path}"
        hdrs = dict(headers or {})
        data: Optional[bytes] = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")

        req = urllib.request.Request(url, data=data, headers=hdrs, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                code = int(getattr(resp, "status", 200))
                try:
                    return code, json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    return code, raw
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = raw
            return int(e.code), payload

    
    def list_models(self) -> tuple[int, Any]:
        return self._request("GET", "/v1/models")

    def chat_completions(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 32,
        temperature: float = 0.0,
        extra: Optional[JsonDict] = None,
    ) -> tuple[int, Any]:
        body: JsonDict = {
            "model": self.model_name,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra:
            body.update(extra)
        return self._request("POST", "/v1/chat/completions", body=body)

    def completions(
        self,
        prompt: Union[str, Sequence[str]],
        *,
        max_tokens: int = 32,
        temperature: float = 0.0,
        extra: Optional[JsonDict] = None,
    ) -> tuple[int, Any]:
        body: JsonDict = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra:
            body.update(extra)
        return self._request("POST", "/v1/completions", body=body)

    def simple_chat(self, user_text: str, **kwargs: Any) -> tuple[int, Any]:
        """Convenience: single user message."""
        return self.chat_completions(
            [{"role": "user", "content": user_text}],
            **kwargs,
        )

    def chat_completions_structured_book(
            self,
            *,
            max_tokens: int = 256,
            temperature: float = 0.0,
        ) -> tuple[int, Any]:
        """
        POST /v1/chat/completions with ``response_format: {type: json_object}``
        and a prompt that asks for one book (title, author, year, genre).
        """
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "Respond with a single JSON object only. "
                    "No markdown or text outside the JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return one book as a JSON object with keys: "
                    "title (string), author (string), year (integer), genre (string)."
                ),
            },
        ]
        extra: JsonDict = {"response_format": {"type": "json_object"}}
        return self.chat_completions(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra=extra,
        )