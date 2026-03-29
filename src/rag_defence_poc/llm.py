from __future__ import annotations

import httpx


def ollama_chat(
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    timeout_s: float = 120.0,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=payload)
        if r.status_code == 404:
            raise RuntimeError(
                f"Ollama model not found: {model!r}. Install with: ollama pull {model}"
            )
        r.raise_for_status()
        data = r.json()
    msg = data.get("message") or {}
    return str(msg.get("content", "")).strip()


def openai_chat(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    timeout_s: float = 120.0,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install with: pip install openai") from e

    client = OpenAI(api_key=api_key, timeout=timeout_s)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    choice = resp.choices[0].message
    return (choice.content or "").strip()
