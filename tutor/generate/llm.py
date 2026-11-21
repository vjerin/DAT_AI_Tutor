"""HTTP clients for OpenAI-compatible servers (vLLM, Ollama, OpenAI) and Anthropic."""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Optional

import httpx


def _b64(path: Path) -> tuple[str, str]:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return mime, base64.b64encode(path.read_bytes()).decode()


class LLM:
    def complete(self, messages: list[dict], json_mode: bool = True, images: Optional[list[Path]] = None,
                 max_tokens: Optional[int] = None) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAICompatibleLLM(LLM):
    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY",
                 temperature: float = 0.0, max_tokens: int = 700, timeout_s: float = 120):
        self.base_url = base_url.rstrip("/")
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.api_key = os.environ.get(api_key_env, "EMPTY")
        self.client = httpx.Client(timeout=timeout_s)

    def complete(self, messages, json_mode=True, images=None, max_tokens=None):
        msgs = [dict(m) for m in messages]
        if images:
            content = [{"type": "text", "text": msgs[-1]["content"]}]
            for p in images:
                mime, data = _b64(Path(p))
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
            msgs[-1]["content"] = content
        body = {"model": self.model, "messages": msgs, "temperature": self.temperature,
                "max_tokens": max_tokens or self.max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self.client.post(f"{self.base_url}/chat/completions", json=body,
                             headers={"Authorization": f"Bearer {self.api_key}"})
        if r.status_code == 400 and json_mode:      # some servers reject response_format
            body.pop("response_format")
            r = self.client.post(f"{self.base_url}/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


class AnthropicLLM(LLM):
    def __init__(self, model: str, api_key_env: str = "ANTHROPIC_API_KEY", temperature: float = 0.0,
                 max_tokens: int = 700, timeout_s: float = 120, base_url: str = "https://api.anthropic.com"):
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.api_key = os.environ.get(api_key_env, "")
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout_s)

    def complete(self, messages, json_mode=True, images=None, max_tokens=None):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [dict(m) for m in messages if m["role"] != "system"]
        if images:
            content = []
            for p in images:
                mime, data = _b64(Path(p))
                content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
            content.append({"type": "text", "text": msgs[-1]["content"]})
            msgs[-1]["content"] = content
        body = {"model": self.model, "max_tokens": max_tokens or self.max_tokens, "temperature": self.temperature,
                "messages": msgs}
        if system:
            body["system"] = system
        r = self.client.post(f"{self.base_url}/v1/messages", json=body, headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []))


def make_llm(cfg) -> Optional[LLM]:
    s = cfg.section("llm")
    if cfg.get("generator.type") != "llm" and cfg.get("ingest.captioner") != "vlm" \
            and cfg.get("verifier.type") != "llm" and cfg.get("rewrite.type") != "llm" \
            and cfg.get("memory.extractor") != "llm":
        return None
    kw = dict(temperature=s.get("temperature", 0.0), max_tokens=s.get("max_tokens", 700), timeout_s=s.get("timeout_s", 120))
    if s.get("provider") == "anthropic":
        return AnthropicLLM(s["model"], s.get("api_key_env", "ANTHROPIC_API_KEY"), **kw)
    return OpenAICompatibleLLM(s.get("base_url"), s["model"], s.get("api_key_env", "OPENAI_API_KEY"), **kw)


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates code fences / think tags)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in model output")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object in model output")
