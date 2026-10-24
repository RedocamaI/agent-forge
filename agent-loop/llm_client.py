import os
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError, InternalServerError

load_dotenv()

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-flash-latest",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "default_model": "llama3.2:latest",
        "timeout": 300,
    },
}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        max_retries: int = 4,
        timeout: float = 60.0,
    ):
        provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}, choose from {list(PROVIDERS)}")
        cfg = PROVIDERS[provider]
        print("provider: ", provider)

        api_key = os.getenv(cfg["key_env"]) if cfg["key_env"] else "ollama"
        if not api_key:
            raise RuntimeError(f"Missing {cfg["key_env"]}. Add it to your .env file.")

        self.provider = provider
        self.model = model or os.getenv("LLM_MODEL") or cfg["default_model"]
        self.max_retries = max_retries

        self._client = OpenAI(
            api_key=api_key,
            base_url=cfg["base_url"],
            timeout=cfg.get("timeout", timeout),
            max_retries=0,
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        for attempt in range(self.max_retries+1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError):
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)
        
        msg = resp.choices[0].message
        calls = [
            ToolCall(
                id=c.id or f"call_{i}",
                name=c.function.name,
                arguments=c.function.arguments or "{}",
            )
            for i, c in enumerate(msg.tool_calls or [])
        ]
        usage = resp.usage

        return LLMResponse(
            content=msg.content,
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
