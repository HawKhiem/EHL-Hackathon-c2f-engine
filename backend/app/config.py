"""Central configuration. Everything env-driven, validated once at import."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Per-provider default model. Override with LLM_MODEL in .env.
# Only Anthropic has a default — the OpenAI path is an escape hatch, and
# pinning a model id we have not verified would fail confusingly at runtime.
# Set LLM_MODEL explicitly when you switch providers.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-5",
    "openai": "",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Read the repo-root .env so backend and frontend share one file.
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- Supabase ----------
    # publishable = browser-safe, RLS applies (`anon` role)
    # secret      = backend only, BYPASSES RLS (`service_role`)
    # These replace the legacy anon / service_role JWTs, deprecated end of 2026.
    supabase_url: str = ""
    supabase_publishable_key: str = ""
    supabase_secret_key: str = ""

    # ---------- LLM ----------
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_enable_refusal_fallback: bool = True

    # ---------- Server ----------
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def model(self) -> str:
        """Resolved model id for the active provider."""
        resolved = self.llm_model or DEFAULT_MODELS[self.llm_provider]
        if not resolved:
            raise RuntimeError(
                f"LLM_PROVIDER={self.llm_provider} has no default model. "
                "Set LLM_MODEL in .env to the model id you want to use."
            )
        return resolved

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
