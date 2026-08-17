"""Config — reads from .env, points to LiteLLM gateway + privacy routing."""

import json
import os
from pathlib import Path


def _load_dotenv(env_path: str) -> None:
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)
    except FileNotFoundError:
        pass


_ENV_PATH = Path(__file__).parent / ".env"
_load_dotenv(str(_ENV_PATH))

# ——— LiteLLM gateway ———
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1")
BASE_URL = os.environ.get("LLM_BASE_URL", LITELLM_BASE_URL)

# ——— Model aliases ———
MODEL_LOCAL = os.environ.get("MODEL_LOCAL", "ollama-chat")
MODEL_CLOUD = os.environ.get("MODEL_CLOUD", "deepseek-flash")
LLM_MODEL = os.environ.get("LLM_MODEL", MODEL_LOCAL)

# ——— Privacy routing ———
_PRIVACY_FILE = Path(__file__).resolve().parent / "privacy_routes.json"


def _load_privacy() -> dict:
    try:
        with open(_PRIVACY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"default_route": "ollama-chat", "whitelist": {"allowed_sources": [], "allowed_urls": [], "allowed_collections": []}}


PRIVACY = _load_privacy()

# ——— Existing configs ———
RAG_URL = os.environ.get("RAG_URL", "http://localhost:8100")

_TOOLHUB_ROOTS = os.environ.get(
    "TOOLHUB_ROOTS",
    str(Path.home() / "Downloads")
    + os.pathsep
    + "D:\\AI_Control"
    + os.pathsep
    + "G:\\FUNASR",
)

_MCP_DIR = Path(__file__).resolve().parent.parent / "mcp-toolhub"
TOOLHUB_SERVER = os.environ.get("TOOLHUB_SERVER", str(_MCP_DIR / "server.py"))
TOOLHUB_ROOT = os.environ.get("TOOLHUB_ROOT", _TOOLHUB_ROOTS)


def resolve_route(source_path: str = "") -> str:
    """Return model alias based on privacy whitelist.

    If source_path matches any whitelisted prefix → deepseek-flash (cloud).
    Otherwise → ollama-chat (local).
    """
    whitelist = PRIVACY.get("whitelist", {})
    allowed = whitelist.get("allowed_sources", [])
    for prefix in allowed:
        if source_path.startswith(prefix):
            return MODEL_CLOUD
    return MODEL_LOCAL
