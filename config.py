"""Simple config — reads from .env file or environment variables."""

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

ALIBABA_API_KEY = os.environ.get("ALIBABA_API_KEY", "")
BASE_URL = os.environ.get(
    "LLM_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-plus")
RAG_URL = os.environ.get("RAG_URL", "http://localhost:8100")

# ToolHub: allow access to workspace directories, not entire HOME
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
