"""Simple config — set env vars or edit defaults below.

Required: ALIBABA_API_KEY (env var)
All others have sensible defaults.
"""

import os
from pathlib import Path

ALIBABA_API_KEY = os.environ.get("ALIBABA_API_KEY", "")
BASE_URL = os.environ.get(
    "LLM_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-plus")
RAG_URL = os.environ.get("RAG_URL", "http://localhost:8100")

_MCP_DIR = Path(__file__).resolve().parent.parent / "mcp-toolhub"
TOOLHUB_SERVER = os.environ.get("TOOLHUB_SERVER", str(_MCP_DIR / "server.py"))
TOOLHUB_ROOT = os.environ.get("TOOLHUB_ROOT", os.environ.get("USERPROFILE", str(Path.home())))
