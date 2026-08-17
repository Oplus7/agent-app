"""
Agent App Server — FastAPI wrapper with SSE streaming + web UI.

Usage:
    python server.py

Opens http://localhost:8101 with chat interface.
"""

import asyncio
import datetime
import json
import os
import re
import sys
from pathlib import Path

import httpx

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import OpenAI

import config
from rag_client import RAGClient
from mcp_client import ToolHubClient

app = FastAPI(title="Agent App")

SYSTEM_PROMPT = """You are a personal AI agent with access to knowledge base and filesystem tools.
Be concise but helpful. When asked to find or recall information, check the knowledge base first.
Synthesize results clearly, citing sources when available."""

RAG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Search the personal knowledge base. Returns relevant document chunks with source info and scores.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Natural language search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chat_knowledge",
            "description": "Ask the knowledge base a question. Retrieves documents and generates an answer.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Full question"}},
                "required": ["query"],
            },
        },
    },
]


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extract_tool_call(text: str):
    """qwen3 开启思考时，工具调用可能被塞进 reasoning_content 而非结构化 tool_calls。
    这里尝试从思考文本里解析出一个工具调用 JSON（{\"name\": ..., \"arguments\": ...}）。"""
    t = (text or "").strip()
    candidates = [t]
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            return obj
    return None


class AgentService:
    def __init__(self):
        self.rag = RAGClient(config.RAG_URL)
        self.toolhub = ToolHubClient(config.TOOLHUB_SERVER, config.TOOLHUB_ROOT)
        self._rag_ok = False
        self._toolhub_ok = False
        self._tool_names: list[str] = []

    async def connect(self):
        try:
            self._rag_ok = self.rag.health()
        except Exception:
            self._rag_ok = False
        try:
            await self.toolhub.connect()
            self._toolhub_ok = True
            self._tool_names = self.toolhub.tool_names
        except Exception:
            self._toolhub_ok = False

    async def disconnect(self):
        if self._toolhub_ok:
            await self.toolhub.disconnect()

    @property
    def tools(self):
        all_tools = list(RAG_TOOLS)
        if self._toolhub_ok:
            all_tools += self.toolhub.tool_schemas
        return all_tools

    async def chat_stream(self, user_message: str, history: list[dict]):
        api_key = config.DEEPSEEK_API_KEY or "no-key-required"
        llm = OpenAI(api_key=api_key, base_url=config.BASE_URL)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_message},
        ]

        tooling_disabled = False
        tool_calls_made = 0
        TOOL_BUDGET = 4
        for _ in range(8):
            # 流式：qwen3 的思考经 litellm 映射到 delta.reasoning_content，答案在 delta.content
            stream = llm.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                tools=[] if tooling_disabled else self.tools,
                temperature=0.3,
                stream=True,
            )

            tool_calls: dict = {}
            content_buf = ""
            thinking_buf = ""
            cleared_thinking = False

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # 思考过程
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    thinking_buf += rc
                    yield _sse({"type": "thinking", "content": rc})

                # 可见回答
                if delta.content:
                    content_buf += delta.content
                    yield _sse({"type": "message", "content": delta.content})

                # 工具调用：流式增量累积
                if delta.tool_calls:
                    # 本轮出现工具调用时，思考内容通常只是工具 JSON，清掉思考块避免把 JSON 当思考展示
                    if thinking_buf and not cleared_thinking:
                        yield _sse({"type": "thinking_clear"})
                        cleared_thinking = True
                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        slot = tool_calls.setdefault(
                            idx, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                slot["name"] += tc.function.name
                            if tc.function.arguments:
                                slot["arguments"] += tc.function.arguments

            # qwen3 开启思考时，工具调用会进入 reasoning_content 而非结构化 tool_calls。
            # 兜底：从思考文本里解析出工具调用，保证 agent 仍能执行工具。
            if not tool_calls and thinking_buf.strip():
                parsed = _extract_tool_call(thinking_buf)
                if parsed:
                    tool_calls = {0: {
                        "id": "call_fallback",
                        "name": parsed.get("name", ""),
                        "arguments": json.dumps(parsed.get("arguments", {}), ensure_ascii=False),
                    }}
                    # 这一轮的“思考”其实只是工具调用 JSON，清掉思考块，避免把 JSON 当思考展示
                    yield _sse({"type": "thinking_clear"})

            # qwen3 也可能把工具调用以纯文本 JSON 写进 content（而非结构化 tool_calls）。
            # 兜底：从答案文本里解析出工具调用并执行，同时清掉被误显示为答案的 JSON。
            if not tool_calls and content_buf.strip():
                parsed = _extract_tool_call(content_buf)
                if parsed and parsed.get("name") in {t["function"]["name"] for t in self.tools}:
                    tool_calls = {0: {
                        "id": "call_fallback",
                        "name": parsed.get("name", ""),
                        "arguments": json.dumps(parsed.get("arguments", {}), ensure_ascii=False),
                    }}
                    yield _sse({"type": "message_clear"})

            # 工具调用预算：整个对话最多执行 TOOL_BUDGET 次工具，超过即禁用后续工具调用，
            # 逼模型给出最终回答，避免 qwen3 反复调工具死循环。
            if tool_calls:
                deduped = {}
                for idx, c in tool_calls.items():
                    if tool_calls_made >= TOOL_BUDGET:
                        tooling_disabled = True
                        continue
                    tool_calls_made += 1
                    deduped[idx] = c
                tool_calls = deduped

            if tool_calls:
                calls = list(tool_calls.values())
                messages.append({
                    "role": "assistant",
                    "content": content_buf or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]},
                        }
                        for c in calls
                    ],
                })

                for c in calls:
                    try:
                        args = json.loads(c["arguments"]) if c["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    yield _sse({"type": "tool_call", "tool": c["name"], "args": c["arguments"]})
                    result = await self._execute(c["name"], args)
                    yield _sse({"type": "tool_result", "tool": c["name"], "content": result})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": c["id"],
                        "content": result,
                    })
            else:
                # 若模型在多轮工具后仍吐出工具调用 JSON 而非自然语言（qwen3 常见），
                # 不把 JSON 当答案展示，改为一句干净的提示（结果已在上方工具块中）。
                final = content_buf
                if _extract_tool_call(final):
                    final = "（已执行工具，结果见上方工具块。）"
                    yield _sse({"type": "message_clear"})
                    yield _sse({"type": "message", "content": final})
                messages.append({"role": "assistant", "content": final})
                yield _sse({"type": "done"})
                return

        yield _sse({"type": "error", "content": "Max tool-calling turns reached"})

    async def _execute(self, name: str, args: dict) -> str:
        if name == "search_knowledge":
            if not self._rag_ok:
                return "Knowledge base is not available. Start local-rag first."
            results = self.rag.search(args.get("query", ""))
            return self.rag.format_results(results)
        elif name == "chat_knowledge":
            if not self._rag_ok:
                return "Knowledge base is not available."
            return self.rag.chat(args.get("query", ""))
        elif name in self._tool_names:
            return await self.toolhub.call(name, args)
        return f"Error: unknown tool '{name}'"


_agent: AgentService | None = None


@app.on_event("startup")
async def startup():
    global _agent
    _agent = AgentService()
    # Check RAG immediately
    try:
        _agent._rag_ok = _agent.rag.health()
    except Exception:
        _agent._rag_ok = False
    # Connect ToolHub in background task with timeout
    async def _connect_th():
        try:
            await _agent.toolhub.connect()
            _agent._toolhub_ok = True
            _agent._tool_names = _agent.toolhub.tool_names
        except Exception:
            _agent._toolhub_ok = False
    import asyncio
    task = asyncio.create_task(_connect_th())
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=8.0)
    except (TimeoutError, asyncio.TimeoutError):
        _agent._toolhub_ok = False
        print("ToolHub connection timed out")
    except Exception as e:
        _agent._toolhub_ok = False
        print(f"ToolHub: {e}")
    print(f"rag={_agent._rag_ok}  toolhub={_agent._toolhub_ok}  tools={_agent._tool_names}")


@app.on_event("shutdown")
async def shutdown():
    global _agent
    if _agent:
        await _agent.disconnect()


@app.get("/api/status")
async def status():
    return {
        "gateway": config.BASE_URL,
        "model_local": config.MODEL_LOCAL,
        "model_cloud": config.MODEL_CLOUD,
        "llm_model": config.LLM_MODEL,
        "rag_available": _agent._rag_ok if _agent else False,
        "toolhub_available": _agent._toolhub_ok if _agent else False,
        "toolhub_tools": _agent._tool_names if _agent else [],
    }


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    user_message = body.get("message", "")
    history = body.get("history", [])

    async def event_stream():
        async for event in _agent.chat_stream(user_message, history):
            yield event

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ============================================================
# File Browser
# ============================================================

def _resolve_file_path(input_path: str) -> Path:
    allowed = [Path(r).resolve() for r in config.TOOLHUB_ROOT.split(os.pathsep)]
    path = Path(input_path)
    if path.is_absolute():
        resolved = path.resolve()
        for root in allowed:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue
        raise ValueError(f"Access denied: '{input_path}' is outside allowed directories.")
    first = None
    for root in allowed:
        candidate = (root / path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if first is None:
            first = candidate
        if candidate.exists():
            return candidate
    if first is not None:
        return first
    raise ValueError(f"Access denied: '{input_path}' resolves outside allowed directories.")


@app.get("/api/files")
async def list_files(path: str = ""):
    try:
        if not path:
            roots = [r for r in config.TOOLHUB_ROOT.split(os.pathsep) if os.path.isdir(r)]
            return {"ok": True, "entries": [
                {"name": r, "type": "dir", "path": r, "size": None, "modified": None}
                for r in roots
            ], "path": ""}
        target = _resolve_file_path(path)
        if not target.is_dir():
            raise HTTPException(400, f"Not a directory: {path}")
        entries = []
        for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            st = entry.stat()
            entries.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "path": str(entry),
                "size": st.st_size if entry.is_file() else None,
                "modified": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(),
            })
        return {"ok": True, "entries": entries, "path": str(target)}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/files/read")
async def read_file_endpoint(path: str):
    try:
        target = _resolve_file_path(path)
        if not target.is_file():
            raise HTTPException(400, f"Not a file: {path}")
        content = target.read_text(encoding="utf-8", errors="replace")
        return {"ok": True, "content": content, "path": str(target)}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


# ============================================================
# Document Management (proxy to local-rag)
# ============================================================

@app.get("/api/documents")
async def list_documents():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{config.RAG_URL}/api/documents")
            return r.json()
    except Exception as e:
        raise HTTPException(502, f"Cannot reach local-rag: {e}")


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{config.RAG_URL}/api/documents/upload",
                files={"file": (file.filename, content, file.content_type or "application/octet-stream")},
            )
            return r.json()
    except Exception as e:
        raise HTTPException(502, f"Cannot reach local-rag: {e}")


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.delete(f"{config.RAG_URL}/api/documents/{doc_id}")
            if r.status_code == 404:
                raise HTTPException(404, f"Document {doc_id} not found")
            return r.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Cannot reach local-rag: {e}")


# ============================================================
# Model switching
# ============================================================

@app.post("/api/model")
async def set_model(request: Request):
    body = await request.json()
    model = body.get("model", "")
    valid = {"ollama-chat", "deepseek-flash"}
    if model not in valid:
        raise HTTPException(400, f"Invalid model. Choose: {', '.join(valid)}")
    config.LLM_MODEL = model
    return {"ok": True, "model": model}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101, log_level="info")
