"""
Agent App Server — FastAPI wrapper with SSE streaming + web UI.

Usage:
    set ALIBABA_API_KEY=sk-...
    python server.py

Opens http://localhost:8101 with chat interface.
"""

import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI, Request
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
        if not config.ALIBABA_API_KEY:
            yield _sse({"type": "error", "content": "ALIBABA_API_KEY not configured"})
            return

        llm = OpenAI(api_key=config.ALIBABA_API_KEY, base_url=config.BASE_URL)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_message},
        ]

        for turn in range(8):
            response = llm.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                tools=self.tools,
                temperature=0.3,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(msg.model_dump())

                for call in msg.tool_calls:
                    name = call.function.name
                    try:
                        args = json.loads(call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    yield _sse({"type": "tool_call", "tool": name, "args": args})

                    result = await self._execute(name, args)
                    yield _sse({"type": "tool_result", "tool": name, "content": result})

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    })
            else:
                content = msg.content or ""
                messages.append({"role": "assistant", "content": content})
                yield _sse({"type": "message", "content": content})
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
    await _agent.connect()
    print(f"rag={_agent._rag_ok}  toolhub={_agent._toolhub_ok}  tools={_agent._tool_names}")


@app.on_event("shutdown")
async def shutdown():
    global _agent
    if _agent:
        await _agent.disconnect()


@app.get("/api/status")
async def status():
    return {
        "api_key_set": bool(config.ALIBABA_API_KEY),
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101, log_level="info")
