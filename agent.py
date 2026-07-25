"""
Personal AI Agent — CLI tool-calling agent powered by qwen-plus.

Connects to local-rag (HTTP) + mcp-toolhub (MCP) for file & knowledge ops.

Usage:
    set ALIBABA_API_KEY=sk-...
    python agent.py
"""

import asyncio
import json
import sys

from openai import OpenAI

import config
from rag_client import RAGClient
from mcp_client import ToolHubClient

SYSTEM_PROMPT = """You are a personal AI agent running on the user's local machine.

You have access to two sets of tools:

1. KNOWLEDGE BASE:
   - search_knowledge: Search the user's personal knowledge base. Use for finding notes, documents, or any stored information.
   - chat_knowledge: Get an AI answer augmented with the knowledge base. Use for complex questions.

2. FILE SYSTEM (mcp-toolhub):
   - Filesystem tools for reading, writing, searching files.

How to work:
- When the user asks to find or recall information, check the knowledge base first.
- Synthesize results into a clear, concise answer. Show sources when available.
- Use filesystem tools to save summaries, read configuration, or organize files.
- Be helpful and direct."""

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
            "description": "Ask the knowledge base a question. The system retrieves relevant documents and generates an answer. Best for questions requiring synthesis.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Full question"}},
                "required": ["query"],
            },
        },
    },
]


class Agent:
    def __init__(self):
        if not config.ALIBABA_API_KEY:
            print("ERROR: ALIBABA_API_KEY not set.")
            print('       set ALIBABA_API_KEY=sk-...')
            sys.exit(1)

        self.llm = OpenAI(api_key=config.ALIBABA_API_KEY, base_url=config.BASE_URL)
        self.rag = RAGClient(config.RAG_URL)
        self.toolhub = ToolHubClient(config.TOOLHUB_SERVER, config.TOOLHUB_ROOT)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def start(self):
        print("Initializing...")
        print(f"  LLM: {config.LLM_MODEL} @ {config.BASE_URL}")
        if self.rag.health():
            print("  local-rag: connected")
        else:
            print("  local-rag: NOT AVAILABLE (RAG tools disabled)")
            self._no_rag = True

        try:
            await self.toolhub.connect()
            print(f"  toolhub: {len(self.toolhub.tool_names)} tools ({', '.join(self.toolhub.tool_names)})")
        except Exception as e:
            print(f"  toolhub: NOT AVAILABLE ({e})")

        print("\nReady. Type a query (or /exit, /clear)\n")

        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("/exit", "/quit", "/q"):
                break
            if user_input.lower() in ("/clear", "/c"):
                self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("Context cleared.")
                continue
            if user_input.lower().startswith("/"):
                print("Commands: /exit, /clear")
                continue

            self.messages.append({"role": "user", "content": user_input})
            await self._think()

        await self.toolhub.disconnect()

    async def _think(self):
        tools = RAG_TOOLS + self.toolhub.tool_schemas

        for _ in range(8):
            response = self.llm.chat.completions.create(
                model=config.LLM_MODEL,
                messages=self.messages,
                tools=tools,
                temperature=0.3,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                self.messages.append(msg.model_dump())

                for call in msg.tool_calls:
                    name = call.function.name
                    try:
                        args = json.loads(call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    print(f"  -> {name}({json.dumps(args, ensure_ascii=False)})")
                    result = await self._execute(name, args)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    })
            else:
                if msg.content:
                    print(f"\n{msg.content}\n")
                self.messages.append({"role": "assistant", "content": msg.content or ""})
                return

        print("(reached max tool-calling turns)")

    async def _execute(self, name: str, args: dict) -> str:
        if name == "search_knowledge":
            results = self.rag.search(args.get("query", ""))
            return self.rag.format_results(results)
        elif name == "chat_knowledge":
            return self.rag.chat(args.get("query", ""))
        elif name in self.toolhub.tool_names:
            return await self.toolhub.call(name, args)
        return f"Error: unknown tool '{name}'"


async def main():
    agent = Agent()
    await agent.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
