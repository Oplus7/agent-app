"""MCP stdio client for ToolHub filesystem tools.

Connects to mcp-toolhub server via subprocess stdio,
discovers tools dynamically, and provides call_tool().
"""

import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


class ToolHubClient:
    def __init__(self, server_path: str, allowed_root: str):
        self._server = server_path
        self._root = allowed_root
        self._session: ClientSession | None = None
        self._tools: list[dict] = []
        self._read = None
        self._write = None

    async def connect(self, timeout: float = 5.0):
        params = StdioServerParameters(
            command=sys.executable,
            args=[self._server, self._root],
        )
        transport = await asyncio.wait_for(
            stdio_client(params).__aenter__(), timeout=timeout
        )
        self._read, self._write = transport
        self._session = ClientSession(self._read, self._write)
        init_task = asyncio.create_task(self._session.initialize())
        await asyncio.wait_for(init_task, timeout=timeout)
        response = await asyncio.wait_for(self._session.list_tools(), timeout=timeout)
        for tool in response.tools:
            params = tool.inputSchema if hasattr(tool, "inputSchema") else {}
            self._tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": params,
                },
            })

    @property
    def tool_schemas(self) -> list[dict]:
        return list(self._tools)

    @property
    def tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self._tools]

    async def call(self, name: str, arguments: dict) -> str:
        if self._session is None:
            return "Error: ToolHub not connected"
        try:
            result = await self._session.call_tool(name, arguments)
            if hasattr(result, "content") and result.content:
                return "\n".join(
                    c.text for c in result.content if hasattr(c, "text")
                )
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    async def disconnect(self):
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
