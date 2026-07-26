"""MCP stdio client for ToolHub filesystem tools."""

import sys

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


class ToolHubClient:
    def __init__(self, server_path: str, allowed_root: str):
        self._server = server_path
        self._root = allowed_root
        self._session: ClientSession | None = None
        self._tools: list[dict] = []
        self._stdio_ctx = None
        self._session_ctx = None

    async def connect(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=[self._server, self._root],
        )
        self._stdio_ctx = stdio_client(params)
        read, write = await self._stdio_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()
        response = await self._session.list_tools()
        for tool in response.tools:
            ps = tool.inputSchema if hasattr(tool, "inputSchema") else {}
            self._tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": ps,
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
        if self._session_ctx:
            await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None
            self._session = None
        if self._stdio_ctx:
            await self._stdio_ctx.__aexit__(None, None, None)
            self._stdio_ctx = None
