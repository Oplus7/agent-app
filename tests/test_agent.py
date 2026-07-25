"""Tests for the agent framework — mocks LLM to verify tool-calling loop."""

import asyncio
import json
import types
from unittest.mock import MagicMock, AsyncMock

import pytest

import config


class FakeChoice:
    class Message:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

        def model_dump(self):
            result = {"role": "assistant", "content": self.content}
            if self.tool_calls:
                result["tool_calls"] = self.tool_calls
            return result

    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, choices):
        self.choices = [FakeChoice(msg) for msg in choices]


def make_tool_call(name, args_dict):
    call = types.SimpleNamespace()
    call.id = f"call_{name}_1"
    call.function = types.SimpleNamespace()
    call.function.name = name
    call.function.arguments = json.dumps(args_dict)
    return call


@pytest.mark.asyncio
async def test_agent_responds_directly():
    """Agent should print LLM response when no tool needed."""
    from agent import Agent

    config.ALIBABA_API_KEY = "test-key"
    agent = Agent()

    agent.rag.health = lambda: False
    agent.toolhub.connect = AsyncMock()
    agent.toolhub._tools = []
    agent.toolhub._session = None

    fake_msg = FakeChoice.Message(content="Hello! I can help with that.")
    agent.llm.chat.completions.create = MagicMock(
        return_value=FakeResponse([fake_msg])
    )

    agent.messages = [{"role": "system", "content": "test"}]
    agent.messages.append({"role": "user", "content": "help"})

    await agent._think()

    assert len(agent.messages) == 3
    assert agent.messages[-1]["role"] == "assistant"
    assert "Hello" in agent.messages[-1]["content"]


@pytest.mark.asyncio
async def test_agent_uses_tools():
    """Agent should call tool, feed result back, then respond."""
    from agent import Agent

    config.ALIBABA_API_KEY = "test-key"
    agent = Agent()

    agent.rag.health = lambda: False
    agent.toolhub.connect = AsyncMock()
    agent.toolhub._tools = []
    agent.toolhub._session = None

    tool_call_msg = FakeChoice.Message(
        content=None,
        tool_calls=[make_tool_call("search_knowledge", {"query": "ML notes"})],
    )
    agent.rag.search = MagicMock(return_value=[
        {"content": "ML is cool", "source": "notes/ml.md", "score": 0.9}
    ])

    final_msg = FakeChoice.Message(content="Found ML notes: Machine Learning is cool.")

    agent.llm.chat.completions.create = MagicMock(side_effect=[
        FakeResponse([tool_call_msg]),
        FakeResponse([final_msg]),
    ])

    agent.messages = [{"role": "system", "content": "test"}]
    agent.messages.append({"role": "user", "content": "find ML notes"})

    await agent._think()

    assert agent.rag.search.called
    assert len(agent.messages) == 5
    assert agent.messages[-1]["role"] == "assistant"


def test_rag_format_results():
    from rag_client import RAGClient

    c = RAGClient()
    results = [
        {"content": "First result", "source": "doc1.md", "score": 0.95},
        {"content": "Second result", "source": "doc2.md", "score": 0.78},
    ]
    formatted = c.format_results(results)
    assert "doc1.md" in formatted
    assert "doc2.md" in formatted
    assert "0.95" in formatted
    assert "First result" in formatted


def test_rag_format_empty():
    from rag_client import RAGClient
    c = RAGClient()
    assert "No relevant" in c.format_results([])
    assert "error" in c.format_results([{"error": "failed"}])
