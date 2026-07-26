"""HTTP client for local-rag knowledge base."""

import httpx


class RAGClient:
    def __init__(self, base_url: str = "http://localhost:8100"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30)

    def health(self) -> bool:
        try:
            r = self._client.get(f"{self.base_url}/api/health")
            return r.status_code == 200
        except Exception:
            return False

    def search(self, query: str, collection: str = "rag_documents", top_k: int = 5) -> list[dict]:
        """Search knowledge base for top_k relevant chunks."""
        try:
            r = self._client.post(
                f"{self.base_url}/api/query",
                json={"query": query, "collection": collection, "top_k": top_k},
            )
            r.raise_for_status()
            data = r.json()
            sources = data.get("sources", [])
            return sources
        except Exception as e:
            return [{"error": str(e)}]

    def chat(self, query: str, collection: str = "rag_documents") -> str:
        """Chat with knowledge base — RAG-augmented answer."""
        try:
            r = self._client.post(
                f"{self.base_url}/api/chat?collection={collection}",
                json={"question": query, "k": 5},
            )
            r.raise_for_status()
            data = r.json()
            return data.get("answer", str(data))
        except Exception as e:
            return f"Error: {e}"

    def format_results(self, results: list[dict]) -> str:
        """Format search results for LLM consumption."""
        if not results:
            return "No relevant documents found."
        if results and "error" in results[0]:
            return f"Search error: {results[0]['error']}"
        lines = []
        for i, r in enumerate(results, 1):
            content = r.get("content", str(r)[:500])
            source = r.get("source_file", r.get("media_name", r.get("source", "unknown")))
            score = r.get("score", "?")
            lines.append(f"[{i}] ({source}, score={score})\n{content}")
        return "\n\n".join(lines)
