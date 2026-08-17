# Agent App — 个人 AI Agent

基于 LLM 工具调用（Function Calling）的个人 AI Agent，编排本地 RAG 知识库与文件系统，实现「知识检索 + 文件操作」的自主决策与执行。支持本地/云端模型隐私路由。

## 功能

- **工具编排**：LLM 自主决定调用知识库检索还是文件操作，多轮循环直到给出答案
- **RAG 集成**：连接 local-rag，语义搜索知识库 + 问答式检索
- **MCP 文件工具**：通过 MCP 协议连接 ToolHub，读写/搜索/浏览文件系统
- **隐私路由**：`privacy_routes.json` 白名单决定数据走本地 Ollama 还是云端 DeepSeek
- **流式输出**：SSE 实时推送思考过程、工具调用、最终答案
- **Web UI**：聊天界面 + 文件浏览器 + 文档管理 + 模型切换

## 架构

```
浏览器 (http://localhost:8101)
   │
   ▼
Agent App (FastAPI + SSE)
   │  ├─ OpenAI client ──▶ LiteLLM Gateway (http://localhost:4000)
   │  │                      ├─ ollama-chat    → 本地 Ollama (qwen3:4b)
   │  │                      └─ deepseek-flash → 云端 DeepSeek
   │  ├─ RAGClient (HTTP) ──▶ local-rag (http://localhost:8100)
   │  └─ ToolHubClient (MCP/stdio) ──▶ mcp-toolhub (11 文件工具)
```

### 依赖服务

| 服务 | 端口 | 说明 |
|------|------|------|
| LiteLLM Gateway | 4000 | 统一模型网关，路由本地/云端 |
| local-rag | 8100 | 知识库后端 |
| Ollama | 11434 | 本地 LLM |
| mcp-toolhub | stdio | 文件系统 MCP Server |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install litellm[proxy] fastapi uvicorn
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入：

```
DEEPSEEK_API_KEY=sk-xxx          # 云端模型密钥（可选，仅云路由需要）
LLM_BASE_URL=http://localhost:4000/v1
```

### 3. 启动 LiteLLM 网关

```bash
litellm --config litellm_config.yaml --port 4000
```

### 4. 启动 Agent App

```bash
python server.py
```

浏览器打开 `http://localhost:8101`。

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/status` | 服务状态（模型/网关/工具可用性） |
| POST | `/api/chat` | SSE 聊天流（工具编排） |
| GET | `/api/files` | 浏览文件（安全沙箱内） |
| GET | `/api/files/read` | 读取文件 |
| GET | `/api/documents` | 文档列表（代理 local-rag） |
| POST | `/api/documents/upload` | 上传文档（代理 local-rag） |
| DELETE | `/api/documents/{id}` | 删除文档（代理 local-rag） |
| POST | `/api/model` | 切换模型（ollama-chat / deepseek-flash） |

## 隐私路由

`privacy_routes.json` 定义哪些数据源允许走云端模型：

```json
{
  "default_route": "ollama-chat",
  "whitelist": {
    "allowed_sources": ["D:\\AI_Control\\docs\\public"],
    "allowed_urls": [],
    "allowed_collections": []
  }
}
```

默认所有数据走本地 Ollama，仅白名单内的路径允许调用云端 DeepSeek。

## 测试

```bash
python -m pytest tests/ -v
```

## License

MIT
