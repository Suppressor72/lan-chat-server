# LAN Chat Server

A lightweight chat UI for any OpenAI-compatible LLM API (vLLM, Ollama, etc.) with optional built-in web search tool use. The core server uses only the Python standard library; web search additionally requires `ddgs`.

![Architecture](https://img.shields.io/badge/python-3.10+-blue) ![Dependencies](https://img.shields.io/badge/core-stdlib_only-success) ![Web Search](https://img.shields.io/badge/web_search-ddgs_optional-informational)

## Features

- **Single-file server** (`server.py`) — serves the chat UI, proxies LLM API calls, handles web search tool use, and detects client disconnects during streaming
- **Optional autonomous web search** — when enabled, the model decides when to search, fetches full page content (not just snippets), and synthesizes answers with citations
- **Server-Sent Events streaming** — responses stream to the browser as they're generated
- **Multi-turn conversations** — full conversation history saved in `localStorage`, survives page reloads
- **Reasoning token streaming** — live thinking/reasoning tokens render in a collapsible block during generation, auto-collapse when content begins, persisted with conversation history
- **Per-phase token stats** — thinking t/s, content t/s, and total t/s displayed below each assistant message, using vLLM's `usage` chunk for accurate token counts (not chunk counting)
- **Model selection dropdown** — populated from the swap proxy `/v1/models` endpoint; switch models on the fly with localStorage persistence. Supports hot-swap backends via systemd
- **Live health indicators** — three status dots (Server / Proxy / vLLM) poll a `/health` endpoint every 10 seconds and show the currently loaded model name
- **Stop generation** — `AbortController` cancels the fetch mid-stream; server detects client disconnect via socket probe and stops generating. Partial content is saved
- **Thinking mode toggle** — enable/disable model reasoning output
- **Conversation sidebar** — create, switch between, and delete conversations; auto-titled from first message
- **Markdown rendering** — links, bold, inline/block code with copy buttons, GFM tables (dark-themed, content-width)
- **System prompt customization** — gear icon opens settings for custom system prompt, date/time injection, and web search toggle
- **LAN accessible** — any browser on your network can connect
- **Stdlib-only core** — with web search disabled, the server runs on Python stdlib only (`http.server`, `urllib`, `socketserver`, etc.); `ddgs` is imported only when a search is actually requested

## How It Works

```
Browser (chat.html)
    │
    │ POST /chat (SSE stream)  ·  GET /models  ·  GET /health
    ▼
server.py ─── ddgs Search (optional) ─── Page Fetching (parallel)
    │
    │ POST /v1/chat/completions  ·  GET /v1/models
    ▼
Swap Proxy (:8001) ─── routes to correct backend, hot-swaps via systemd
    │
    ├──▶ vLLM Backend A (:8002)
    └──▶ vLLM Backend B (:8003)
```

The server acts as a middleman between the browser and your LLM. When web search is enabled and the model decides it needs current information, it emits a `[[search]]query[[/search]]` marker. The server uses `ddgs` to search, fetches the top 5 result pages in parallel, strips them to clean text, and feeds them back to the model as tool results. The model then synthesizes a cited answer from the actual page content.

Web search can be disabled per browser in Settings. When disabled, the server does not advertise the search tool to the model, does not call the search path, and never imports `ddgs`.

When used with a swap proxy (any OpenAI-compatible reverse proxy), the chat server's `/models` endpoint discovers available backends and the dropdown lets you switch between them. The swap proxy handles stopping the current backend and starting the requested one. The `/health` endpoint checks proxy reachability and reports which model is currently loaded.

## Quick Start

### 1. Configure

Edit the top of `server.py`:

```python
VLLM_BASE = "http://127.0.0.1:8001"   # Your LLM API endpoint
MODEL = "your-model-name"              # Model name your API serves
```

### 2. Run

```bash
python3 server.py
```

This requires no third-party Python packages when Web Search is disabled in the UI. To enable web search, install the optional dependency in the same Python environment used to run the server:

```bash
python3 -m pip install ddgs
```

### 3. Open in browser

```
http://localhost:8080          # Local
http://<your-lan-ip>:8080      # From another machine on your network
```

## Web Search Details

When enabled, the search pipeline is:

1. **`ddgs` multi-backend search** — no API key required; replaces direct DuckDuckGo HTML scraping, which is blocked by an anomaly/bot challenge
2. **Parallel page fetching** — top 5 result pages fetched concurrently (~2-3s total)
3. **HTML-to-text extraction** — strips scripts, styles, nav, and boilerplate; keeps content
4. **Truncation** — each page capped at 4,000 chars to fit model context window

Tunable parameters at the top of `server.py`:

```python
MAX_PAGES_TO_FETCH = 5       # How many result pages to read
MAX_PAGE_CHARS = 4000        # Max chars per page sent to model
MAX_SEARCH_ITERATIONS = 8    # Search rounds before forcing a final answer
PAGE_FETCH_TIMEOUT = 8       # Seconds per page fetch
```

## Requirements

- Python 3.10+
- Any OpenAI-compatible `/v1/chat/completions` endpoint (vLLM, Ollama, LM Studio, text-generation-webui, etc.)
- Optional: `ddgs` for web search. It is not imported or required when Web Search is disabled.

## Files

| File | Description |
|------|-------------|
| `server.py` | HTTP server, search engine, page fetcher, tool-use loop, health/proxy endpoints, disconnect detection |
| `chat.html` | Chat UI frontend (dark theme, reasoning UI, markdown, token stats, model selector, health indicators, sidebar) |

## License

MIT
