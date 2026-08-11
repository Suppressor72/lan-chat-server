#!/usr/bin/env python3
"""Web server: serves the chat UI, proxies API requests to vLLM, and provides
web search tool-use capability. Serves on port 8080 — accessible from any
browser on the LAN."""
import http.server
import socket
import socketserver
import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import re
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

PORT = 8080
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
VLLM_BASE = "http://127.0.0.1:8001"
MODEL = "ThinkingCap-Qwen3.6-27B-FP8"

# ── Page-fetching config ──────────────────────────────────────────────────────
MAX_PAGES_TO_FETCH = 5       # how many result pages to read in full
MAX_PAGE_CHARS = 4000        # truncate each page to this many chars for the model
PAGE_FETCH_TIMEOUT = 8       # seconds per page fetch

# ── Tool-use system prompt ────────────────────────────────────────────────────
TOOL_SYSTEM_PROMPT = """You are a helpful AI assistant with web search capability.

## Web Search Tool
When you need current information, recent events, live data, or anything you cannot verify from your training knowledge, use web search. You decide when to search — be proactive about it.

To search the web, output your search query on a single line wrapped in this exact format:
[[search]]your search query here[[/search]]

After you output the search marker, the system will run the search and return results. Use those results to answer the user's question.

## When to search
- Current events, news, or anything time-sensitive
- Stock prices, sports scores, weather
- Technical documentation that may have changed
- Anything where your training data might be outdated
- When the user explicitly asks for current info

## When NOT to search
- General knowledge, history, math, coding help
- Creative writing, analysis of provided text
- Questions about your own capabilities
- Simple factual questions you're confident about

Search results will be returned after your search marker. Read them carefully and cite sources when relevant."""


def build_system_prompt(
    custom_prompt: str = "",
    include_datetime: bool = True,
    include_search: bool = True,
) -> str:
    """Build the system prompt from optional layers.

    Order: date/time → custom prompt → web-search tool instructions.
    Any layer can be omitted via the boolean flags / empty custom prompt.
    """
    parts = []

    if include_datetime:
        now = datetime.now()
        parts.append(
            f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')} "
            f"({now.tzname() or 'local time'})"
        )

    if custom_prompt and custom_prompt.strip():
        parts.append(custom_prompt.strip())

    if include_search:
        parts.append(TOOL_SYSTEM_PROMPT)
    elif not parts:
        # Minimal fallback so the model always gets a system message
        parts.append("You are a helpful AI assistant.")

    return "\n\n".join(parts)

# ── DuckDuckGo search (no API key) ────────────────────────────────────────────
def search_web(query: str, num_results: int = 5) -> list[dict]:
    """Search DuckDuckGo and return results as [{title, url, snippet}]."""
    results = []
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Parse DuckDuckGo HTML results.
        # DDG markup changes frequently — use a robust approach that handles
        # attributes in any order (rel="nofollow" may precede class=).
        # Result blocks: <a ... class="result__a" ... href="...">Title</a>
        #                 <a ... class="result__snippet" ...>Snippet</a>
        link_pattern = re.compile(
            r'<a\s+[^>]*class="result__a"[^>]*>(.*?)</a>', re.DOTALL
        )
        snippet_pattern = re.compile(
            r'<a\s+[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL
        )
        href_pattern = re.compile(r'href="([^"]*)"')

        # Find all result links, then grab the next snippet after each
        link_iter = list(link_pattern.finditer(html))
        snippet_iter = list(snippet_pattern.finditer(html))

        for i, link_match in enumerate(link_iter):
            link_html = link_match.group(0)
            title = re.sub(r"<[^>]+>", "", link_match.group(1)).strip()

            # Extract real URL from DDG redirect (href may contain uddg= param)
            href_match = href_pattern.search(link_html)
            real_url = ""
            if href_match:
                raw_href = href_match.group(1)
                uddg = re.search(r"uddg=([^&\"]*)", raw_href)
                if uddg:
                    real_url = urllib.parse.unquote(uddg.group(1))

            # Match snippet — the closest one after this link
            snippet = ""
            for sm in snippet_iter:
                if sm.start() > link_match.start():
                    snippet = re.sub(r"<[^>]+>", "", sm.group(1)).strip()
                    break

            results.append({"title": title, "url": real_url, "snippet": snippet})
            if len(results) >= num_results:
                break
    except Exception as e:
        results = [{"title": "Search error", "url": "", "snippet": str(e)}]

    return results if results else [{"title": "No results", "url": "", "snippet": f"No results found for: {query}"}]


def format_search_results(results: list[dict]) -> str:
    """Format search results into a readable string for the model."""
    lines = ["Web search results:"]
    for i, r in enumerate(results, 1):
        lines.append(f"  [{i}] {r['title']}")
        lines.append(f"      {r['url']}")
        lines.append(f"      {r['snippet']}")
    return "\n".join(lines)


# ── HTML-to-text conversion ───────────────────────────────────────────────────
class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML→text: extracts visible text, skips script/style/nav noise."""
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "nav", "footer", "aside"}
    _BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th"}

    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag in self._BLOCK_TAGS and self._parts:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self):
        raw = "".join(self._parts)
        # Collapse whitespace within lines, keep paragraph breaks
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.split("\n")]
        # Remove empty lines
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)


def html_to_text(html: str) -> str:
    """Convert HTML page to clean readable text."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


# ── Page fetching ─────────────────────────────────────────────────────────────
def fetch_page_content(url: str, timeout: int = PAGE_FETCH_TIMEOUT) -> str:
    """Fetch a URL and return cleaned text content. Returns empty string on failure."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            # Skip non-HTML (PDFs, images, binaries)
            if "text/html" not in content_type and "text/plain" not in content_type:
                return ""
            raw = resp.read(2_000_000)  # 2MB cap to avoid massive pages
            charset = resp.headers.get_content_charset() or "utf-8"
            html = raw.decode(charset, errors="replace")

        text = html_to_text(html)
        if len(text) > MAX_PAGE_CHARS:
            text = text[:MAX_PAGE_CHARS] + "\n...[truncated]"
        return text
    except Exception:
        return ""


def fetch_pages_parallel(results: list[dict], max_pages: int = MAX_PAGES_TO_FETCH) -> dict:
    """Fetch multiple pages concurrently. Returns {url: page_text}."""
    pages = {}
    to_fetch = [r for r in results[:max_pages] if r.get("url")]
    if not to_fetch:
        return pages

    with ThreadPoolExecutor(max_workers=len(to_fetch)) as pool:
        future_to_url = {
            pool.submit(fetch_page_content, r["url"]): r["url"]
            for r in to_fetch
        }
        for future in as_completed(future_to_url, timeout=PAGE_FETCH_TIMEOUT + 2):
            url = future_to_url[future]
            try:
                text = future.result(timeout=PAGE_FETCH_TIMEOUT + 2)
                if text:
                    pages[url] = text
            except Exception:
                pass
    return pages


def format_search_results_with_pages(results: list[dict], pages: dict) -> str:
    """Format snippets + full page content for the model."""
    lines = ["Web search results (with page content):"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n--- Result {i}: {r['title']} ---")
        lines.append(f"URL: {r['url']}")
        lines.append(f"Snippet: {r['snippet']}")
        page_text = pages.get(r["url"])
        if page_text:
            lines.append(f"Page content ({len(page_text)} chars):")
            lines.append(page_text)
        else:
            lines.append("Page content: [not available]")
    return "\n".join(lines)


# ── vLLM helpers ──────────────────────────────────────────────────────────────
def vllm_stream(messages: list[dict], thinking: bool = True, model: str | None = None):
    """Stream a completion from vLLM, yielding content/reasoning chunks.

    Yields:
        tuple(str, str|dict) — (chunk_type, data) where chunk_type is
        'content', 'reasoning', or 'usage' (data is dict for usage).
    """
    payload = json.dumps({
        "model": model or MODEL,
        "messages": messages,
        "max_tokens": 8192,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }).encode()
    req = urllib.request.Request(
        f"{VLLM_BASE}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                # Usage chunk (last chunk when include_usage=True)
                if "usage" in chunk and chunk.get("usage"):
                    yield ("usage", chunk["usage"])
                    continue
                delta = chunk["choices"][0].get("delta", {})
                content_piece = delta.get("content")
                reasoning_piece = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning_piece:
                    yield ("reasoning", reasoning_piece)
                if content_piece:
                    yield ("content", content_piece)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


# ── Chat handler with tool-use loop ──────────────────────────────────────────
# Hold back this many chars from the streaming edge — enough to detect a
# [[search]] or [[/search]] marker that arrives split across token chunks.
_HOLD_BACK = len("[[/search]]")  # 10


def run_chat(
    messages: list[dict],
    thinking: bool,
    callback,
    system_prompt: str = "",
    include_datetime: bool = True,
    include_search: bool = True,
    model: str | None = None,
):
    """Run the chat loop with optional tool use. Sends SSE events via callback(data_str).

    Tokens are streamed to the client in real-time. When include_search is True,
    a hold-back buffer prevents partial [[search]] markers from leaking; on
    detection, streaming pauses, the search runs, and results are fed back.
    """
    full_messages = [{
        "role": "system",
        "content": build_system_prompt(system_prompt, include_datetime, include_search),
    }] + messages

    # Token timing accumulators
    total_content_tokens = 0
    total_thinking_tokens = 0
    total_thinking_time = 0.0
    total_content_time = 0.0

    for iteration in range(5):
        content = ""
        reasoning = ""
        sent_len = 0
        search_detected = False
        started = False
        iter_content_tokens = 0
        iter_thinking_tokens = 0

        # Per-phase timing: starts on FIRST token (excludes TTFT/prefill)
        phase_start = None  # set on first token arrival
        last_phase = None

        for chunk_type, token in vllm_stream(full_messages, thinking, model=model):
            # Capture usage data (vLLM sends it in the final chunk)
            if chunk_type == "usage":
                u = token  # token is actually a dict here
                iter_content_tokens = u.get("completion_tokens", iter_content_tokens)
                # vLLM may report reasoning tokens separately in usage
                iter_thinking_tokens = u.get("reasoning_tokens", 0)
                if iter_thinking_tokens:
                    iter_content_tokens = max(0, iter_content_tokens - iter_thinking_tokens)
                continue

            # Start timer on first actual token (excludes prefill/TTFT)
            now = time.monotonic()
            if phase_start is None:
                phase_start = now
                last_phase = chunk_type

            # Track phase transitions for accurate per-phase timing
            if chunk_type != last_phase:
                if last_phase == "reasoning":
                    total_thinking_time += now - phase_start
                elif last_phase == "content":
                    total_content_time += now - phase_start
                phase_start = now
                last_phase = chunk_type

            if chunk_type == "reasoning":
                reasoning += token
                if not callback(json.dumps({"type": "reasoning", "text": token})):
                    return  # client disconnected — abort
                continue
            else:
                content += token

            # Check if client is still connected
            if not callback(""):
                return  # client disconnected — abort

            # Suppress leading newlines/whitespace (model often emits blank
            # lines before the actual answer, especially after tool-use rounds)
            if not started:
                stripped = content.lstrip()
                if stripped:
                    content = stripped  # discard leading whitespace
                else:
                    continue  # all whitespace so far — keep buffering

            # Search markers only honored when web search is enabled
            if include_search and "[[search]]" in content:
                search_detected = True
                started = True
                marker_idx = content.index("[[search]]")
                # Flush everything between what was already sent and the marker
                if marker_idx > sent_len:
                    chunk = content[sent_len:marker_idx]
                    callback(json.dumps({"type": "content", "text": chunk}))
                continue  # keep buffering but don't stream past marker

            if search_detected:
                continue  # buffering only — marker text is not real content

            # Stream content to client, holding back _HOLD_BACK chars only
            # when search is enabled (marker may be forming at the edge)
            hold = _HOLD_BACK if include_search else 0
            safe_end = max(sent_len, len(content) - hold)
            if safe_end > sent_len:
                callback(json.dumps({"type": "content", "text": content[sent_len:safe_end]}))
                sent_len = safe_end
                started = True  # real content has begun

        # Stream complete — process the accumulated response

        if include_search and search_detected:
            # Flush any held-back non-marker content before the marker
            marker_idx = content.index("[[search]]")
            if marker_idx > sent_len:
                callback(json.dumps({"type": "content", "text": content[sent_len:marker_idx]}))

            # Extract search query
            search_match = re.search(r"\[\[search\]\](.+?)(?:\[\[/search\]\])?$", content, re.DOTALL)
            if search_match:
                query = search_match.group(1).replace("[[/search]]", "").strip()

                callback(json.dumps({"type": "tool_call", "tool": "web_search", "query": query}))

                results = search_web(query)
                pages = fetch_pages_parallel(results)
                result_text = format_search_results_with_pages(results, pages)

                callback(json.dumps({"type": "tool_result", "tool": "web_search", "results": result_text, "pages_fetched": len(pages)}))

                response_clean = re.sub(r"\[\[/?search\]\]", "", content).strip()
                full_messages.append({"role": "assistant", "content": response_clean})
                full_messages.append({"role": "tool", "content": result_text})
                continue
            # Edge case: marker detected but regex failed — treat as content

        # No tool call — flush any remaining held-back content, then done
        content_clean = re.sub(r"\[\[/?search\]\]", "", content) if include_search else content
        if len(content_clean) > sent_len:
            callback(json.dumps({"type": "content", "text": content_clean[sent_len:]}))
        full_messages.append({"role": "assistant", "content": content_clean.strip()})

        # Close out the final phase timing
        now = time.monotonic()
        if phase_start is not None:
            if last_phase == "reasoning":
                total_thinking_time += now - phase_start
            elif last_phase == "content":
                total_content_time += now - phase_start

        # Accumulate token counts
        total_content_tokens += iter_content_tokens
        total_thinking_tokens += iter_thinking_tokens

        # Calculate per-phase rates using phase-specific denominators
        total_gen_time = total_thinking_time + total_content_time
        content_tps = total_content_tokens / total_content_time if total_content_time > 0 else 0
        thinking_tps = total_thinking_tokens / total_thinking_time if total_thinking_time > 0 else 0
        total_tps = (total_content_tokens + total_thinking_tokens) / total_gen_time if total_gen_time > 0 else 0
        callback(json.dumps({
            "type": "stats",
            "content_tokens": total_content_tokens,
            "thinking_tokens": total_thinking_tokens,
            "content_tps": round(content_tps, 1),
            "thinking_tps": round(thinking_tps, 1),
            "total_tps": round(total_tps, 1),
            "thinking_time_s": round(total_thinking_time, 1),
            "content_time_s": round(total_content_time, 1),
            "gen_time_s": round(total_gen_time, 1),
        }))
        callback(json.dumps({"type": "done"}))
        return

    # If we hit max iterations
    callback(json.dumps({"type": "content", "text": "\n[Search iterations exceeded maximum.]"}))
    callback(json.dumps({"type": "done"}))


# ── HTTP handlers ─────────────────────────────────────────────────────────────
class ChatHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.path = '/chat.html'
        elif self.path == '/models':
            self._proxy_get('/v1/models')
            return
        elif self.path == '/health':
            self._handle_health()
            return
        return super().do_GET()

    def do_POST(self):
        if self.path == '/chat':
            self._handle_chat()
        elif self.path == '/search':
            self._handle_search()
        elif self.path == '/refresh':
            self._proxy_post('/refresh')
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_health(self):
        """Check proxy and vLLM backend health, return JSON status."""
        result = {"server": True, "proxy": False, "vllm": False,
                  "loaded_model": None}
        # Check swap proxy
        try:
            req = urllib.request.Request(f"{VLLM_BASE}/v1/models")
            with urllib.request.urlopen(req, timeout=3) as resp:
                result["proxy"] = True
                data = json.loads(resp.read())
                # Find loaded model
                for m in data.get("data", []):
                    if m.get("owned_by") == "local":
                        result["vllm"] = True
                        result["loaded_model"] = m.get("id")
                        break
        except Exception:
            pass
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def _proxy_get(self, backend_path):
        """Proxy a GET request to vLLM swap proxy."""
        try:
            req = urllib.request.Request(f"{VLLM_BASE}{backend_path}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.send_response(resp.status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(resp.read())
        except Exception as e:
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _proxy_post(self, backend_path):
        """Proxy a POST request to vLLM swap proxy."""
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else b''
        try:
            req = urllib.request.Request(
                f"{VLLM_BASE}{backend_path}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                self.send_response(resp.status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(resp.read())
        except Exception as e:
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_search(self):
        """Standalone search endpoint for direct use."""
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        query = body.get("query", "")
        results = search_web(query)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(results).encode())

    def _handle_chat(self):
        """Handle chat with tool-use loop. Uses SSE for streaming events."""
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        messages = body.get("messages", [])
        thinking = body.get("thinking", True)
        system_prompt = body.get("system_prompt", "")
        include_datetime = body.get("include_datetime", True)
        include_search = body.get("include_search", True)
        model = body.get("model", None)

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.close_connection = True  # ensure socket closes after SSE response
        self.end_headers()

        client_connected = [True]  # mutable flag for the callback

        def callback(data):
            if data:  # skip empty writes (used as connection check)
                try:
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                except Exception:
                    client_connected[0] = False
            else:
                # Connection check: probe if the client socket is still open
                try:
                    # Non-blocking recv — if it returns empty bytes, the client
                    # has disconnected. If it raises BlockingIOError, the socket
                    # is still open (no data available to read).
                    sock = self.connection
                    sock.setblocking(False)
                    try:
                        data = sock.recv(1, socket.MSG_PEEK)
                        if not data:
                            client_connected[0] = False
                    except (BlockingIOError, InterruptedError):
                        pass  # socket is still open, no data to read
                    finally:
                        sock.setblocking(True)
                except Exception:
                    client_connected[0] = False
            return client_connected[0]

        run_chat(
            messages, thinking, callback, system_prompt,
            include_datetime=bool(include_datetime),
            include_search=bool(include_search),
            model=model,
        )


if __name__ == '__main__':
    # Threaded server — allows concurrent requests (page load while a
    # multi-round /chat is running, multiple users, etc.)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), ChatHandler) as httpd:
        httpd.daemon_threads = True
        print(f"Chat UI serving on http://0.0.0.0:{PORT} (threaded)")
        print(f"Access from LAN: http://<your-ip>:{PORT}")
        print(f"vLLM API at {VLLM_BASE}")
        print(f"Web search enabled (DuckDuckGo)")
        httpd.serve_forever()
