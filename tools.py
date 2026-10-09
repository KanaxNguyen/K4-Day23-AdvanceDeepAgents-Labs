"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

_LAST_ARXIV_CALL = 0.0


def _redact_key(msg: str) -> str:
    key = (os.getenv("EXA_API_KEY") or "").strip()
    if key and key in msg:
        msg = msg.replace(key, "[REDACTED]")
    return msg


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    Backoff: exponential base * 2**attempt with random jitter, capped at `cap` seconds.
    Respects retry_after when provided. Does not sleep after the last attempt.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as e:
            if attempt == attempts - 1:
                raise
            retry_after = getattr(e, "retry_after", None)
            if retry_after is not None and retry_after > 0:
                delay = min(float(retry_after), cap)
            else:
                delay = min(base * (2**attempt) + random.uniform(0, 1), cap)
            time.sleep(delay)


def _handle_httpx_error(exc: Exception):
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (429, 500, 502, 503, 504):
            retry_after = None
            ra_hdr = exc.response.headers.get("Retry-After")
            if ra_hdr:
                try:
                    retry_after = float(ra_hdr)
                except ValueError:
                    pass
            raise RetryableError(f"HTTP {status}", retry_after=retry_after) from exc
        raise exc
    if isinstance(exc, httpx.TransportError):
        raise RetryableError(f"Transport error: {exc}") from exc
    raise exc


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _LAST_ARXIV_CALL
    try:
        tokens = re.findall(r"[\w-]+", query)
        if not tokens:
            return "NO RESULTS"

        elapsed = time.monotonic() - _LAST_ARXIV_CALL
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        _LAST_ARXIV_CALL = time.monotonic()

        clamped_max = max(1, min(max_results, 30))
        search_query = " AND ".join(f"all:{t}" for t in tokens)
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": clamped_max,
        }

        def _fetch():
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(ARXIV_URL, params=params)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        ra_hdr = resp.headers.get("Retry-After")
                        retry_after = float(ra_hdr) if ra_hdr and ra_hdr.isdigit() else None
                        raise RetryableError(f"HTTP {resp.status_code}", retry_after=retry_after)
                    resp.raise_for_status()
                    return resp.text
            except Exception as exc:
                _handle_httpx_error(exc)

        xml_text = with_retry(_fetch, attempts=5, base=2.0, cap=60.0)
        root = xml.etree.ElementTree.fromstring(xml_text)
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        if not entries:
            entries = [el for el in root if el.tag.endswith("entry")]

        records = []
        for entry in entries:
            id_el = entry.find("{http://www.w3.org/2005/Atom}id")
            if id_el is None:
                id_el = next((c for c in entry if c.tag.endswith("id")), None)
            if id_el is None or not id_el.text:
                continue

            raw_id = id_el.text.strip().split("/abs/")[-1]
            paper_id = re.sub(r"v\d+$", "", raw_id)
            url = f"https://arxiv.org/abs/{paper_id}"

            pub_el = entry.find("{http://www.w3.org/2005/Atom}published")
            if pub_el is None:
                pub_el = next((c for c in entry if c.tag.endswith("published")), None)
            published = pub_el.text.strip()[:10] if pub_el is not None and pub_el.text else ""

            title_el = entry.find("{http://www.w3.org/2005/Atom}title")
            if title_el is None:
                title_el = next((c for c in entry if c.tag.endswith("title")), None)
            title = " ".join(title_el.text.split()) if title_el is not None and title_el.text else ""

            summary_el = entry.find("{http://www.w3.org/2005/Atom}summary")
            if summary_el is None:
                summary_el = next((c for c in entry if c.tag.endswith("summary")), None)
            summary = " ".join(summary_el.text.split())[:600] if summary_el is not None and summary_el.text else ""

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _redact_key(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        clamped_limit = max(1, min(limit, 100))
        params = {"limit": clamped_limit}
        if date:
            params["date"] = date

        def _fetch():
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(HF_DAILY_URL, params=params)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        ra_hdr = resp.headers.get("Retry-After")
                        retry_after = float(ra_hdr) if ra_hdr and ra_hdr.isdigit() else None
                        raise RetryableError(f"HTTP {resp.status_code}", retry_after=retry_after)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as exc:
                _handle_httpx_error(exc)

        data = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list):
            return "NO RESULTS"

        records = []
        for item in data:
            paper = item.get("paper") if isinstance(item, dict) else None
            if not isinstance(paper, dict):
                continue
            paper_id = paper.get("id")
            if not paper_id:
                continue

            title = " ".join(str(paper.get("title") or item.get("title") or "").split())
            summary = " ".join(str(paper.get("summary") or item.get("summary") or "").split())[:600]
            record = {
                "id": str(paper_id),
                "url": f"https://huggingface.co/papers/{paper_id}",
                "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
                "title": title,
                "summary": summary,
                "upvotes": int(paper.get("upvotes") or item.get("upvotes") or 0),
                "github": str(paper.get("githubRepo") or item.get("githubRepo") or ""),
                "stars": int(paper.get("githubStars") or item.get("githubStars") or 0),
            }
            if keyword:
                if keyword.lower() not in (record["title"] + " " + record["summary"]).lower():
                    continue
            records.append(record)

        records.sort(key=lambda x: x["upvotes"], reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _redact_key(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        clamped_limit = max(1, min(limit, 50))
        params = {"q": query, "limit": clamped_limit}

        def _fetch():
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(HF_SEARCH_URL, params=params)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        ra_hdr = resp.headers.get("Retry-After")
                        retry_after = float(ra_hdr) if ra_hdr and ra_hdr.isdigit() else None
                        raise RetryableError(f"HTTP {resp.status_code}", retry_after=retry_after)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as exc:
                _handle_httpx_error(exc)

        data = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list):
            return "NO RESULTS"

        records = []
        for item in data:
            paper = item.get("paper") if isinstance(item, dict) else None
            if not isinstance(paper, dict):
                paper = item
            paper_id = paper.get("id")
            if not paper_id:
                continue

            raw_summary = paper.get("ai_summary") or item.get("ai_summary") or paper.get("summary") or item.get("summary") or ""
            title = " ".join(str(paper.get("title") or item.get("title") or "").split())
            summary = " ".join(str(raw_summary).split())[:600]
            records.append({
                "id": str(paper_id),
                "url": f"https://huggingface.co/papers/{paper_id}",
                "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
                "title": title,
                "summary": summary,
                "upvotes": int(paper.get("upvotes") or item.get("upvotes") or 0),
                "github": str(paper.get("githubRepo") or item.get("githubRepo") or ""),
                "stars": int(paper.get("githubStars") or item.get("githubStars") or 0),
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _redact_key(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa_mcp(name: str, arguments: dict) -> str:
    key = (os.getenv("EXA_API_KEY") or "").strip()
    url = f"{EXA_URL}?exaApiKey={key}" if key else EXA_URL
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments,
        },
    }

    def _post():
        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(
                    url,
                    json=body,
                    headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                )
        except Exception as exc:
            _handle_httpx_error(exc)

        if resp.status_code in (429, 500, 502, 503, 504):
            ra_hdr = resp.headers.get("Retry-After")
            retry_after = float(ra_hdr) if ra_hdr and ra_hdr.isdigit() else None
            raise RetryableError(f"HTTP {resp.status_code}", retry_after=retry_after)
        resp.raise_for_status()

        data = None
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    try:
                        data = json.loads(payload)
                        break
                    except json.JSONDecodeError:
                        pass
        if data is None:
            try:
                data = resp.json()
            except Exception as e:
                raise RuntimeError(f"Cannot parse response from Exa: {resp.text[:200]}") from e

        if "error" in data:
            raise RetryableError(f"Exa RPC error: {data['error']}")

        result = data.get("result", {})
        meta = result.get("_meta", {})
        is_rate_limited = False
        meta_str = str(meta).lower()
        if "ratelimit" in meta_str or "rate limit" in meta_str or "rate_limit" in meta_str:
            is_rate_limited = True

        contents = result.get("content", [])
        texts = []
        for c in contents:
            if isinstance(c, dict) and c.get("type") == "text":
                txt = c.get("text", "")
                if "rate limit" in txt.lower() or "too many requests" in txt.lower():
                    is_rate_limited = True
                texts.append(txt)

        if is_rate_limited:
            raise RetryableError("Exa rate limit encountered", retry_after=20.0)

        out_text = "\n".join(texts).strip()
        return out_text

    return with_retry(_post, attempts=5, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        if not objective:
            objective = f"find research papers, articles, and surveys on {query}"
        clamped_num = max(1, min(num_results, 20))
        arguments = {
            "query": query,
            "objective": objective,
            "numResults": clamped_num,
        }
        text = _call_exa_mcp("web_search_exa", arguments)
        if not text:
            return "NO RESULTS"
        return text
    except Exception as exc:
        return _redact_key(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        arguments = {"urls": [url]}
        text = _call_exa_mcp("web_fetch_exa", arguments)
        if not text:
            return "NO RESULTS"
        return text[:12000]
    except Exception as exc:
        return _redact_key(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
