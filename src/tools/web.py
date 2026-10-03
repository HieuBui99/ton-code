from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup
from markdownify import ATX, markdownify
from pydantic_ai import ModelRetry, RunContext

from src.agent.deps import AgentDeps
from src.config.settings import settings

logger = logging.getLogger(__name__)

WEB_FETCH_TOOL_NAME = "web_fetch"


_MAX_RESPONSE_BYTES = 2_000_000

_DROP_TAGS = ("script", "style")

_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")

_TEXT_CONTENT_PREFIXES = ("text/",)
_TEXT_CONTENT_TYPES = ("application/json", "application/xml")

_TRANSPORT: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None


async def web_fetch(ctx: RunContext[AgentDeps], url: str) -> str:
    """GET ``url`` and return its content as model-readable text
    """
    target = url.strip()
    _validate_url(target)

    response = await _get(target)
    body = _decode_body(response)
    rendered = _render_body(response, body)
    logger.debug(
        "web_fetch ok (status=%d, final_url=%s, %d chars)",
        response.status_code,
        response.url,
        len(rendered),
    )
    return f"HTTP {response.status_code} {response.url}\n\n{rendered}"


def _validate_url(url: str) -> None:
    """Reject a non-http(s) URL with a ModelRetry *before* any connection is attempted.

    httpx mangles relative URLs deep inside redirect/cookie handling, so a model mistake is
    caught up front with a clear message instead of an opaque parse error.
    """
    if not url:
        raise ModelRetry("url is empty; provide an http(s) URL to fetch.")
    scheme = httpx.URL(url).scheme if _parseable(url) else ""
    if scheme not in ("http", "https"):
        raise ModelRetry(
            f"{url!r} is not an http(s) URL; provide an absolute http:// or https:// URL."
        )


def _parseable(url: str) -> bool:
    """Whether ``url`` parses as an :class:`httpx.URL` at all (a malformed URL does not)."""
    try:
        httpx.URL(url)
    except httpx.InvalidURL:
        return False
    return True


async def _get(url: str) -> httpx.Response:
    """Perform the GET, mapping every transport/HTTP failure to a model-readable ModelRetry."""
    try:
        async with _client() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response
    except httpx.TimeoutException as exc:
        logger.debug("web_fetch timed out (url=%r): %s", url, exc)
        raise ModelRetry(
            f"Fetching {url} timed out after {settings.web_fetch_timeout_s:g}s."
        ) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        logger.debug("web_fetch got HTTP %d (url=%r)", status, url)
        raise ModelRetry(
            f"Fetching {url} returned HTTP {status}; the page is not retrievable."
        ) from exc
    except httpx.RequestError as exc:
        logger.debug("web_fetch connection error (url=%r): %s", url, exc)
        raise ModelRetry(f"Could not connect to {url}: {exc}.") from exc


def _client() -> httpx.AsyncClient:
    """Build the per-call :class:`httpx.AsyncClient` (settings timeout, redirects, the test seam)."""
    return httpx.AsyncClient(
        timeout=settings.web_fetch_timeout_s,
        follow_redirects=True,
        transport=_TRANSPORT,  # type: ignore[arg-type]
    )


def _decode_body(response: httpx.Response) -> str:
    """Decode the response to text, refusing non-text content with a ModelRetry naming the type."""
    content_type = _content_type(response)
    if not _is_textual(content_type):
        logger.debug("web_fetch refused non-text content (%s, url=%s)", content_type, response.url)
        raise ModelRetry(
            f"{response.url} returned non-text content ({content_type or 'unknown'}); "
            "web_fetch only reads text and HTML pages."
        )
    return response.text


def _render_body(response: httpx.Response, body: str) -> str:
    """Cap, then (for HTML) Markdown-convert the decoded body into the model-facing text.

    Capped at :data:`_MAX_RESPONSE_BYTES` *first* so conversion never chews through an unbounded
    document; a truncation notice is appended when the body was clipped.
    """
    capped, truncated = _cap(body)
    text = _html_to_markdown(capped, response.url) if _is_html(_content_type(response)) else capped
    if truncated:
        text += f"\n\n[response truncated to {_MAX_RESPONSE_BYTES} bytes; the page was clipped]"
    return text


def _cap(body: str) -> tuple[str, bool]:
    """Hard-cap ``body`` to :data:`_MAX_RESPONSE_BYTES` UTF-8 bytes; return ``(text, truncated)``.

    A plain byte cap (no line structure to preserve), backed off to the last valid UTF-8
    character boundary so a multi-byte character is never split.
    """
    encoded = body.encode("utf-8")
    if len(encoded) <= _MAX_RESPONSE_BYTES:
        return body, False
    # Decode the capped prefix, ignoring a partial trailing character at the cut point.
    head = encoded[:_MAX_RESPONSE_BYTES].decode("utf-8", errors="ignore")
    return head, True


def _html_to_markdown(html: str, url: httpx.URL) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(_DROP_TAGS):
            tag.decompose()
        return markdownify(str(soup), heading_style=ATX).strip()
    except RecursionError as exc:
        logger.debug("web_fetch hit recursion limit converting HTML (url=%s)", url)
        raise ModelRetry(
            f"Could not parse the HTML at {url} (too deeply nested); try a different page."
        ) from exc
    except Exception as exc:
        logger.debug("web_fetch failed converting HTML (url=%s): %s", url, exc)
        raise ModelRetry(
            f"Could not parse the HTML at {url} (malformed); try a different page."
        ) from exc


def _content_type(response: httpx.Response) -> str:
    """The lower-cased media type from the ``Content-Type`` header (parameters dropped)."""
    raw = response.headers.get("content-type", "")
    return raw.split(";", 1)[0].strip().lower()


def _is_html(content_type: str) -> bool:
    """Whether ``content_type`` denotes HTML (→ convert to Markdown)."""
    return content_type in _HTML_CONTENT_TYPES


def _is_textual(content_type: str) -> bool:
    """Whether ``content_type`` is something the model can read (HTML or any other text)."""
    if _is_html(content_type):
        return True
    if content_type in _TEXT_CONTENT_TYPES:
        return True
    return content_type.startswith(_TEXT_CONTENT_PREFIXES)
