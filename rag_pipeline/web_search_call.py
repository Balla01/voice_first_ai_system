import os
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from groq import Groq

from constants import GROQ_MODEL

load_dotenv()

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; InsureAssistBot/1.0)"}
REQUEST_TIMEOUT = 3
MAX_BYTES = 300_000  # most useful text is in the first part of the HTML

# Shared across requests so repeated calls to the same host reuse the TCP/TLS
# connection instead of paying handshake cost every time.
_session = requests.Session()
_session.headers.update(HEADERS)


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo search -> list of {title, href, body}."""
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def scrape_page_text(url: str) -> str:
    """Fetch a URL and return its visible text, stripped of scripts/styles/nav."""
    resp = _session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
    resp.raise_for_status()
    raw = resp.raw.read(MAX_BYTES, decode_content=True)

    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _scrape_result(r: dict) -> dict | None:
    url = r.get("href")
    try:
        text = scrape_page_text(url)
    except Exception as e:
        print(f"skip {url}: {e}")
        return None
    return {"title": r.get("title", ""), "href": url, "text": text}


def search_and_scrape(query: str, max_results: int = 2) -> list[dict]:
    """Search the web and scrape each result page's text, in parallel.

    Returns [{title, href, text}], skipping pages that fail to fetch.
    """
    results = search_web(query, max_results=max_results)
    with ThreadPoolExecutor(max_workers=max(1, len(results))) as pool:
        scraped = list(pool.map(_scrape_result, results))
    return [s for s in scraped if s is not None]


def summarize_web_results(query: str, pages: list[dict]) -> str:
    """Summarize scraped web pages into a direct answer for `query` via Groq.

    Returns "" if there's nothing to summarize (no pages, or no Groq key) so
    callers can treat an empty string as "web search yielded nothing useful".
    """
    api_key = os.getenv("groq_api")
    if not api_key or not pages:
        return ""

    combined = "\n\n".join(
        f"[{p['title']}]({p['href']})\n{p['text'][:2000]}" for p in pages
    )
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the web page excerpts "
                    "given below. Be concise and factual. If the excerpts don't "
                    "contain the answer, say so plainly."
                ),
            },
            {"role": "user", "content": f"Question: {query}\n\nWeb pages:\n{combined[:6000]}"},
        ],
        temperature=0.3,
        max_completion_tokens=400,
        top_p=1,
        stream=False,
    )
    return completion.choices[0].message.content.strip()


def web_search_answer(query: str, max_results: int = 5) -> str:
    """Search the web, scrape results, and summarize into one final answer."""
    pages = search_and_scrape(query, max_results=max_results)
    return summarize_web_results(query, pages)


if __name__ == "__main__":
    pages = search_and_scrape("todays date", max_results=5)
    for p in pages:
        print(p["title"], p["href"])
        print(p["text"][:500])
        print("---")
