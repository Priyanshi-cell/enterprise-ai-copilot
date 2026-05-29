import logging
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 15
MIN_LINE_LENGTH = 40
MIN_USEFUL_LINES = 5
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "DNT": "1",
    "Connection": "keep-alive",
}
TAGS_TO_REMOVE = [
    "script",   # JavaScript code
    "style",    # CSS code
    "nav",      # navigation menus
    "footer",   # page footer
    "header",   # page header
    "noscript", # fallback JS messages
    "svg",      # vector graphics
    "aside",    # sidebars
    "form",     # input forms
    "iframe",   # embedded frames
]
BLOCKED_STATUS_CODES = {401, 403, 406, 429}

def scrape_website(url: str) -> dict:
    """
    Scrape a single webpage and return its visible text content.

    Fetches the page, strips all noise tags, then extracts
    clean readable text line by line.

    Args:
        url: The full URL to scrape, e.g. "https://example.com"

    Returns:
        {
            "text": str,           # cleaned visible text from the page
            "pages_scraped": int,  # always 1 for single-page scraping
            "source": str,         # the URL that was scraped
        }

    Raises:
        Exception: With a human-readable message describing what went wrong.
                   The caller (main.py) catches this and returns it as JSON.
    """

    url = url.strip().rstrip("/")
    logger.info("Scraping URL: %s", url)

    
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

    except requests.exceptions.Timeout:
        logger.warning("Request timed out: %s", url)
        raise Exception(
            f"The website took too long to respond (timeout: {REQUEST_TIMEOUT}s). "
            "Try again or use a different URL."
        )

    except requests.exceptions.ConnectionError:
        logger.warning("Connection error: %s", url)
        raise Exception(
            f"Could not connect to {url}. "
            "Check the URL is correct and the site is reachable."
        )

    except requests.exceptions.TooManyRedirects:
        logger.warning("Too many redirects: %s", url)
        raise Exception(
            f"Too many redirects at {url}. "
            "The URL may be broken or redirecting in a loop."
        )

    except requests.exceptions.InvalidURL:
        logger.warning("Invalid URL: %s", url)
        raise Exception(
            f"'{url}' is not a valid URL. "
            "Make sure it starts with https:// or http://"
        )

    except Exception as e:
        logger.error("Unexpected fetch error for %s: %s", url, str(e))
        raise Exception(f"Failed to fetch the page: {str(e)}")
    
    if response.status_code in BLOCKED_STATUS_CODES:
        logger.warning("Site blocked request | status=%d | url=%s", response.status_code, url)
        raise Exception(
            f"The website refused the request (HTTP {response.status_code}). "
            "It may block automated access. Try uploading a PDF instead."
        )

    if response.status_code == 404:
        raise Exception(
            f"Page not found (HTTP 404): {url}. "
            "Check the URL is correct."
        )

    if response.status_code != 200:
        logger.warning("Non-200 response | status=%d | url=%s", response.status_code, url)
        raise Exception(
            f"The website returned an error (HTTP {response.status_code}). "
            "Try a different URL."
        )



    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        logger.warning("Non-HTML content type: %s | url=%s", content_type, url)
        raise Exception(
            f"The URL returned '{content_type}' instead of a webpage. "
            "Please provide a standard web page URL."
        )

    logger.info("Page fetched | status=200 | bytes=%d", len(response.content))

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        logger.error("HTML parsing failed: %s", str(e))
        raise Exception(f"Failed to parse the page HTML: {str(e)}")

    
    removed = 0
    for tag_name in TAGS_TO_REMOVE:
        for tag in soup.find_all(tag_name):
            tag.decompose()
            removed += 1

    logger.info("Removed %d noise tags", removed)

    content_area = (
        soup.find("main")
        or soup.find("article")
        or soup.find("body")
        or soup
    )

    raw_text = content_area.get_text(separator="\n")

    clean_lines = []
    seen_lines = set()  # deduplication 

    for line in raw_text.splitlines():
        line = line.strip()

        # Skip lines that are too short to be meaningful content
        if len(line) < MIN_LINE_LENGTH:
            continue

        # Skip duplicate lines (e.g. the same tagline in meta + body)
        if line in seen_lines:
            continue

        clean_lines.append(line)
        seen_lines.add(line)

    logger.info("Extracted %d clean lines", len(clean_lines))


    if len(clean_lines) < MIN_USEFUL_LINES:
        logger.warning(
            "Too few lines extracted (%d) from %s — site may use JavaScript rendering",
            len(clean_lines), url,
        )
        raise Exception(
            f"Only {len(clean_lines)} lines of text could be extracted from {url}. "
            "The site likely loads its content with JavaScript, which this scraper "
            "cannot execute. Try uploading a PDF of the page content instead."
        )

    combined_text = "\n".join(clean_lines)

    logger.info(
        "Scrape complete | url=%s | lines=%d | characters=%d",
        url, len(clean_lines), len(combined_text),
    )

    return {
        "text": combined_text,
        "pages_scraped": 1,
        "source": url,
    }