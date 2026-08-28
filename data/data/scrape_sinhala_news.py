#!/usr/bin/env python3.11
"""
Sinhala News Scraper
Collects 250+ Sinhala news articles from verified sources.
Output CSV columns: source_url, text_content
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "si,en;q=0.9",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

TARGET = 260
DELAY  = 0.8   # seconds between article requests

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_soup(url: str, timeout: int = 15) -> BeautifulSoup | None:
    try:
        r = SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return BeautifulSoup(r.text, "lxml")
    except Exception as exc:
        logger.warning(f"  x {url}  ->  {exc}")
        return None


def clean_text(el) -> str:
    for tag in el.select("script,style,figure,figcaption,.ad,.advertisement,iframe,noscript"):
        tag.decompose()
    return el.get_text(separator="\n", strip=True)


def first_content(soup: BeautifulSoup, selectors: list, min_len: int = 80) -> str:
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            t = clean_text(el)
            if len(t) >= min_len:
                return t
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 – Ada Derana Sinhala
# ─────────────────────────────────────────────────────────────────────────────

def scrape_adaderana(max_articles: int) -> list:
    records = []
    CONTENT_SELS = ["div.news-content", "div#news-content"]

    soup = get_soup("https://sinhala.adaderana.lk/news.php")
    if soup is None:
        return records

    latest_ids = sorted(
        int(a["href"].split("/")[-1])
        for a in soup.find_all("a", href=True)
        if "/news/" in a["href"] and a["href"].split("/")[-1].isdigit()
    )
    if not latest_ids:
        logger.warning("Ada Derana: could not find latest ID")
        return records

    top_id = max(latest_ids)
    logger.info(f"Ada Derana: latest ID = {top_id}")

    for nid in tqdm(range(top_id, top_id - 500, -1), desc="Ada Derana", unit="art"):
        if len(records) >= max_articles:
            break
        url = f"http://sinhala.adaderana.lk/news/{nid}"
        soup = get_soup(url)
        if soup is None:
            continue
        text = first_content(soup, CONTENT_SELS)
        if text:
            records.append({"source_url": url, "text_content": text})
        time.sleep(DELAY)

    logger.info(f"Ada Derana: collected {len(records)} articles")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 – Lankadeepa
# ─────────────────────────────────────────────────────────────────────────────

def scrape_lankadeepa(max_articles: int) -> list:
    records = []
    CONTENT_SELS = ["div.article-body", "div.article-body.sinhala-body"]
    seen = set()

    article_links = []
    for page in range(1, 12):
        url = f"https://www.lankadeepa.lk/latest-news/{page}"
        soup = get_soup(url)
        if soup is None:
            break
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/news/" in href and href not in seen and len(href) > 45:
                seen.add(href)
                article_links.append(href)
        time.sleep(DELAY)
        if len(article_links) > max_articles + 20:
            break

    for url in tqdm(article_links[:max_articles + 20], desc="Lankadeepa", unit="art"):
        if len(records) >= max_articles:
            break
        soup = get_soup(url)
        if soup is None:
            continue
        text = first_content(soup, CONTENT_SELS)
        if text:
            records.append({"source_url": url, "text_content": text})
        time.sleep(DELAY)

    logger.info(f"Lankadeepa: collected {len(records)} articles")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 – Dinamina
# ─────────────────────────────────────────────────────────────────────────────

def scrape_dinamina(max_articles: int) -> list:
    records = []
    CONTENT_SELS = ["div.entry-content", "div.td-post-content", "div.post-content"]
    seen = set()

    listing_pages = [
        "https://www.dinamina.lk/2026/05",
        "https://www.dinamina.lk/2026/04",
        "https://www.dinamina.lk/2026/03",
        "https://www.dinamina.lk/2026/02",
        "https://www.dinamina.lk/2026/01",
        "https://www.dinamina.lk/category/local",
        "https://www.dinamina.lk/category/politics",
        "https://www.dinamina.lk/category/sports",
        "https://www.dinamina.lk/category/world",
    ]

    article_links = []
    for lurl in listing_pages:
        soup = get_soup(lurl)
        if soup is None:
            continue
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "dinamina.lk" in href and "/20" in href and href not in seen:
                seen.add(href)
                article_links.append(href)
        time.sleep(DELAY)

    for url in tqdm(article_links, desc="Dinamina", unit="art"):
        if len(records) >= max_articles:
            break
        soup = get_soup(url)
        if soup is None:
            continue
        text = first_content(soup, CONTENT_SELS)
        if text:
            records.append({"source_url": url, "text_content": text})
        time.sleep(DELAY)

    logger.info(f"Dinamina: collected {len(records)} articles")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Source 4 – Mawbima
# ─────────────────────────────────────────────────────────────────────────────

def scrape_mawbima(max_articles: int) -> list:
    records = []
    CONTENT_SELS = ["div.td-post-content", "div.entry-content", "div.post-content"]
    seen = set()

    listing_pages = [
        "https://mawbima.lk/",
        "https://mawbima.lk/page/2/",
        "https://mawbima.lk/page/3/",
        "https://mawbima.lk/page/4/",
        "https://mawbima.lk/page/5/",
        "https://mawbima.lk/page/6/",
        "https://mawbima.lk/page/7/",
        "https://mawbima.lk/page/8/",
    ]

    article_links = []
    for lurl in listing_pages:
        soup = get_soup(lurl)
        if soup is None:
            continue
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "mawbima.lk/20" in href and href not in seen and len(href) > 40:
                seen.add(href)
                article_links.append(href)
        time.sleep(DELAY)

    for url in tqdm(article_links, desc="Mawbima", unit="art"):
        if len(records) >= max_articles:
            break
        soup = get_soup(url)
        if soup is None:
            continue
        text = first_content(soup, CONTENT_SELS)
        if text:
            records.append({"source_url": url, "text_content": text})
        time.sleep(DELAY)

    logger.info(f"Mawbima: collected {len(records)} articles")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Source 5 – Divaina
# ─────────────────────────────────────────────────────────────────────────────

def scrape_divaina(max_articles: int) -> list:
    records = []
    CONTENT_SELS = ["div.td-post-content", "div.entry-content", "div.post-content"]
    seen = set()

    listing_pages = [
        "https://www.divaina.com/",
        "https://www.divaina.com/page/2/",
        "https://www.divaina.com/page/3/",
        "https://www.divaina.com/page/4/",
        "https://www.divaina.com/page/5/",
        "https://www.divaina.com/breaking-news/",
        "https://www.divaina.com/category/local/",
        "https://www.divaina.com/category/politics/",
    ]

    article_links = []
    for lurl in listing_pages:
        soup = get_soup(lurl)
        if soup is None:
            continue
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "divaina.com/20" in href and href not in seen and len(href) > 40:
                seen.add(href)
                article_links.append(href)
        time.sleep(DELAY)

    for url in tqdm(article_links, desc="Divaina", unit="art"):
        if len(records) >= max_articles:
            break
        soup = get_soup(url)
        if soup is None:
            continue
        text = first_content(soup, CONTENT_SELS)
        if text:
            records.append({"source_url": url, "text_content": text})
        time.sleep(DELAY)

    logger.info(f"Divaina: collected {len(records)} articles")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Source 6 – Silumina
# ─────────────────────────────────────────────────────────────────────────────

def scrape_silumina(max_articles: int) -> list:
    records = []
    CONTENT_SELS = ["div.article-body", "div.field-items", "div.entry-content"]
    seen = set()

    listing_pages = [
        "https://www.silumina.lk/",
        "https://www.silumina.lk/latest-news/1",
        "https://www.silumina.lk/latest-news/2",
        "https://www.silumina.lk/latest-news/3",
        "https://www.silumina.lk/latest-news/4",
    ]

    article_links = []
    for lurl in listing_pages:
        soup = get_soup(lurl)
        if soup is None:
            continue
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "silumina.lk" in href and len(href) > 40 and href not in seen:
                seen.add(href)
                article_links.append(href)
        time.sleep(DELAY)

    for url in tqdm(article_links, desc="Silumina", unit="art"):
        if len(records) >= max_articles:
            break
        soup = get_soup(url)
        if soup is None:
            continue
        text = first_content(soup, CONTENT_SELS)
        if text:
            records.append({"source_url": url, "text_content": text})
        time.sleep(DELAY)

    logger.info(f"Silumina: collected {len(records)} articles")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

SCRAPERS = [
    ("Ada Derana Sinhala", scrape_adaderana),
    ("Lankadeepa",         scrape_lankadeepa),
    ("Dinamina",           scrape_dinamina),
    ("Mawbima",            scrape_mawbima),
    ("Divaina",            scrape_divaina),
    ("Silumina",           scrape_silumina),
]


def main():
    import sys

    all_records = []
    remaining = TARGET

    for name, fn in SCRAPERS:
        if remaining <= 0:
            break
        logger.info(f"\n{'='*60}\n  Source: {name}  (need {remaining} more)\n{'='*60}")
        try:
            batch = fn(max_articles=remaining)
        except Exception as exc:
            logger.error(f"{name} crashed: {exc}")
            batch = []
        all_records.extend(batch)
        remaining = TARGET - len(all_records)
        logger.info(f"  -> Total so far: {len(all_records)}")

    if not all_records:
        logger.error("No articles collected. Check network and site availability.")
        sys.exit(1)

    df = pd.DataFrame(all_records, columns=["source_url", "text_content"])
    df.drop_duplicates(subset="source_url", inplace=True)
    out = "sinhala_news.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n  Done! Saved {len(df)} articles -> '{out}'")
    print(df["source_url"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
