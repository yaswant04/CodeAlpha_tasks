"""
CodeAlpha Data Analytics Internship — TASK 1: WEB SCRAPING
=============================================================
Goal: Use BeautifulSoup to extract a real, structured dataset from a live
public web page and save it to a clean CSV file for downstream analysis.

Target page : https://github.com/trending  (GitHub's "Trending repositories" page)
Why this page: It is public, has a stable HTML structure, updates daily, and
gives a genuinely useful dataset (repo name, author, description, language,
star count, forks, and "stars today") that can feed real analysis.

Libraries used: requests (HTTP), BeautifulSoup (HTML parsing), pandas (storage)
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

URL = "https://github.com/trending"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def fetch_page(url: str) -> str:
    """Download the raw HTML of a page, with basic error handling & retry."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            print(f"[attempt {attempt + 1}] request failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"Could not fetch {url} after 3 attempts")


def clean_number(text: str) -> int:
    """Turn strings like '1,234' or '' into an int (0 if not found)."""
    if not text:
        return 0
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else 0


def parse_trending(html: str) -> pd.DataFrame:
    """Parse the trending page HTML into a tidy DataFrame."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for repo_box in soup.select("article.Box-row"):
        # Repo name looks like "owner / repo-name"
        title_tag = repo_box.select_one("h2 a")
        if not title_tag:
            continue
        full_name = " ".join(title_tag.get_text(strip=True).split())
        full_name = full_name.replace(" / ", "/")

        # Description (optional — some repos have none)
        desc_tag = repo_box.select_one("p")
        description = desc_tag.get_text(strip=True) if desc_tag else ""

        # Programming language
        lang_tag = repo_box.select_one('[itemprop="programmingLanguage"]')
        language = lang_tag.get_text(strip=True) if lang_tag else "Unknown"

        # Total stars & forks (the two "octicon" links near the bottom)
        stat_links = repo_box.select("a.Link--muted")
        total_stars = clean_number(stat_links[0].get_text()) if len(stat_links) > 0 else 0
        total_forks = clean_number(stat_links[1].get_text()) if len(stat_links) > 1 else 0

        # "X stars today"
        stars_today_tag = repo_box.select_one("span.d-inline-block.float-sm-right")
        stars_today = clean_number(stars_today_tag.get_text()) if stars_today_tag else 0

        rows.append(
            {
                "repository": full_name,
                "description": description,
                "language": language,
                "total_stars": total_stars,
                "total_forks": total_forks,
                "stars_today": stars_today,
            }
        )

    return pd.DataFrame(rows)


def main():
    print(f"Fetching: {URL}")
    html = fetch_page(URL)

    df = parse_trending(html)
    if df.empty:
        raise RuntimeError(
            "No repositories parsed — GitHub may have changed its page "
            "structure. Inspect the HTML and update the CSS selectors."
        )

    out_path = "../data/github_trending_scraped.csv"
    df.to_csv(out_path, index=False)

    print(f"\nScraped {len(df)} trending repositories.")
    print(df.head(10).to_string(index=False))
    print(f"\nSaved full dataset to: {out_path}")


if __name__ == "__main__":
    main()
