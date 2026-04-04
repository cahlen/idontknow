#!/usr/bin/env python3
"""
Ping IndexNow + Google with changed URLs after deployment.

IndexNow notifies Bing, Yandex, Seznam, Naver, etc.
Google doesn't support IndexNow but we can ping their sitemap endpoint.

Usage:
    python3 scripts/indexnow.py                     # ping all finding URLs
    python3 scripts/indexnow.py --urls /findings/kronecker-s40-character-table/
"""

import argparse
import json
import sys
import glob
from pathlib import Path

SITE = "https://bigcompute.science"
# IndexNow key — this is NOT a secret, it's a public verification key
INDEXNOW_KEY = "bigcompute-science-indexnow-key"


def get_all_finding_urls():
    """Get all finding URLs from the sitemap or content directory."""
    urls = []
    website = Path(__file__).parent.parent.parent / "bigcompute.science"

    # From content directory
    findings_dir = website / "src" / "content" / "findings"
    if findings_dir.exists():
        for f in findings_dir.glob("*.md"):
            with open(f) as fh:
                for line in fh:
                    if line.startswith("slug:"):
                        slug = line.split(":", 1)[1].strip().strip('"')
                        urls.append(f"{SITE}/findings/{slug}/")
                        break

    # Also include key pages
    urls.extend([
        f"{SITE}/",
        f"{SITE}/findings/",
        f"{SITE}/verification/",
        f"{SITE}/about/",
        f"{SITE}/interactive/",
    ])

    return urls


def ping_indexnow(urls):
    """Ping IndexNow API (Bing, Yandex, Seznam, Naver)."""
    import httpx

    payload = {
        "host": "bigcompute.science",
        "key": INDEXNOW_KEY,
        "keyLocation": f"{SITE}/{INDEXNOW_KEY}.txt",
        "urlList": urls[:10000],  # IndexNow max 10K per request
    }

    try:
        resp = httpx.post(
            "https://api.indexnow.org/indexnow",
            json=payload,
            timeout=15.0,
        )
        print(f"IndexNow: {resp.status_code} ({len(urls)} URLs)")
        return resp.status_code in (200, 202)
    except Exception as e:
        print(f"IndexNow failed: {e}")
        return False


def ping_google_sitemap():
    """Ping Google to re-read the sitemap."""
    import httpx

    try:
        resp = httpx.get(
            f"https://www.google.com/ping?sitemap={SITE}/sitemap-index.xml",
            timeout=10.0,
        )
        print(f"Google sitemap ping: {resp.status_code}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Google ping failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="*", help="Specific URL paths to ping")
    args = parser.parse_args()

    if args.urls:
        urls = [f"{SITE}{u}" if not u.startswith("http") else u for u in args.urls]
    else:
        urls = get_all_finding_urls()

    print(f"Pinging {len(urls)} URLs...")
    ping_indexnow(urls)
    ping_google_sitemap()


if __name__ == "__main__":
    main()
