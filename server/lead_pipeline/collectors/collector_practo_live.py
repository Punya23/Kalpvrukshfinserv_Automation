"""
Kalpvruksh Finserv — Practo Live Collector
Scrapes Practo listing pages for profile URLs, then fetches individual
doctor profiles to extract phone numbers from JSON-LD structured data.

Verified to work from cloud/server IPs (Railway).
Phones come from application/ld+json on each profile page.
"""
import json
import logging
import time
import re
import random
from typing import List

import requests
from bs4 import BeautifulSoup

from server.lead_pipeline.core.schema import MasterLead

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.practo.com/",
}

# Practo listing URLs for each profession
PRACTO_LISTING_URLS = {
    "doctor":   "https://www.practo.com/pune/doctors",
    "dentist":  "https://www.practo.com/pune/dentist",
    "CA":       None,   # Practo doesn't have CAs — falls back to other collectors
    "architect": None,
    "interior_designer": None,
}


class PractoLiveCollector:
    """Live Practo scraper — works from cloud IPs via JSON-LD structured data."""

    def fetch_leads(self, profession: str = "doctor", limit: int = 15) -> List[MasterLead]:
        listing_url = PRACTO_LISTING_URLS.get(profession)
        if not listing_url:
            logger.info(f"[PractoLive] No Practo listing for profession '{profession}'")
            return []

        logger.info(f"[PractoLive] Fetching listing: {listing_url}")
        profile_urls = self._get_profile_urls(listing_url, max_profiles=limit)
        if not profile_urls:
            logger.warning(f"[PractoLive] No profile URLs found on listing page")
            return []

        leads: List[MasterLead] = []
        for url in profile_urls[:limit]:
            lead = self._extract_lead_from_profile(url, profession)
            if lead:
                leads.append(lead)
            time.sleep(random.uniform(0.5, 1.5))  # polite delay

        logger.info(f"[PractoLive] [{profession}] → {len(leads)} leads with phone")
        return leads

    def _get_profile_urls(self, listing_url: str, max_profiles: int = 20) -> List[str]:
        """Fetch the listing page and extract individual doctor profile URLs."""
        try:
            resp = requests.get(listing_url, headers=_HEADERS, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"[PractoLive] Listing page HTTP {resp.status_code}")
                return []

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract profile URLs from JSON-LD on listing page
            urls = []
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string or "")
                    if isinstance(data, dict) and data.get("url"):
                        u = data["url"]
                        if "/pune/doctor/" in u or "/pune/therapist/" in u or "/pune/dentist/" in u:
                            if u not in urls:
                                urls.append(u)
                except Exception:
                    pass

            # Also look for <a> tags with doctor profile hrefs
            for a in soup.find_all("a", href=re.compile(r"/pune/(doctor|therapist|dentist)/")):
                u = "https://www.practo.com" + a["href"] if a["href"].startswith("/") else a["href"]
                # Remove query strings
                u = u.split("?")[0]
                if u not in urls:
                    urls.append(u)

            logger.info(f"[PractoLive] Found {len(urls)} profile URLs")
            return urls[:max_profiles]

        except Exception as e:
            logger.warning(f"[PractoLive] Failed to fetch listing: {e}")
            return []

    def _extract_lead_from_profile(self, profile_url: str, profession: str) -> MasterLead | None:
        """Fetch a doctor profile page and extract name + phone from JSON-LD or raw HTML."""
        try:
            resp = requests.get(profile_url, headers=_HEADERS, timeout=12)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            
            name = ""
            phone = ""
            city = "Pune"
            
            # Try JSON-LD first
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string or "")
                    if isinstance(data, list):
                        data = data[0] if data else {}
                    if not isinstance(data, dict):
                        continue
                        
                    if data.get("name"):
                        name = data.get("name", "").strip()
                    if data.get("telephone") or data.get("phone"):
                        phone = (data.get("telephone", "") or data.get("phone", "")).strip()
                    
                    address = data.get("address", {})
                    if isinstance(address, dict) and address.get("addressLocality"):
                        city = address.get("addressLocality")
                except Exception:
                    pass

            # Fallback: Extract from raw HTML if JSON-LD failed
            if not name:
                name_tag = soup.find("h1")
                if name_tag:
                    name = name_tag.text.strip()
            
            if not phone:
                # Try finding practice_phone in raw JSON strings embedded in HTML
                phones = re.findall(r'\"practice_phone\":\"(\d+)\"', resp.text)
                if not phones:
                    # Generic Indian mobile number regex (starts with 6-9, 10 digits)
                    phones = re.findall(r'\b[6-9]\d{9}\b', resp.text)
                if phones:
                    # Pick the first valid-looking number that isn't a Practo toll-free
                    for p in phones:
                        if len(p) == 10 and not p.startswith("777777"):
                            phone = p
                            break
                    if not phone and phones:
                        phone = phones[0]

            if name and phone:
                phone = phone.replace("+91", "").replace(" ", "").replace("-", "")
                return MasterLead.create(
                    name=name,
                    phone=phone,
                    profession=profession,
                    city=city,
                    source="practo_live",
                    source_url=profile_url,
                    lead_method="public_listing",
                )
        except Exception as e:
            logger.debug(f"[PractoLive] Profile fetch failed {profile_url}: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    c = PractoLiveCollector()
    leads = c.fetch_leads("doctor", limit=5)
    for l in leads:
        print(f"{l.name} | {l.phone} | {l.city}")
