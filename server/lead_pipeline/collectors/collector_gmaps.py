"""
Kalpvruksh Finserv — Google Maps Collector
Extracts local business profiles from Google Local and maps them into the Master Lead schema.
"""

import re
import random
import time
import logging
import requests
from bs4 import BeautifulSoup
from typing import List
from server.lead_pipeline.core.schema import MasterLead

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0"
]

class GoogleMapsCollector:
    def __init__(self):
        self.headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

    def _get_headers(self):
        headers = self.headers.copy()
        headers["User-Agent"] = random.choice(USER_AGENTS)
        return headers

    def fetch_business_leads(self, query: str, city: str = "Pune", max_results: int = 20) -> List[MasterLead]:
        """
        Scrapes local business listings from Google Search Local Results.
        Normalizes the results into MasterLead format.
        """
        logger.info(f"Starting Google Maps extraction for query: '{query}' in {city}")
        
        leads = []
        start = 0
        search_query = f"{query} in {city}"
        
        while len(leads) < max_results:
            url = f"https://www.google.com/search?q={requests.utils.quote(search_query)}&tbm=lcl&start={start}"
            
            try:
                response = requests.get(url, headers=self._get_headers(), timeout=15)
                if response.status_code != 200:
                    logger.error(f"Failed to fetch Google page. Status code: {response.status_code}")
                    break
                    
                soup = BeautifulSoup(response.text, "html.parser")
                cards = soup.find_all("div", class_=re.compile(r"VkCve|rllt__details"))
                
                if not cards:
                    cards = soup.find_all("div", attrs={"data-cid": True})
                    
                if not cards:
                    logger.info("No more GMaps listings found or page structure changed.")
                    break
                    
                for card in cards:
                    if len(leads) >= max_results:
                        break
                        
                    try:
                        # Extract Name
                        name_elem = card.find(class_=re.compile(r"OSrXXb|dbg0pd|q81Yee"))
                        name = name_elem.text.strip() if name_elem else "N/A"
                        
                        if name == "N/A" or name in [lead.name for lead in leads]:
                            continue
                            
                        details_text = card.text.strip()
                        
                        # Extract Rating
                        rating_match = re.search(r"(\d\.\d)\s*★", details_text)
                        rating = rating_match.group(1) if rating_match else "N/A"
                        
                        # Extract Phone Number
                        phone_match = re.search(
                            r"(\+91[\s-]?\d{4,5}[\s-]?\d{5}|\b\d{5}[\s-]?\d{5}\b|\b0\d{2,4}[\s-]?\d{6,8}\b)", 
                            details_text
                        )
                        phone = phone_match.group(1).replace(" ", "").replace("-", "") if phone_match else "N/A"
                        
                        if phone == "N/A":
                            continue # Skip leads without phone numbers
                        
                        # Normalization Layer: Convert into Master Schema
                        lead = MasterLead.create(
                            name=name,
                            phone=phone,
                            profession=query.title(),  # e.g., "Architects"
                            company_or_clinic=name,
                            city=city,
                            source="google_maps",
                            source_url=url,
                            lead_method="public_listing",
                            notes=f"Rating: {rating} stars"
                        )
                        leads.append(lead)
                        logger.info(f"Extracted: {lead.name} | Phone: {lead.phone}")
                        
                    except Exception as card_e:
                        continue
                
                start += 20
                time.sleep(random.uniform(2.0, 5.0))
                
            except Exception as e:
                logger.error(f"Error fetching GMaps page: {e}")
                break
                
        logger.info(f"Successfully extracted {len(leads)} valid leads from Google Maps.")
        return leads

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = GoogleMapsCollector()
    gmaps_leads = collector.fetch_business_leads(query="Architects", city="Pune", max_results=5)
