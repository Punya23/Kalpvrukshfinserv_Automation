"""
Kalpvruksh Finserv — Robust DDGS Web Collector
Fetches real phone numbers securely using HTML backend to bypass blocks.
"""

from ddgs import DDGS
import re
import logging
from typing import List
from server.lead_pipeline.core.schema import MasterLead
import time

logger = logging.getLogger(__name__)

class RobustWebCollector:
    def fetch_real_leads(self, query: str, limit: int = 40) -> List[MasterLead]:
        logger.info(f"Querying DuckDuckGo (HTML) for '{query}'...")
        leads = []
        seen_phones = set()
        
        try:
            with DDGS() as ddgs:
                # Use backend='lite' which does not get blocked by IP filters
                results = ddgs.text(query, backend="lite", max_results=limit * 4)
                
                for r in results:
                    text = r.get("body", "") + r.get("title", "")
                    
                    # Regex for Indian phone numbers
                    phone_match = re.search(r"(\+91[\s-]?\d{4,5}[\s-]?\d{5}|\b[7-9]\d{4}[\s-]?\d{5}\b|\b0\d{2,4}[\s-]?\d{6,8}\b)", text)
                    
                    if phone_match:
                        phone = phone_match.group(1).replace(" ", "").replace("-", "")
                        if phone in seen_phones:
                            continue
                            
                        seen_phones.add(phone)
                        
                        raw_title = r.get("title", "Pune Professional").split("-")[0].split("|")[0].strip()
                        
                        # Verify it's an individual professional
                        if any(kw in raw_title.lower() for kw in ["dr", "dr.", "ca", "c.a", "adv", "advocate"]):
                            name = raw_title
                        else:
                            continue
                        
                        lead = MasterLead.create(
                            name=name,
                            phone=phone,
                            profession="Professional",
                            company_or_clinic=name,
                            city="Pune",
                            source="duckduckgo_web",
                            source_url=r.get("href", ""),
                            lead_method="public_search",
                            notes=f"Fetched via DDG query: {query}"
                        )
                        leads.append(lead)
                        
                    if len(leads) >= limit:
                        break
                        
            logger.info(f"Successfully extracted {len(leads)} REAL leads.")
            return leads
            
        except Exception as e:
            logger.error(f"Failed to fetch from Web: {e}")
            return []
