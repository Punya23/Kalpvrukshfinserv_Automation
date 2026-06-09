"""
Kalpvruksh Finserv — DuckDuckGo Local Collector
Fetches real business leads bypassing Google's strict bot protections.
"""

from duckduckgo_search import DDGS
import logging
from typing import List
from server.lead_pipeline.core.schema import MasterLead

logger = logging.getLogger(__name__)

class DDGCollector:
    def fetch_real_leads(self, query: str = "doctors in pune", limit: int = 21) -> List[MasterLead]:
        logger.info(f"Querying DuckDuckGo Maps for '{query}'...")
        leads = []
        
        try:
            with DDGS() as ddgs:
                results = ddgs.maps(query, max_results=limit)
                
                for r in results:
                    name = r.get("title", "")
                    phone = r.get("phone", "")
                    address = r.get("address", "")
                    url = r.get("url", "")
                    
                    if name and phone:
                        lead = MasterLead.create(
                            name=name,
                            phone=phone,
                            profession=query.split(" ")[0].title(),
                            company_or_clinic=name,
                            city=address,
                            source="duckduckgo_maps",
                            source_url=url,
                            lead_method="public_search",
                            notes="Fetched via live DDG search"
                        )
                        leads.append(lead)
                        
            logger.info(f"Successfully extracted {len(leads)} REAL leads from DDG.")
            return leads
            
        except Exception as e:
            logger.error(f"Failed to fetch from DDG: {e}")
            return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = DDGCollector()
    leads = collector.fetch_real_leads("doctors in pune", 5)
    for l in leads:
        print(f"{l.name} | {l.phone}")
