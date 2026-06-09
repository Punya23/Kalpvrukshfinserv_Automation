"""
Kalpvruksh Finserv — Nominatim Maps Collector
Fetches real business leads using OSM Geocoding.
"""

import requests
import logging
from typing import List
from server.lead_pipeline.core.schema import MasterLead

logger = logging.getLogger(__name__)

class NominatimCollector:
    def fetch_real_leads(self, query: str = "hospital in pune", limit: int = 21) -> List[MasterLead]:
        logger.info(f"Querying Nominatim for '{query}'...")
        leads = []
        
        try:
            # We must use a custom User-Agent per Nominatim Terms of Use
            headers = {"User-Agent": "KalpvrukshLeadGen/1.0 (test@example.com)"}
            
            # Nominatim search endpoint
            url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(query)}&format=json&extratags=1&limit={limit*3}"
            
            response = requests.get(url, headers=headers)
            results = response.json()
            
            for r in results:
                extratags = r.get("extratags", {})
                
                # We need a name and phone
                name = r.get("name") or extratags.get("name")
                phone = extratags.get("phone") or extratags.get("contact:phone") or extratags.get("contact:mobile")
                
                if name and phone:
                    phone = phone.split(";")[0].split(",")[0].strip()
                    
                    lead = MasterLead.create(
                        name=name,
                        phone=phone,
                        profession=query.split(" ")[0].title(),
                        company_or_clinic=name,
                        city=r.get("display_name", "").split(",")[-2].strip() if "," in r.get("display_name", "") else "Pune",
                        source="nominatim_maps",
                        source_url="https://www.openstreetmap.org/node/" + str(r.get("osm_id")),
                        lead_method="public_search",
                        notes="Fetched via live Nominatim search"
                    )
                    leads.append(lead)
                    
                if len(leads) >= limit:
                    break
                        
            logger.info(f"Successfully extracted {len(leads)} REAL leads from Nominatim.")
            return leads
            
        except Exception as e:
            logger.error(f"Failed to fetch from Nominatim: {e}")
            return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = NominatimCollector()
    leads = collector.fetch_real_leads("doctors in pune", 21)
    for l in leads:
        print(f"{l.name} | {l.phone}")
