"""
Kalpvruksh Finserv — OSM Collector
Fetches real business leads using the OpenStreetMap API.
"""

import requests
import logging
from typing import List
from server.lead_pipeline.core.schema import MasterLead

logger = logging.getLogger(__name__)

class OSMCollector:
    def fetch_real_leads(self, city: str = "Pune", limit: int = 21) -> List[MasterLead]:
        logger.info(f"Querying OpenStreetMap for real individual professionals in {city}...")
        
        query = f"""
        [out:json][timeout:25];
        area[name="{city}"]->.searchArea;
        (
          node["amenity"~"doctors|dentist|veterinary"](area.searchArea);
          node["office"~"lawyer|tax_advisor|accountant"](area.searchArea);
        );
        out {limit * 4};
        """
        
        try:
            response = requests.post(
                "https://overpass.kumi.systems/api/interpreter", 
                data={'data': query},
                headers={"User-Agent": "KalpvrukshLeadGen/1.0"}
            )
            data = response.json()
            
            leads = []
            for element in data.get("elements", []):
                tags = element.get("tags", {})
                name = tags.get("name")
                phone = tags.get("phone") or tags.get("contact:phone") or tags.get("contact:mobile")
                
                if name and phone:
                    phone = phone.split(";")[0].split(",")[0].strip()
                    
                    # Deduce profession
                    prof = "Doctor" if tags.get("amenity") in ["doctors", "dentist", "veterinary"] else "CA/Advocate"
                    
                    lead = MasterLead.create(
                        name=name,
                        phone=phone,
                        profession=prof,
                        company_or_clinic=name,
                        city=city,
                        source="openstreetmap",
                        source_url="https://www.openstreetmap.org/node/" + str(element.get("id")),
                        lead_method="public_api",
                        notes="Fetched via live OSM API"
                    )
                    leads.append(lead)
            
            # Cap it precisely to the limit requested
            leads = leads[:limit]
            logger.info(f"Successfully extracted {len(leads)} REAL leads from OpenStreetMap.")
            return leads
            
        except Exception as e:
            logger.error(f"Failed to fetch from OSM API: {e}")
            return []
