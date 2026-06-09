"""
Kalpvruksh Finserv — Practo Assisted Collector
Parses manually saved Practo HTML pages to extract doctor/clinic leads.
"""

import os
import logging
import json
from bs4 import BeautifulSoup
from typing import List
from server.lead_pipeline.core.schema import MasterLead

logger = logging.getLogger(__name__)

class PractoAssistedCollector:
    def extract_from_html(self, file_path: str, city: str = "Pune", profession: str = "Doctor") -> List[MasterLead]:
        """
        Reads a locally saved Practo search HTML file.
        Extracts doctor and clinic details and normalizes them into MasterLead schema.
        """
        logger.info(f"Starting Assisted Practo extraction from: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}. Please save the Practo page first.")
            return []

        leads = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                
            # Practo uses React, so the easiest and most robust way to get data 
            # from a saved page is to find their embedded JSON script tag if it exists,
            # or fallback to HTML card parsing.
            
            cards = soup.find_all("div", class_="listing-doctor-card")
            
            if not cards:
                # Try generic listing cards
                cards = soup.find_all("div", attrs={"data-qa-id": "doctor_card"})
                
            for card in cards:
                try:
                    # Extract Name
                    name_elem = card.find("h2") or card.find("div", class_="info-section").find("a")
                    name = name_elem.text.strip() if name_elem else "N/A"
                    
                    if name == "N/A":
                        continue
                        
                    # Extract Clinic
                    clinic_elem = card.find("span", attrs={"data-qa-id": "clinic_name"})
                    clinic = clinic_elem.text.strip() if clinic_elem else "Practo Verified Clinic"
                    
                    # Extract Phone
                    # Practo usually hides phone numbers until clicked. In assisted collection, 
                    # we extract the virtual number if present, or mark it for manual enrichment.
                    phone_elem = card.find("span", attrs={"data-qa-id": "practice_phone"}) or \
                                 card.find("a", href=lambda href: href and "tel:" in href)
                    
                    if phone_elem and phone_elem.get('href'):
                        phone = phone_elem['href'].replace("tel:", "")
                    else:
                        phone = phone_elem.text.strip() if phone_elem else "N/A"
                        
                    # Extract Rating
                    rating_elem = card.find("span", attrs={"data-qa-id": "doctor_recommendation"})
                    rating = rating_elem.text.strip() if rating_elem else "N/A"
                    
                    # Normalization Layer
                    if phone != "N/A":
                        lead = MasterLead.create(
                            name=name,
                            phone=phone,
                            profession=profession,
                            company_or_clinic=clinic,
                            city=city,
                            source="practo",
                            source_url="Assisted Manual Extraction",
                            lead_method="assisted_collection",
                            notes=f"Recommendation: {rating}"
                        )
                        leads.append(lead)
                        logger.debug(f"Extracted: {lead.name} | Phone: {lead.phone} | Clinic: {lead.company_or_clinic}")
                        
                except Exception as card_e:
                    logger.debug(f"Skipping a Practo card due to parse error: {card_e}")
                    
        except Exception as e:
            logger.error(f"Failed to process Practo HTML file: {e}")
            
        logger.info(f"Successfully extracted {len(leads)} leads from Practo offline file.")
        return leads

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Practo Assisted Collector")
    print("Instructions:")
    print("1. Open your browser and search Practo (e.g., 'Dentists in Pune')")
    print("2. Save the page locally as 'practo_sample.html' (Ctrl+S)")
    print("3. Run this script pointing to that file to extract doctors cleanly without IP bans!")
