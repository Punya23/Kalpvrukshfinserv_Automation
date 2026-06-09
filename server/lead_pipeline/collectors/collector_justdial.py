"""
Kalpvruksh Finserv — JustDial Assisted Collector
Parses manually saved JustDial HTML pages to extract leads without triggering anti-bot systems.
"""

import os
import re
import logging
from bs4 import BeautifulSoup
from typing import List
from server.lead_pipeline.core.schema import MasterLead

logger = logging.getLogger(__name__)

class JustDialAssistedCollector:
    def __init__(self):
        # JustDial obfuscates numbers using CSS classes. 
        # This mapping decodes their standard icon classes to actual digits.
        self.icon_to_digit = {
            "icon-acb": "0", "icon-ji": "9", "icon-dc": "+",
            "icon-fe": "1", "icon-hg": "2", "icon-ba": "3",
            "icon-rq": "4", "icon-wx": "5", "icon-vu": "6",
            "icon-ts": "7", "icon-po": "8", "icon-nm": "9", # Variants
            "icon-lk": "8", "icon-ed": "5", "icon-gf": "6"  # Common variants, needs tuning per page
        }

    def _decode_phone(self, phone_element) -> str:
        """Decodes the obfuscated JustDial phone number from CSS span icons."""
        if not phone_element:
            return "N/A"
            
        phone = ""
        for span in phone_element.find_all("span", class_=re.compile(r"icon-")):
            classes = span.get("class", [])
            for c in classes:
                if c.startswith("icon-") and c in self.icon_to_digit:
                    phone += self.icon_to_digit[c]
        return phone if phone else "N/A"

    def extract_from_html(self, file_path: str, city: str = "Pune", profession: str = "Local Professional") -> List[MasterLead]:
        """
        Reads a locally saved JustDial HTML file (to avoid bot bans).
        Extracts businesses and normalizes them into MasterLead schema.
        """
        logger.info(f"Starting Assisted JustDial extraction from: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}. Please save the JustDial page first.")
            return []

        leads = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                
            # Find all business listing cards
            cards = soup.find_all("div", class_="resultbox_info")
            if not cards:
                # Try fallback selector
                cards = soup.find_all("li", class_="cntanr")
                
            for card in cards:
                try:
                    # Extract Name
                    name_elem = card.find("h2") or card.find(class_="jcn")
                    name = name_elem.text.strip() if name_elem else "N/A"
                    
                    if name == "N/A":
                        continue
                        
                    # Extract Phone (Decoded from spans)
                    phone_elem = card.find("p", class_="contact-info") or card.find("p", class_="comp-contact")
                    phone = self._decode_phone(phone_elem)
                    
                    # Normalization Layer
                    if phone != "N/A":
                        lead = MasterLead.create(
                            name=name,
                            phone=phone,
                            profession=profession,
                            company_or_clinic=name,
                            city=city,
                            source="justdial",
                            source_url="Assisted Manual Extraction",
                            lead_method="assisted_collection",
                            notes="Extracted via offline HTML parse"
                        )
                        leads.append(lead)
                        logger.debug(f"Extracted: {lead.name} | Phone: {lead.phone}")
                        
                except Exception as card_e:
                    logger.debug(f"Skipping a card due to parse error: {card_e}")
                    
        except Exception as e:
            logger.error(f"Failed to process HTML file: {e}")
            
        logger.info(f"Successfully extracted {len(leads)} leads from JustDial offline file.")
        return leads

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("JustDial Assisted Collector")
    print("Instructions:")
    print("1. Open your browser and search JustDial (e.g., 'CAs in Pune')")
    print("2. Save the page locally as 'justdial_sample.html' (Ctrl+S)")
    print("3. Run this script pointing to that file to extract leads cleanly without IP bans!")
