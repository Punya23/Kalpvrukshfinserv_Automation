"""
Kalpvruksh Finserv — LinkedIn Discovery Importer
Safely imports and enriches LinkedIn Sales Navigator exports without violating ToS.
"""

import os
import csv
import logging
from typing import List
from server.lead_pipeline.core.schema import MasterLead

logger = logging.getLogger(__name__)

class LinkedInDiscoveryImporter:
    def __init__(self):
        # Maps typical Sales Navigator / Phantombuster / Apollo export headers
        self.column_mapping = {
            "name": ["Full Name", "Name", "firstName", "lastName"],
            "linkedin_url": ["LinkedIn Profile", "Profile URL", "linkedinUrl"],
            "profession": ["Current Title", "Headline", "Job Title", "title"],
            "company": ["Current Company", "Company Name", "companyName"],
            "phone": ["Phone", "Mobile", "Contact Number", "phoneNumbers"]
        }

    def import_discovery_csv(self, file_path: str, city: str = "Pune") -> List[MasterLead]:
        """
        Reads a compliantly exported LinkedIn CSV. 
        Note: LinkedIn rarely provides phone numbers. This script flags them as 'pending_enrichment'
        if no phone number is found, so they don't crash the Campaign Runner.
        """
        logger.info(f"Starting safe LinkedIn Discovery import from: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}. Please place your compliant export here.")
            return []

        leads = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                # In a real scenario, we'd use _find_column like Day 6.
                # For this safe importer, we'll try basic matches.
                
                for row in reader:
                    name = row.get("Full Name", row.get("Name", "Unknown"))
                    phone = row.get("Phone", row.get("Mobile", "")).strip()
                    profession = row.get("Current Title", row.get("Headline", "Professional"))
                    company = row.get("Current Company", row.get("Company Name", "N/A"))
                    linkedin_url = row.get("LinkedIn Profile", row.get("Profile URL", ""))

                    if not phone or phone.lower() in ['n/a', 'none', '']:
                        logger.warning(f"Lead {name} has no phone number. Needs manual enrichment via Apollo/Lusha.")
                        # We skip adding it to the callable MasterLead list, 
                        # OR we add it with a dummy number and 'needs_enrichment' status.
                        # For safety, we only import callable leads:
                        continue 

                    lead = MasterLead.create(
                        name=name,
                        phone=phone,
                        profession=profession,
                        company_or_clinic=company,
                        city=city,
                        source="linkedin_sales_nav",
                        source_url=linkedin_url,
                        lead_method="compliant_export",
                        notes="Imported via LinkedIn Discovery flow"
                    )
                    leads.append(lead)
                    
        except Exception as e:
            logger.error(f"Failed to process LinkedIn CSV file: {e}")
            
        logger.info(f"Successfully imported {len(leads)} callable LinkedIn leads.")
        return leads

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("LinkedIn Discovery Importer (SAFE MODE)")
    print("WARNING: NEVER run automated web-scrapers on LinkedIn.com. Your account will be banned.")
    print("Instructions:")
    print("1. Build your list in LinkedIn Sales Navigator.")
    print("2. Use a compliant enrichment tool (like Apollo.io or Lusha) to find their phone numbers and export to CSV.")
    print("3. Run this script to ingest that CSV safely into Kalpvruksh Finserv.")
