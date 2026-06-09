"""
Kalpvruksh Finserv — BizKonnect Paid Leads Importer
Maps and imports verified B2B CSV data from paid providers like BizKonnect into the Master Schema.
"""

import os
import csv
import logging
from typing import List
from server.lead_pipeline.core.schema import MasterLead

logger = logging.getLogger(__name__)

class BizKonnectImporter:
    def __init__(self):
        # Expected column headers from a standard BizKonnect/B2B data export
        self.column_mapping = {
            "name": ["Full Name", "Contact Name", "Name", "First Name"],
            "phone": ["Mobile", "Direct Phone", "Phone Number", "Mobile Phone"],
            "profession": ["Job Title", "Designation", "Title", "Role"],
            "company": ["Company Name", "Account Name", "Organization", "Company"],
            "city": ["City", "Location", "HQ City"]
        }

    def _find_column(self, headers: List[str], possible_names: List[str]) -> str:
        """Helper to dynamically find the correct column name based on common variants."""
        for header in headers:
            if header.strip() in possible_names:
                return header
        return None

    def import_csv(self, file_path: str) -> List[MasterLead]:
        """
        Reads a paid provider CSV file and maps it to the MasterLead schema.
        """
        logger.info(f"Starting BizKonnect import from: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}. Please place the CSV file in the directory.")
            return []

        leads = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                # Map columns dynamically
                name_col = self._find_column(headers, self.column_mapping["name"])
                phone_col = self._find_column(headers, self.column_mapping["phone"])
                prof_col = self._find_column(headers, self.column_mapping["profession"])
                comp_col = self._find_column(headers, self.column_mapping["company"])
                city_col = self._find_column(headers, self.column_mapping["city"])
                
                if not name_col or not phone_col:
                    logger.error("CRITICAL: Missing mandatory Name or Phone columns in the CSV.")
                    return []

                for row in reader:
                    phone = row.get(phone_col, "").strip()
                    if not phone or phone.lower() in ['n/a', 'none', '']:
                        continue # Skip records without direct phones
                        
                    name = row.get(name_col, "Unknown")
                    
                    # Handle split First Name / Last Name if necessary
                    if name_col == "First Name" and "Last Name" in headers:
                        name = f"{row['First Name']} {row.get('Last Name', '')}".strip()

                    lead = MasterLead.create(
                        name=name,
                        phone=phone,
                        profession=row.get(prof_col, "Professional") if prof_col else "Professional",
                        company_or_clinic=row.get(comp_col, "N/A") if comp_col else "N/A",
                        city=row.get(city_col, "Pune") if city_col else "Pune",
                        source="bizkonnect",
                        source_url="Paid Data Import",
                        lead_method="paid_verified",
                        notes="Imported via CSV"
                    )
                    leads.append(lead)
                    
        except Exception as e:
            logger.error(f"Failed to process CSV file: {e}")
            
        logger.info(f"Successfully imported and normalized {len(leads)} B2B leads.")
        return leads

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("BizKonnect / Paid Provider Importer")
    print("Instructions:")
    print("1. Export your purchased lead list as a CSV.")
    print("2. Run this script pointing to that CSV file to automatically map the fields to Kalpvruksh Finserv's Master Schema.")
