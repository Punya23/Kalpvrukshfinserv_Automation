"""
Kalpvruksh Finserv — Unified CSV Manager
Handles saving normalized MasterLeads into the Campaign Runner input format.
"""

import csv
import os
from pathlib import Path
from typing import List
import logging
from server.lead_pipeline.core.schema import MasterLead

logger = logging.getLogger(__name__)

class UnifiedCSVManager:
    def __init__(self, output_dir: str = "data/leads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / "unified_compliant_leads.csv"
        
        # Initialize file with headers if it doesn't exist
        if not self.filepath.exists():
            self._write_headers()

    def _write_headers(self):
        """Creates the file with the Master Schema headers."""
        # Using a dummy lead to dynamically grab all field names from the dataclass
        dummy = MasterLead.create("Test", "0", "Test", "Test", "Test", "Test", "Test")
        headers = list(dummy.to_csv_row().keys())
        
        with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def save_leads(self, leads: List[MasterLead]):
        """Appends normalized leads to the master CSV, ensuring no duplicate phone numbers."""
        if not leads:
            return
            
        # 1. Load existing leads to check for duplicates
        existing_leads = self.load_leads()
        existing_phones = {row['phone'] for row in existing_leads}
        
        # 2. Filter out leads that are already in the database
        unique_new_leads = []
        for lead in leads:
            if lead.phone not in existing_phones:
                unique_new_leads.append(lead)
                existing_phones.add(lead.phone) # Add to set to prevent duplicates within the new batch itself
                
        if not unique_new_leads:
            logger.info("No new unique leads to save (all were duplicates).")
            return
            
        headers = list(unique_new_leads[0].to_csv_row().keys())
        
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            for lead in unique_new_leads:
                writer.writerow(lead.to_csv_row())
                
        logger.info(f"Successfully saved {len(unique_new_leads)} NEW unique leads to {self.filepath} (skipped {len(leads) - len(unique_new_leads)} duplicates).")

    def load_leads(self) -> List[dict]:
        """Loads all leads for the Campaign Runner to process."""
        if not self.filepath.exists():
            return []
            
        leads = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                leads.append(row)
        return leads
