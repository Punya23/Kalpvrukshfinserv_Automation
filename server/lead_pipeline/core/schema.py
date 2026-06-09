"""
Kalpvruksh Finserv — Master Lead Schema
Defines the normalized structure for all imported leads before they enter the campaign runner.
"""

from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime

@dataclass
class MasterLead:
    name: str
    phone: str
    profession: str
    company_or_clinic: str
    city: str
    source: str  # e.g., 'icai_directory', 'practo', 'google_maps', 'bizkonnect'
    source_url: str
    lead_method: str  # e.g., 'official_directory', 'public_listing', 'paid_verified', 'manual_discovery'
    consent_status: str  # e.g., 'pending_disclosure', 'consented', 'refused'
    dnd_status: str  # e.g., 'unchecked', 'clean', 'dnd_active'
    date_collected: str
    notes: str = ""

    def to_csv_row(self) -> dict:
        return asdict(self)
    
    @classmethod
    def create(cls, name: str, phone: str, profession: str, city: str, source: str, source_url: str, lead_method: str, company_or_clinic: str = "N/A", notes: str = ""):
        """Helper to create a lead with auto-standardized formatting."""
        # Standardize phone number format for Exotel (+91XXXXXXXXXX)
        clean_phone = phone.replace(" ", "").replace("-", "")
        if clean_phone and not clean_phone.startswith("+"):
            if clean_phone.startswith("91") and len(clean_phone) == 12:
                clean_phone = f"+{clean_phone}"
            elif len(clean_phone) == 10:
                clean_phone = f"+91{clean_phone}"

        return cls(
            name=name.strip(),
            phone=clean_phone,
            profession=profession,
            company_or_clinic=company_or_clinic,
            city=city,
            source=source,
            source_url=source_url,
            lead_method=lead_method,
            consent_status="pending_disclosure", # Must be disclosed by AI bot during call
            dnd_status="unchecked",              # Must be checked by compliance layer
            date_collected=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            notes=notes
        )
