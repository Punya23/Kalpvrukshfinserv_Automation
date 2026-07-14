"""
Kalpvruksh Finserv — Unified CSV Manager
Handles saving normalized MasterLeads into the Campaign Runner input format.

Persistence strategy (Railway-safe):
  - Each night's scrape is saved to a DATED file: data/leads/leads_YYYY-MM-DD.csv
  - A rolling 'unified_compliant_leads.csv' is kept as an alias to today's file
  - The campaign runner resolves the most recent dated file automatically
  - This way data survives across redeploys as long as a Railway Volume is mounted
    at /app/data (or the files are committed to git as fallback)
"""

import csv
import os
import shutil
from datetime import date
from pathlib import Path
from typing import List
import logging
from server.lead_pipeline.core.schema import MasterLead

logger = logging.getLogger(__name__)

LEADS_DIR = Path("data/leads")


def get_latest_dated_csv() -> Path | None:
    """
    Return the most recent leads_YYYY-MM-DD.csv in data/leads/.
    Returns None if no dated file exists yet.
    """
    LEADS_DIR.mkdir(parents=True, exist_ok=True)
    dated = sorted(LEADS_DIR.glob("leads_????-??-??.csv"), reverse=True)
    return dated[0] if dated else None


def get_best_campaign_csv() -> str:
    """
    Return the best available CSV path for a campaign run.
    Priority:
      1. Most recent dated file (today's or yesterday's scrape)
      2. unified_compliant_leads.csv (legacy / Railway Volume)
      3. hni_leads_pune.csv (hardcoded seed — always in git)
    """
    latest = get_latest_dated_csv()
    if latest and latest.stat().st_size > 200:
        logger.info(f"[CSVManager] Using dated leads file: {latest}")
        return str(latest)

    unified = LEADS_DIR / "unified_compliant_leads.csv"
    if unified.exists() and unified.stat().st_size > 200:
        logger.info(f"[CSVManager] Using unified_compliant_leads.csv")
        return str(unified)

    seed = LEADS_DIR / "hni_leads_pune.csv"
    logger.info(f"[CSVManager] Falling back to seed file: {seed}")
    return str(seed)


class UnifiedCSVManager:
    def __init__(self, output_dir: str = "data/leads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Primary file: dated (survives redeploys on Railway Volume)
        today = date.today().isoformat()
        self.dated_filepath = self.output_dir / f"leads_{today}.csv"

        # Alias: keep unified_compliant_leads.csv pointing at today's file
        self.filepath = self.output_dir / "unified_compliant_leads.csv"

        # Initialize dated file with headers if needed
        if not self.dated_filepath.exists():
            self._write_headers(self.dated_filepath)
        # Keep the unified alias in sync
        if not self.filepath.exists():
            self._write_headers(self.filepath)

    def _write_headers(self, path: Path):
        """Creates the file with the Master Schema headers."""
        dummy = MasterLead.create("Test", "0", "Test", "Test", "Test", "Test", "Test")
        headers = list(dummy.to_csv_row().keys())
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def save_leads(self, leads: List[MasterLead]):
        """Appends normalized leads to BOTH the dated file and the unified alias."""
        if not leads:
            return

        # Deduplicate against the dated file (primary) and unified (alias)
        existing_phones: set = set()
        for fpath in [self.dated_filepath, self.filepath]:
            for row in self._load_from(fpath):
                existing_phones.add(row.get("phone", ""))

        unique_new_leads = []
        for lead in leads:
            if lead.phone not in existing_phones:
                unique_new_leads.append(lead)
                existing_phones.add(lead.phone)

        if not unique_new_leads:
            logger.info("No new unique leads to save (all were duplicates).")
            return

        headers = list(unique_new_leads[0].to_csv_row().keys())

        # Write to dated file (Railway Volume persistent)
        with open(self.dated_filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            for lead in unique_new_leads:
                writer.writerow(lead.to_csv_row())

        # Keep unified alias in sync (append same rows)
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            for lead in unique_new_leads:
                writer.writerow(lead.to_csv_row())

        logger.info(
            f"Saved {len(unique_new_leads)} NEW leads → {self.dated_filepath.name} "
            f"(skipped {len(leads) - len(unique_new_leads)} dupes)"
        )

    def _load_from(self, path: Path) -> List[dict]:
        if not path.exists():
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return list(csv.DictReader(f))
        except Exception:
            return []

    def load_leads(self) -> List[dict]:
        """Load leads from the dated file (preferred) or unified alias."""
        latest = get_latest_dated_csv()
        source = latest if (latest and latest.stat().st_size > 200) else self.filepath
        return self._load_from(source)
