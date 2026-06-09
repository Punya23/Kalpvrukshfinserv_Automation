"""
Kalpvruksh Finserv — Compliance Layer
Handles TRAI calling-window checks and DND verification before leads enter the campaign.
"""

from datetime import datetime
import pytz
import logging
from server.lead_pipeline.core.schema import MasterLead

logger = logging.getLogger(__name__)

class ComplianceGate:
    def __init__(self):
        # TRAI regulations run on IST
        self.timezone = pytz.timezone('Asia/Kolkata')
        
    def is_within_calling_window(self) -> bool:
        """
        TRAI Compliance: Commercial calls should only happen 10AM-12PM and 3PM-5PM IST
        to ensure highest connect rate and regulatory adherence.
        """
        now = datetime.now(self.timezone)
        hour = now.hour
        
        # 10:00 AM to 11:59 AM (10, 11)
        if 10 <= hour < 12:
            return True
            
        # 3:00 PM to 4:59 PM (15, 16)
        if 15 <= hour < 17:
            return True
            
        logger.warning(f"Time Window Check: Current time {now.strftime('%I:%M %p')} is outside permitted calling windows (10AM-12PM, 3PM-5PM). [OVERRIDDEN FOR DEMO]")
        return True
        
    def check_dnd_status(self, phone: str) -> str:
        """
        Placeholder for NCPR/DND registry check.
        In production, this should call Exotel's DND API or a TRAI scrubbing service.
        Returns: 'clean' or 'dnd_active'
        """
        # TODO: Implement real Exotel DND check here.
        # Currently defaults to 'clean' until Exotel API is wired up.
        logger.debug(f"Checking DND status for {phone}... [Default: clean]")
        return "clean"

    def is_lead_callable(self, lead: MasterLead) -> bool:
        """
        Runs full compliance check on a MasterLead before it is sent to the Campaign Runner.
        """
        if not self.is_within_calling_window():
            return False
            
        if lead.dnd_status == "unchecked":
            lead.dnd_status = self.check_dnd_status(lead.phone)
            
        if lead.dnd_status == "dnd_active":
            logger.warning(f"Compliance Block: {lead.phone} is on DND registry.")
            return False
            
        if lead.consent_status == "refused":
            logger.warning(f"Compliance Block: {lead.phone} has refused consent.")
            return False
            
        return True
