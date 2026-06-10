"""
Kalpvruksh Finserv AI Automation — Lead Scoring Engine
Scores leads from 0-10 based on conversation signals.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class LeadCategory(str, Enum):
    HOT = "HOT"          # Score >= 7 → immediate manager callback
    WARM = "WARM"        # Score 4-6 → nurture pipeline
    COLD = "COLD"        # Score < 4 → thank and close
    DNC = "DNC"          # Do Not Contact — explicitly refused


class LeadSource(str, Enum):
    INBOUND_WHATSAPP = "inbound_whatsapp"
    INBOUND_CALL = "inbound_call"
    OUTBOUND_CALL = "outbound_call"
    WEBSITE_FORM = "website_form"
    REFERRAL = "referral"
    SOCIAL_MEDIA = "social_media"


class BotType(str, Enum):
    INSURANCE = "insurance"
    INVESTMENT = "investment"
    REMINDER = "reminder"
    RECRUITMENT = "recruitment"


@dataclass
class LeadData:
    """Structured lead data collected during conversation."""
    # Basic Info
    name: str = ""
    phone: str = ""
    email: str = ""

    # Demographics
    age: Optional[int] = None
    occupation: str = ""
    annual_income: Optional[str] = None
    city: str = "Pune"

    # Family
    family_members: int = 1
    has_spouse: bool = False
    has_children: bool = False
    num_children: int = 0
    has_senior_parents: bool = False

    # Insurance-Specific
    currently_insured: bool = False
    current_insurer: str = ""
    current_sum_insured: Optional[int] = None
    insurance_interest: str = ""  # health/life/both

    # Investment-Specific
    current_investments: str = ""  # FD/gold/MF/none
    investable_surplus: Optional[int] = None
    financial_goal: str = ""  # education/retirement/wealth/tax
    time_horizon_years: Optional[int] = None
    risk_appetite: str = ""  # conservative/moderate/aggressive

    # Engagement Signals
    asked_about_premium: bool = False
    asked_about_specific_plan: bool = False
    mentioned_urgency: bool = False  # medical event, job change
    ready_to_buy: bool = False
    said_not_interested: bool = False
    asked_for_callback: bool = False

    # Meta
    source: LeadSource = LeadSource.INBOUND_WHATSAPP
    bot_type: BotType = BotType.INSURANCE
    conversation_summary: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Computed
    score: int = 0
    category: LeadCategory = LeadCategory.COLD


def score_insurance_lead(lead: LeadData) -> LeadData:
    """Score an insurance lead based on qualification signals."""
    score = 0

    # DNC check first
    if lead.said_not_interested:
        lead.score = 0
        lead.category = LeadCategory.DNC
        return lead

    # Buying intent signals (+3 each)
    if lead.ready_to_buy:
        score += 3
    if lead.asked_about_premium:
        score += 3

    # Family signals (+2 each)
    if lead.has_spouse or lead.has_children:
        score += 2
    if not lead.currently_insured:
        score += 2

    # Demographic fit (+1 each)
    if lead.age and 30 <= lead.age <= 55:
        score += 1
    if lead.annual_income and lead.annual_income not in ("", "unknown"):
        score += 1

    # Engagement signals (+1-2 each)
    if lead.asked_about_specific_plan:
        score += 1
    if lead.mentioned_urgency:
        score += 2
    if lead.asked_for_callback:
        score += 1

    # Cap at 10
    lead.score = min(score, 10)

    # Categorize
    if lead.score >= 7:
        lead.category = LeadCategory.HOT
    elif lead.score >= 4:
        lead.category = LeadCategory.WARM
    else:
        lead.category = LeadCategory.COLD

    return lead


def score_investment_lead(lead: LeadData) -> LeadData:
    """Score an investment lead based on qualification signals."""
    score = 0

    if lead.said_not_interested:
        lead.score = 0
        lead.category = LeadCategory.DNC
        return lead

    # Intent signals
    if lead.ready_to_buy:
        score += 3
    if lead.asked_about_specific_plan:
        score += 3

    # Financial goal clarity
    if lead.financial_goal:
        score += 2

    # Has money to invest
    if lead.investable_surplus and lead.investable_surplus >= 5000:
        score += 2
    elif lead.investable_surplus and lead.investable_surplus >= 1000:
        score += 1

    # Demographics
    if lead.age and 25 <= lead.age <= 45:
        score += 1
    if lead.current_investments in ("FD", "savings", "gold", "none"):
        score += 1  # Room to optimize

    # Engagement
    if lead.mentioned_urgency:
        score += 2
    if lead.asked_for_callback:
        score += 1

    lead.score = min(score, 10)

    if lead.score >= 7:
        lead.category = LeadCategory.HOT
    elif lead.score >= 4:
        lead.category = LeadCategory.WARM
    else:
        lead.category = LeadCategory.COLD

    return lead


def score_lead(lead: LeadData) -> LeadData:
    """Route to the correct scoring function based on bot type."""
    if lead.bot_type == BotType.INSURANCE:
        return score_insurance_lead(lead)
    elif lead.bot_type == BotType.INVESTMENT:
        return score_investment_lead(lead)
    else:
        return lead  # Reminder bot doesn't score leads
