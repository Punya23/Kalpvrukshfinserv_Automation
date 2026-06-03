"""
Kalpvruksh Finserv — Reminder Bot (Vikram)
Handles renewal reminders, portfolio status checks, and customer servicing.
"""

import json
import logging
from typing import Optional

from server.config import config
from server.lead_scoring import LeadData, BotType, LeadSource
from server.sheets_manager import sheets_manager, whatsapp_notifier

logger = logging.getLogger(__name__)

# Load system prompt
try:
    REMINDER_SYSTEM_PROMPT = config.load_prompt(config.REMINDER_BOT_PROMPT)
except FileNotFoundError:
    REMINDER_SYSTEM_PROMPT = "You are Vikram, a customer success manager at Kalpvruksh Finserv."
    logger.warning("Reminder bot prompt file not found, using fallback.")

# LLM Client
if config.LLM_PROVIDER == "groq":
    from groq import Groq
    _llm_client = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None
elif config.LLM_PROVIDER == "openai":
    from openai import OpenAI
    _llm_client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
else:
    _llm_client = None


# -------------------------------------------------------
# Sample Customer Database (for testing without real DB)
# In production, this would connect to Supabase/Airtable/CRM
# -------------------------------------------------------
SAMPLE_CUSTOMERS = {
    "KF-001": {
        "id": "KF-001",
        "name": "Rajesh Sharma",
        "phone": "919876543001",
        "age": 42,
        "family": "Wife (38), Son (14), Daughter (10)",
        "policies": [
            {
                "type": "Health Insurance",
                "insurer": "Star Health",
                "plan": "Family Health Optima",
                "policy_number": "SH-HI-2024-001",
                "sum_insured": 1000000,
                "premium": 22000,
                "start_date": "2024-07-01",
                "expiry_date": "2026-07-01",
                "status": "Active",
                "ncb_years": 2,
            }
        ],
        "investments": [
            {
                "type": "SIP",
                "fund": "HDFC Flexi Cap Fund",
                "folio": "1234567/89",
                "sip_amount": 5000,
                "start_date": "2023-01-15",
                "total_invested": 90000,
                "current_value": 108000,
                "return_percent": 20.0,
                "xirr": 16.5,
                "status": "Active",
            },
            {
                "type": "SIP",
                "fund": "ICICI Pru Balanced Advantage",
                "folio": "9876543/21",
                "sip_amount": 3000,
                "start_date": "2023-06-01",
                "total_invested": 36000,
                "current_value": 40500,
                "return_percent": 12.5,
                "xirr": 11.8,
                "status": "Active",
            },
        ],
    },
    "KF-002": {
        "id": "KF-002",
        "name": "Priya Deshmukh",
        "phone": "919876543002",
        "age": 65,
        "family": "Husband (68)",
        "policies": [
            {
                "type": "Health Insurance",
                "insurer": "Star Health",
                "plan": "Senior Citizens Red Carpet",
                "policy_number": "SH-HI-2024-002",
                "sum_insured": 500000,
                "premium": 35000,
                "start_date": "2024-06-15",
                "expiry_date": "2026-06-15",
                "status": "Active",
                "ncb_years": 3,
            }
        ],
        "investments": [],
    },
    "KF-003": {
        "id": "KF-003",
        "name": "Amit Kulkarni",
        "phone": "919876543003",
        "age": 35,
        "family": "Wife (32), Son (5)",
        "policies": [
            {
                "type": "Health Insurance",
                "insurer": "Star Health",
                "plan": "Star Comprehensive",
                "policy_number": "SH-HI-2024-003",
                "sum_insured": 2500000,
                "premium": 48000,
                "start_date": "2024-06-08",
                "expiry_date": "2026-06-08",
                "status": "Active",
                "ncb_years": 1,
            },
            {
                "type": "Term Insurance",
                "insurer": "Max Life",
                "plan": "Smart Secure Plus",
                "policy_number": "ML-TI-2023-001",
                "sum_insured": 10000000,
                "premium": 12500,
                "start_date": "2023-03-15",
                "expiry_date": "2053-03-15",
                "status": "Active",
            },
        ],
        "investments": [
            {
                "type": "SIP",
                "fund": "SBI Small Cap Fund",
                "folio": "5555666/77",
                "sip_amount": 10000,
                "start_date": "2022-04-01",
                "total_invested": 500000,
                "current_value": 685000,
                "return_percent": 37.0,
                "xirr": 22.3,
                "status": "Active",
            },
            {
                "type": "SIP",
                "fund": "Axis ELSS Tax Saver",
                "folio": "8888999/00",
                "sip_amount": 5000,
                "start_date": "2023-01-01",
                "total_invested": 90000,
                "current_value": 102000,
                "return_percent": 13.3,
                "xirr": 14.1,
                "status": "Active",
            },
        ],
    },
}

# Map phone numbers to customer IDs for lookup
PHONE_TO_CUSTOMER = {v["phone"]: k for k, v in SAMPLE_CUSTOMERS.items()}


def lookup_customer(identifier: str) -> Optional[dict]:
    """
    Look up a customer by ID (KF-XXXX) or phone number.
    In production, this would query the CRM/database.
    """
    # Try as Customer ID first
    if identifier.upper().startswith("KF-"):
        return SAMPLE_CUSTOMERS.get(identifier.upper())

    # Try as phone number (strip +91, spaces, etc.)
    clean_phone = identifier.replace("+", "").replace(" ", "").replace("-", "")
    if not clean_phone.startswith("91"):
        clean_phone = "91" + clean_phone

    customer_id = PHONE_TO_CUSTOMER.get(clean_phone)
    if customer_id:
        return SAMPLE_CUSTOMERS.get(customer_id)

    return None


# Tool definitions
REMINDER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_portfolio_status",
            "description": "Fetch the complete investment portfolio status for a customer using their Customer ID or phone number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "Customer ID (KF-XXXX) or phone number"},
                },
                "required": ["identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_policy_details",
            "description": "Fetch all insurance policy details for a customer including renewal dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "Customer ID or phone number"},
                },
                "required": ["identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "problem_escalation",
            "description": "Escalate an issue to Sanjeev sir. Use for complaints, claim disputes, cancellation requests, or complex queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "customer_name": {"type": "string"},
                    "phone": {"type": "string"},
                    "issue_type": {"type": "string", "description": "renewal/claim/complaint/cross-sell/cancellation"},
                    "urgency": {"type": "string", "description": "normal/high/critical"},
                    "context": {"type": "string", "description": "Full description of the issue"},
                    "recommended_action": {"type": "string"},
                },
                "required": ["customer_name", "issue_type", "context"],
            },
        },
    },
]


class ReminderBot:
    """Reminder Bot (Vikram) — handles renewals, status checks, and customer servicing."""

    def __init__(self):
        self.conversation_history: dict[str, list] = {}

    def _build_system_prompt_with_context(self, customer: Optional[dict] = None) -> str:
        """Build system prompt, injecting customer context if available."""
        base_prompt = REMINDER_SYSTEM_PROMPT

        if customer:
            context = f"\n\n## CURRENT CUSTOMER CONTEXT (loaded from database)\n"
            context += f"- **Customer ID**: {customer['id']}\n"
            context += f"- **Name**: {customer['name']}\n"
            context += f"- **Age**: {customer['age']}\n"
            context += f"- **Family**: {customer['family']}\n"

            if customer.get("policies"):
                context += f"\n### Active Insurance Policies:\n"
                for p in customer["policies"]:
                    context += (f"- {p['plan']} ({p['insurer']}) — "
                               f"Policy No: {p['policy_number']}, "
                               f"SI: ₹{p['sum_insured']:,}, "
                               f"Premium: ₹{p['premium']:,}/year, "
                               f"Expiry: {p['expiry_date']}, "
                               f"Status: {p['status']}\n")

            if customer.get("investments"):
                context += f"\n### Active Investments:\n"
                total_invested = sum(inv["total_invested"] for inv in customer["investments"])
                total_value = sum(inv["current_value"] for inv in customer["investments"])
                context += f"- **Total Invested**: ₹{total_invested:,}\n"
                context += f"- **Current Value**: ₹{total_value:,}\n"
                overall_return = ((total_value - total_invested) / total_invested * 100) if total_invested > 0 else 0
                context += f"- **Overall Return**: {overall_return:.1f}%\n"
                for inv in customer["investments"]:
                    context += (f"- {inv['fund']} — SIP: ₹{inv['sip_amount']:,}/mo, "
                               f"Invested: ₹{inv['total_invested']:,}, "
                               f"Value: ₹{inv['current_value']:,}, "
                               f"Return: {inv['return_percent']}%, "
                               f"XIRR: {inv['xirr']}%\n")

            base_prompt += context

        return base_prompt

    def get_or_create_history(self, session_id: str, customer: Optional[dict] = None) -> list:
        if session_id not in self.conversation_history:
            system_prompt = self._build_system_prompt_with_context(customer)
            self.conversation_history[session_id] = [
                {"role": "system", "content": system_prompt}
            ]
        return self.conversation_history[session_id]

    async def handle_message(self, session_id: str, user_message: str,
                              customer_id: Optional[str] = None,
                              phone: Optional[str] = None) -> str:
        """Process a message from an existing customer."""
        if _llm_client is None:
            return ("Namaste! Main Vikram, Kalpvruksh Finserv se. "
                    "Abhi system mein thodi issue hai. "
                    f"Please {config.MANAGER_NAME} ji ko call karein: {config.MANAGER_PHONE}")

        # Try to look up customer context
        customer = None
        if customer_id:
            customer = lookup_customer(customer_id)
        elif phone:
            customer = lookup_customer(phone)

        history = self.get_or_create_history(session_id, customer)
        history.append({"role": "user", "content": user_message})

        try:
            response = _llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=history,
                tools=REMINDER_TOOLS,
                tool_choice="auto",
                temperature=0.6,
                max_tokens=1000,
            )

            message = response.choices[0].message

            if message.tool_calls:
                tool_results = []
                for tool_call in message.tool_calls:
                    result = await self._execute_tool(tool_call)
                    tool_results.append(result)

                history.append(message.model_dump())

                for i, tool_call in enumerate(message.tool_calls):
                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_results[i], ensure_ascii=False) if isinstance(tool_results[i], dict) else str(tool_results[i]),
                    })

                followup = _llm_client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=history,
                    temperature=0.6,
                    max_tokens=800,
                )
                bot_response = followup.choices[0].message.content
            else:
                bot_response = message.content

            history.append({"role": "assistant", "content": bot_response})

            if len(history) > 22:
                self.conversation_history[session_id] = [history[0]] + history[-20:]

            return bot_response

        except Exception as e:
            logger.error(f"Reminder bot LLM error: {e}")
            return ("Maaf kijiye, thodi issue aa rahi hai. "
                    f"{config.MANAGER_NAME} ji se seedha baat karein: {config.MANAGER_PHONE}")

    async def _execute_tool(self, tool_call):
        """Execute tool calls for the reminder bot."""
        func_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return {"error": "Failed to parse arguments"}

        if func_name == "fetch_portfolio_status":
            customer = lookup_customer(args["identifier"])
            if customer and customer.get("investments"):
                total_invested = sum(inv["total_invested"] for inv in customer["investments"])
                total_value = sum(inv["current_value"] for inv in customer["investments"])
                return {
                    "customer_name": customer["name"],
                    "total_invested": total_invested,
                    "current_value": total_value,
                    "overall_return_percent": round((total_value - total_invested) / total_invested * 100, 1) if total_invested > 0 else 0,
                    "active_sips": len(customer["investments"]),
                    "total_monthly_sip": sum(inv["sip_amount"] for inv in customer["investments"]),
                    "fund_details": customer["investments"],
                }
            return {"error": f"No investment records found for {args['identifier']}"}

        elif func_name == "fetch_policy_details":
            customer = lookup_customer(args["identifier"])
            if customer and customer.get("policies"):
                return {
                    "customer_name": customer["name"],
                    "policies": customer["policies"],
                }
            return {"error": f"No policy records found for {args['identifier']}"}

        elif func_name == "problem_escalation":
            # Log to sheets and notify manager
            escalation_message = (
                f"⚠️ *ESCALATION — {args.get('urgency', 'normal').upper()}*\n\n"
                f"👤 *Customer:* {args.get('customer_name', 'Unknown')}\n"
                f"📋 *Issue:* {args.get('issue_type', 'general')}\n"
                f"💬 *Details:* {args.get('context', '')}\n"
                f"✅ *Recommended Action:* {args.get('recommended_action', 'Call customer')}"
            )
            await whatsapp_notifier._send_whatsapp(config.MANAGER_WHATSAPP_NUMBER, escalation_message)
            logger.info(f"⚠️ ESCALATION: {args.get('issue_type')} for {args.get('customer_name')}")
            return "Escalation logged. Manager has been notified and will contact the customer."

        return {"error": "Unknown tool"}

    def clear_session(self, session_id: str):
        self.conversation_history.pop(session_id, None)


# Singleton
reminder_bot = ReminderBot()
