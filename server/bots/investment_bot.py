"""
Kalpvruksh Finserv — Investment Bot (Riya)
Handles investment queries, SIP pitches, and lead conversion.
"""

import json
import logging
from typing import Optional

from server.config import config
from server.lead_scoring import LeadData, BotType, LeadSource, score_lead
from server.sheets_manager import sheets_manager, whatsapp_notifier

logger = logging.getLogger(__name__)

# Load system prompt
try:
    INVESTMENT_SYSTEM_PROMPT = config.load_prompt(config.INVESTMENT_BOT_PROMPT)
except FileNotFoundError:
    INVESTMENT_SYSTEM_PROMPT = "You are Riya, a wealth management advisor at Kalpvruksh Finserv."
    logger.warning("Investment bot prompt file not found, using fallback.")

# LLM Client
if config.LLM_PROVIDER == "groq":
    from groq import Groq
    _llm_client = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None
elif config.LLM_PROVIDER == "openai":
    from openai import OpenAI
    _llm_client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
else:
    _llm_client = None


# Tool definitions
INVESTMENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "schedule_investment_consultation",
            "description": "Schedule a consultation with Sanjeev sir for a hot investment lead. Use when the customer wants to invest or see a detailed plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string"},
                    "phone": {"type": "string"},
                    "financial_goal": {"type": "string", "description": "education/retirement/wealth/tax"},
                    "monthly_surplus": {"type": "string", "description": "Monthly investable amount in INR"},
                    "current_investments": {"type": "string", "description": "FD/gold/MF/none"},
                    "risk_appetite": {"type": "string", "description": "conservative/moderate/aggressive"},
                    "time_horizon_years": {"type": "string"},
                    "conversation_summary": {"type": "string"},
                },
                "required": ["customer_name", "phone", "conversation_summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_nurture",
            "description": "Add a warm lead to the nurture pipeline for educational content drip.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string"},
                    "phone": {"type": "string"},
                    "interest": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["customer_name", "phone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sip_goal",
            "description": "Calculate the monthly SIP amount needed to reach a financial goal. Use when customer mentions a specific goal amount and timeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_amount": {"type": "number", "description": "Target amount in INR"},
                    "years": {"type": "integer", "description": "Investment horizon in years"},
                    "expected_return": {"type": "number", "description": "Expected annual return (e.g., 12 for 12%)"},
                },
                "required": ["goal_amount", "years"],
            },
        },
    },
]


def calculate_sip_amount(goal_amount: float, years: int, annual_return: float = 12.0) -> dict:
    """Calculate monthly SIP needed for a goal using the standard formula."""
    monthly_rate = annual_return / 100 / 12
    months = years * 12

    if monthly_rate == 0:
        sip_needed = goal_amount / months
    else:
        # FV = SIP * [(1+r)^n - 1] / r * (1+r)
        # SIP = FV * r / [(1+r)^n - 1] / (1+r)
        factor = ((1 + monthly_rate) ** months - 1) / monthly_rate * (1 + monthly_rate)
        sip_needed = goal_amount / factor

    return {
        "goal_amount": goal_amount,
        "years": years,
        "expected_return_percent": annual_return,
        "monthly_sip_needed": round(sip_needed, 0),
        "total_invested": round(sip_needed * months, 0),
        "wealth_gained": round(goal_amount - (sip_needed * months), 0),
    }


class InvestmentBot:
    """Investment Bot (Riya) — handles investment queries and conversions."""

    def __init__(self):
        self.conversation_history: dict[str, list] = {}

    def get_or_create_history(self, session_id: str) -> list:
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = [
                {"role": "system", "content": INVESTMENT_SYSTEM_PROMPT}
            ]
        return self.conversation_history[session_id]

    async def handle_message(self, session_id: str, user_message: str,
                              source: LeadSource = LeadSource.INBOUND_WHATSAPP) -> str:
        """Process a user message and return the bot's response."""
        if _llm_client is None:
            return ("Namaste! Main Riya, Kalpvruksh Finserv se. "
                    "Abhi humare system mein thodi issue hai. "
                    f"Kya aap {config.MANAGER_NAME} ji ko call kar sakte hain: {config.MANAGER_PHONE}")

        history = self.get_or_create_history(session_id)
        history.append({"role": "user", "content": user_message})

        try:
            response = _llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=history,
                tools=INVESTMENT_TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000,
            )

            message = response.choices[0].message

            # Handle tool calls
            if message.tool_calls:
                tool_results = []
                for tool_call in message.tool_calls:
                    result = await self._execute_tool(tool_call, source)
                    tool_results.append(result)

                history.append(message.model_dump())

                for i, tool_call in enumerate(message.tool_calls):
                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_results[i]) if isinstance(tool_results[i], dict) else str(tool_results[i]),
                    })

                followup = _llm_client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=history,
                    temperature=0.7,
                    max_tokens=500,
                )
                bot_response = followup.choices[0].message.content
            else:
                bot_response = message.content

            history.append({"role": "assistant", "content": bot_response})

            if len(history) > 22:
                self.conversation_history[session_id] = [history[0]] + history[-20:]

            return bot_response

        except Exception as e:
            logger.error(f"Investment bot LLM error: {e}")
            return ("Maaf kijiye, thodi technical difficulty aa rahi hai. "
                    f"{config.MANAGER_NAME} ji aapki zaroor help karenge: {config.MANAGER_PHONE}")

    async def _execute_tool(self, tool_call, source: LeadSource):
        """Execute a tool call."""
        func_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tool arguments: {tool_call.function.arguments}")
            return "Error processing request."

        if func_name == "schedule_investment_consultation":
            lead = LeadData(
                name=args.get("customer_name", "Unknown"),
                phone=args.get("phone", ""),
                financial_goal=args.get("financial_goal", ""),
                investable_surplus=args.get("monthly_surplus"),
                current_investments=args.get("current_investments", ""),
                risk_appetite=args.get("risk_appetite", ""),
                time_horizon_years=args.get("time_horizon_years"),
                conversation_summary=args.get("conversation_summary", ""),
                source=source,
                bot_type=BotType.INVESTMENT,
                ready_to_buy=True,
            )
            lead = score_lead(lead)
            sheets_manager.log_hot_lead(lead)
            await whatsapp_notifier.notify_manager_hot_lead(lead)
            logger.info(f"🔴 HOT INVESTMENT LEAD: {lead.name} (Score: {lead.score})")
            return "Consultation scheduled. Manager notified."

        elif func_name == "add_to_nurture":
            lead = LeadData(
                name=args.get("customer_name", "Unknown"),
                phone=args.get("phone", ""),
                financial_goal=args.get("interest", ""),
                conversation_summary=args.get("notes", ""),
                source=source,
                bot_type=BotType.INVESTMENT,
            )
            lead = score_lead(lead)
            sheets_manager.log_nurture_lead(lead)
            logger.info(f"🟡 WARM INVESTMENT LEAD: {lead.name}")
            return "Lead added to nurture pipeline."

        elif func_name == "calculate_sip_goal":
            result = calculate_sip_amount(
                goal_amount=args["goal_amount"],
                years=args["years"],
                annual_return=args.get("expected_return", 12.0),
            )
            logger.info(f"SIP calculated: ₹{result['monthly_sip_needed']}/month for ₹{result['goal_amount']} in {result['years']} years")
            return result

        return "Unknown tool."

    def clear_session(self, session_id: str):
        self.conversation_history.pop(session_id, None)


# Singleton
investment_bot = InvestmentBot()
