"""
Kalpvruksh Finserv — Insurance Bot (Aarav)
Handles insurance queries, lead qualification, and outbound pitches.
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
    INSURANCE_SYSTEM_PROMPT = config.load_prompt(config.INSURANCE_BOT_PROMPT)
except FileNotFoundError:
    INSURANCE_SYSTEM_PROMPT = "You are Aarav, an insurance advisor at Kalpvruksh Finserv."
    logger.warning("Insurance bot prompt file not found, using fallback.")

# LLM Client
if config.LLM_PROVIDER == "groq":
    from groq import Groq
    _llm_client = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None
elif config.LLM_PROVIDER == "openai":
    from openai import OpenAI
    _llm_client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
else:
    _llm_client = None


# Tool definitions for function calling
INSURANCE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "forward_to_manager",
            "description": "Log a hot lead to the manager's Google Sheet and send a WhatsApp notification for callback. Use when lead score >= 7 or customer explicitly wants to buy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string", "description": "Full name of the customer"},
                    "phone": {"type": "string", "description": "Customer's phone number"},
                    "age": {"type": "integer", "description": "Customer's age"},
                    "family_members": {"type": "integer", "description": "Number of family members"},
                    "currently_insured": {"type": "boolean", "description": "Whether they have existing insurance"},
                    "interest": {"type": "string", "description": "health/life/both"},
                    "budget_range": {"type": "string", "description": "Approximate annual budget"},
                    "conversation_summary": {"type": "string", "description": "2-3 line summary of the conversation"},
                },
                "required": ["customer_name", "phone", "conversation_summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_nurture",
            "description": "Add a warm lead (score 4-6) to the nurture pipeline for follow-up in 7-30 days.",
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
]


class InsuranceBot:
    """Insurance Bot (Aarav) — handles insurance queries and lead generation."""

    def __init__(self):
        self.conversation_history: dict[str, list] = {}  # session_id -> messages

    def get_or_create_history(self, session_id: str) -> list:
        """Get conversation history for a session, or create a new one."""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = [
                {"role": "system", "content": INSURANCE_SYSTEM_PROMPT}
            ]
        return self.conversation_history[session_id]

    async def handle_message(self, session_id: str, user_message: str,
                              source: LeadSource = LeadSource.INBOUND_WHATSAPP) -> str:
        """
        Process a user message and return the bot's response.
        Handles tool calls for lead forwarding automatically.
        """
        if _llm_client is None:
            return ("Namaste! Main Aarav, Kalpvruksh Finserv se. "
                    "Abhi humare system mein thodi technical issue hai. "
                    "Kya aap thodi der baad try kar sakte hain ya seedha "
                    f"{config.MANAGER_NAME} ji ko call kar sakte hain: {config.MANAGER_PHONE}")

        history = self.get_or_create_history(session_id)
        history.append({"role": "user", "content": user_message})

        try:
            response = _llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=history,
                tools=INSURANCE_TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000,
            )

            message = response.choices[0].message

            # Handle tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    await self._execute_tool(tool_call, source)

                # Get the follow-up response after tool execution
                history.append(message.model_dump())
                history.append({
                    "role": "tool",
                    "tool_call_id": message.tool_calls[0].id,
                    "content": "Lead has been successfully logged and manager has been notified.",
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

            # Keep history manageable (last 20 messages + system prompt)
            if len(history) > 22:
                self.conversation_history[session_id] = [history[0]] + history[-20:]

            return bot_response

        except Exception as e:
            logger.error(f"Insurance bot LLM error: {e}")
            return ("Maaf kijiye, abhi response mein thodi delay aa rahi hai. "
                    f"Kya aap {config.MANAGER_NAME} ji ko seedha call kar sakte hain: {config.MANAGER_PHONE}")

    async def _execute_tool(self, tool_call, source: LeadSource):
        """Execute a tool call from the LLM."""
        func_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tool arguments: {tool_call.function.arguments}")
            return

        if func_name == "forward_to_manager":
            lead = LeadData(
                name=args.get("customer_name", "Unknown"),
                phone=args.get("phone", ""),
                age=args.get("age"),
                family_members=args.get("family_members", 1),
                currently_insured=args.get("currently_insured", False),
                insurance_interest=args.get("interest", "health"),
                conversation_summary=args.get("conversation_summary", ""),
                source=source,
                bot_type=BotType.INSURANCE,
                ready_to_buy=True,
                asked_about_premium=True,
            )
            lead = score_lead(lead)
            sheets_manager.log_hot_lead(lead)
            await whatsapp_notifier.notify_manager_hot_lead(lead)
            logger.info(f"🔴 HOT LEAD forwarded: {lead.name} (Score: {lead.score})")

        elif func_name == "add_to_nurture":
            lead = LeadData(
                name=args.get("customer_name", "Unknown"),
                phone=args.get("phone", ""),
                insurance_interest=args.get("interest", ""),
                conversation_summary=args.get("notes", ""),
                source=source,
                bot_type=BotType.INSURANCE,
            )
            lead = score_lead(lead)
            sheets_manager.log_nurture_lead(lead)
            logger.info(f"🟡 WARM LEAD added to nurture: {lead.name}")

    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        self.conversation_history.pop(session_id, None)


# Singleton
insurance_bot = InsuranceBot()
