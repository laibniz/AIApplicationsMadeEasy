"""
Chapter 6 appendix: CopyGenie™ mirrors the Langflow hierarchical agent team—
Planner -> Copywriter -> Delivery—coordinated by a Supervisor Agent that
calls them via OpenAI tool-calling. Install once:

    pip install langchain langchain-openai langchain-community \
        docx2txt python-dotenv composio composio-openai

Place `Social_Media_Calendar_2025_2027.csv` and `Brand Guidelines.docx` next
to this script, then set OPENAI_API_KEY, COMPOSIO_API_KEY, COMPOSIO_USER_ID,
and COMPOSIO_AUTH_CONFIG_ID in `.env` (or uncomment the placeholders below).
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import docx2txt
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI  #A
from langchain_core.tools import tool  #B
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from composio import Composio  #C
from composio_openai import OpenAIProvider

load_dotenv()
# os.environ["OPENAI_API_KEY"] = "REPLACE_WITH_YOUR_KEY"
# os.environ["COMPOSIO_API_KEY"] = "REPLACE_WITH_YOUR_KEY"
# os.environ["COMPOSIO_USER_ID"] = "REPLACE_WITH_YOUR_USER_ID"
# os.environ["COMPOSIO_AUTH_CONFIG_ID"] = "REPLACE_WITH_YOUR_AUTH_ID"

BASE_DIR = Path(__file__).resolve().parent
CALENDAR_FILE = BASE_DIR / "Social_Media_Calendar_2025_2027.csv"
BRAND_FILE = BASE_DIR / "Brand Guidelines.docx"

calendar_text = CALENDAR_FILE.read_text(encoding="utf-8")  #D
brand_text = docx2txt.process(str(BRAND_FILE))

OPENAI_MODEL = "gpt-4.1-mini"
base_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)

COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
COMPOSIO_USER_ID = os.getenv("COMPOSIO_USER_ID")
COMPOSIO_AUTH_CONFIG_ID = os.getenv("COMPOSIO_AUTH_CONFIG_ID")

if not all([COMPOSIO_API_KEY, COMPOSIO_USER_ID, COMPOSIO_AUTH_CONFIG_ID]):
    raise RuntimeError("Set COMPOSIO_* variables before running CopyGenie.")

composio_client = Composio(
    api_key=COMPOSIO_API_KEY,
    provider=OpenAIProvider(),
    dangerously_skip_version_check=True,
)  #E

print("Authorize CopyGenie for Gmail via Composio:")
connection_request = composio_client.connected_accounts.link(
    user_id=COMPOSIO_USER_ID,
    auth_config_id=COMPOSIO_AUTH_CONFIG_ID,
)
print(connection_request.redirect_url)
print("Waiting for Gmail connection...")
connection_request.wait_for_connection()
print("Gmail connection established.\n")


PLANNER_SYSTEM = """
You are Planner Agent for CopyGenie’s social media team.
- Use this calendar to find events and space posts evenly:
  ##{calendar}##
- Default to 4 posts per channel per month unless the user specifies otherwise.
- Output lines like: 11 September: Post about Zero Emissions Day on LinkedIn.
""".strip()

planner_prompt = ChatPromptTemplate.from_messages(
    [("system", PLANNER_SYSTEM), ("human", "{request}")]
)


@tool
def planner_agent(request: str) -> str:
    """Generates a dated content plan using the social media calendar."""

    messages = planner_prompt.format_messages(calendar=calendar_text, request=request)
    return base_llm.invoke(messages).content.strip()  #F


COPYWRITER_SYSTEM = """
You are Copywriter Agent for Sol Vigna Wines.
- Follow these brand guidelines: ##{brand_guidelines}##
- Write ONE post per entry from the planner, referencing date/event/channel.
""".strip()

copywriter_prompt = ChatPromptTemplate.from_messages(
    [("system", COPYWRITER_SYSTEM), ("human", "{plan}")]
)


@tool
def copywriter_agent(plan: str) -> str:
    """Drafts branded posts based on the planner’s schedule."""

    messages = copywriter_prompt.format_messages(
        brand_guidelines=brand_text,
        plan=plan,
    )
    return base_llm.invoke(messages).content.strip()  #G


@tool
def delivery_agent(recipient: str, subject: str, body: str) -> str:
    """Sends finalized posts via Gmail through Composio."""

    if not (recipient and subject and body):
        return "Delivery agent needs recipient, subject, and body."  #H

    result = composio_client.tools.execute(
        slug="gmail_send_email",
        arguments={"to": recipient, "subject": subject, "body": body},
        user_id=COMPOSIO_USER_ID,
        dangerously_skip_version_check=True,
    )
    return f"Composio send result: {result}"


SUPERVISOR_SYSTEM = """
You are CopyGenie, supervising planner_agent, copywriter_agent, delivery_agent.
- Always call planner_agent to read the calendar when users ask about posts.
- Pass planner output to copywriter_agent for branded posts.
- Call delivery_agent only after posts are approved and you have recipient,
  subject, and body. Keep working until the request is fully satisfied.
""".strip()

tools = [planner_agent, copywriter_agent, delivery_agent]
tool_lookup: Dict[str, Any] = {t.name: t for t in tools}
supervisor_model = base_llm.bind_tools(tools)  #I


def run_interactive_chat() -> None:
    """Simple REPL mirroring the Langflow supervisor playground."""

    messages: List[BaseMessage] = [SystemMessage(content=SUPERVISOR_SYSTEM)]

    try:
        while True:
            user_input = input("User: ")
            if not user_input.strip():
                continue

            messages.append(HumanMessage(content=user_input))

            while True:
                ai_msg = supervisor_model.invoke(messages)
                messages.append(ai_msg)

                tool_calls = getattr(ai_msg, "tool_calls", None) or []
                if not tool_calls:
                    print("\nAI:", ai_msg.content, "\n")
                    break

                for tc in tool_calls:
                    tool = tool_lookup[tc["name"]]
                    result = tool.invoke(tc["args"])
                    messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                    )
    except KeyboardInterrupt:
        print("\nCopyGenie signing off!")


if __name__ == "__main__":
    run_interactive_chat()


#A Chat model powering all agents and the supervisor.
#B LangChain @tool decorator wraps each specialist.
#C Composio client provides Gmail access.
#D Load the shared calendar + brand guidelines.
#E Link CopyGenie to Gmail before the supervisor starts.
#F Planner agent mirrors the Langflow calendar node.
#G Copywriter agent enforces Sol Vigna’s tone per channel.
#H Delivery agent requires full email details before sending.
#I Bind the three tools so the supervisor can call them dynamically.
