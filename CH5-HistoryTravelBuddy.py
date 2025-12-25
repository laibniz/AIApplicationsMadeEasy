"""
Chapter 5 appendix: HistoryTravelBuddy™ becomes a LangChain agent that matches
the Langflow tutorial—searching SerpAPI for attractions, loading articles, and
querying Wikipedia so every itinerary turn gains historical color. Install:

    pip install langchain langchain-openai langchain-community \
        google-search-results wikipedia requests python-dotenv

Update `.env` with OPENAI_API_KEY and SERP_API_KEY or paste them inline.
"""

import os
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI  #A
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  #B
from langchain_community.utilities import SerpAPIWrapper, WikipediaAPIWrapper  #C
from langchain_community.tools import WikipediaQueryRun
from langchain.agents import AgentExecutor, create_tool_calling_agent  #D

load_dotenv()
# os.environ["OPENAI_API_KEY"] = "REPLACE_WITH_YOUR_KEY"
# os.environ["SERP_API_KEY"] = "REPLACE_WITH_YOUR_KEY"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERP_API_KEY"))
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())  #E


@tool
def web_search(query: str) -> str:
    """Search the web for historical attractions and day-trip ideas."""
    return search.run(query)  #F


@tool
def fetch_url(url: str) -> str:
    """Load and retrieve data from specified URLs."""
    text = requests.get(url, timeout=15).text
    return text[:2000]  #G


tools = [web_search, fetch_url, wiki_tool]  #H

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are HistoryTravelBuddy, a helpful assistant that can use tools "
            "to answer questions and perform tasks. You act as an agent, and your "
            "main task is to look for top historical attractions in the city that "
            "the user would like to visit. Open the webpages describing them and "
            "compile a one-day itinerary. Check on Wikipedia for historical facts "
            "about these places and enrich the itinerary with one historical "
            "paragraph for each. Add the link to Wikipedia pages for each "
            "interesting fact you talk about.",
        ),  #I
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)  #J
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)  #K


def chat_loop() -> None:  #L
    """Simple REPL to keep asking the agent for day-trip plans."""

    try:
        while True:
            user_input = input("User: ")
            result = agent_executor.invoke({"input": user_input})
            print("\nAI:", result["output"], "\n")
    except KeyboardInterrupt:
        print("\nSafe travels!")


if __name__ == "__main__":  #M
    chat_loop()  #N


#A Chat model driving the agent.
#B Prompt building blocks with placeholders.
#C Tool helper clients (SerpAPI/Wikipedia).
#D Agent factory + executor utilities.
#E Ready-made Wikipedia search+read tool.
#F Tool: search the web.
#G Tool: fetch webpage text safely.
#H Tools list passed to the agent constructor.
#I System prompt that guides HistoryTravelBuddy’s behavior.
#J Build the tool-aware agent graph.
#K Wrap it with an executor for invoke().
#L Helper loop to repeatedly invoke the agent from the terminal.
#M Standard entrypoint guard.
#N Start the chat loop when run directly.
