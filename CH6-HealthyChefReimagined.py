"""
Chapter 6 appendix: HealthyChef™ evolves into a sequential multi-agent duo.
A Chef agent performs cookbook-grounded RAG over Maria Gentile’s 1919 text,
then hands the recipe to a Dietitian agent that revises it with the NHS
EatWell guide fetched live from https://tiny.cc/eatwellguide. Install once:

    pip install langchain langchain-openai langchain-community \
        langchain-text-splitters faiss-cpu requests python-dotenv

Keep `recipe-book.txt` next to this file and update `.env` with your key or
paste it inline as shown below.
"""

from pathlib import Path

import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings  #A
from langchain_community.vectorstores import FAISS  #B
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  #C
from langchain.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()
# os.environ["OPENAI_API_KEY"] = "REPLACE_WITH_YOUR_KEY"

BASE_DIR = Path(__file__).resolve().parent
COOKBOOK_TXT = BASE_DIR / "recipe-book.txt"
INDEX_DIR = BASE_DIR / "cookbook"
EATWELL_URL = "https://tiny.cc/eatwellguide"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; HealthyChef/1.0)"}


def load_vectorstore() -> FAISS:
    """Build or load the local FAISS index backing the cookbook search."""

    embeddings = OpenAIEmbeddings()

    if INDEX_DIR.exists():
        return FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )

    loader = TextLoader(str(COOKBOOK_TXT), encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(INDEX_DIR))
    return store


VECTORSTORE = load_vectorstore()
RETRIEVER = VECTORSTORE.as_retriever(search_kwargs={"k": 4})
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


@tool
def cookbook_search(query: str) -> str:
    """Retrieve cookbook passages tied to the user request."""

    docs = RETRIEVER.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)  #D


@tool
def fetch_url(url: str | None = None) -> str:
    """Fetch a short excerpt from the NHS EatWell guide."""

    target = url or EATWELL_URL
    try:
        response = requests.get(target, timeout=20, headers=HTTP_HEADERS)
        response.raise_for_status()
        return response.text[:4000]
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Unable to reach the EatWell guide at {target}: {exc}"
        ) from exc  #E


chef_tools = [cookbook_search]
dietitian_tools = [fetch_url]

chef_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an Italian chef limited to The Italian Cook Book by Maria "
            "Gentile (1919). Use the cookbook search tool and return recipes "
            "exactly as written, formatted as Title, Ingredients, Preparation.",
        ),  #F
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

dietitian_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Registered Dietitian using the NHS EatWell guide "
            "(https://tiny.cc/eatwellguide). Always fetch it before responding "
            "and return Title, Ingredients, Preparation, Dietitian's Notes that "
            "justify each change with EatWell principles.",
        ),  #G
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

chef_agent = create_tool_calling_agent(LLM, chef_tools, chef_prompt)
chef_executor = AgentExecutor(agent=chef_agent, tools=chef_tools, verbose=False)

dietitian_agent = create_tool_calling_agent(LLM, dietitian_tools, dietitian_prompt)
dietitian_executor = AgentExecutor(
    agent=dietitian_agent, tools=dietitian_tools, verbose=False
)


def chat_loop() -> None:
    """Sequentially run the Chef then Dietitian agents like the Langflow build."""

    try:
        while True:
            user_input = input("Dish idea or ingredient list: ")
            chef_result = chef_executor.invoke({"input": user_input})
            dietitian_result = dietitian_executor.invoke({"input": chef_result["output"]})

            print("\n--- Chef's Recipe ---\n")
            print(chef_result["output"])
            print("\n--- Dietitian's Revision ---\n")
            print(dietitian_result["output"], "\n")
    except KeyboardInterrupt:
        print("\nBuon appetito!")


if __name__ == "__main__":
    chat_loop()


#A Chat + embedding wrappers reused from earlier chapters.
#B Local FAISS vector store that powers the cookbook retriever.
#C Prompt pieces with scratchpads so each agent can call its tools.
#D Tool: semantic lookup over Maria Gentile’s cookbook.
#E Tool: fetch the EatWell guide (defaults to https://tiny.cc/eatwellguide).
#F Chef agent instructions mirroring the Langflow prompt.
#G Dietitian agent instructions grounded in the EatWell guidance.
