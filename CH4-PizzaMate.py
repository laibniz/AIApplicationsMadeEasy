"""
Chapter 4 appendix (ChromaDB edition): PizzaMate’s PDF-to-RAG recipe rebuilt
with LangChain + Chroma. We ingest the Lu Sule Pizzeria PDF, chunk it, embed
with OpenAI, and store vectors inside a local Chroma collection before answering
menu questions through a retrieval-augmented prompt.

Install once:

    pip install pypdf chromadb langchain-openai langchain-community \
        langchain-text-splitters python-dotenv

Keep `Lu Sule Pizzeria - Info and Menu.pdf` next to this script, update `.env`
with your OpenAI key (or uncomment the placeholder), and the vector store will
live under a `menu_chroma/` folder.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  #A
from langchain_community.vectorstores import Chroma  #B
from langchain_community.document_loaders import PyPDFLoader  #C
from langchain_text_splitters import RecursiveCharacterTextSplitter  #D
from langchain_core.prompts import PromptTemplate  #E

BASE_DIR = Path(__file__).resolve().parent
MENU_PDF = BASE_DIR / "Lu Sule Pizzeria - Info and Menu.pdf"
INDEX_DIR = BASE_DIR / "menu_chroma"
COLLECTION_NAME = "pizza_menu"

load_dotenv()
# os.environ["OPENAI_API_KEY"] = "REPLACE_WITH_YOUR_KEY"


def ensure_vectorstore(pdf_path: Path, index_dir: Path) -> Chroma:
    """Load or build the Chroma collection mirroring Langflow’s ingestion."""

    embeddings = OpenAIEmbeddings()

    if index_dir.exists():
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(index_dir),
        )

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(index_dir),
    )
    return vectorstore


def build_prompt() -> PromptTemplate:
    """Context-aware RAG prompt identical to the Langflow PromptTemplate."""

    return PromptTemplate(
        input_variables=["chunks", "message"],
        template=(
            "Answer the user query by using only the provided context.\n"
            "Context:\n{chunks}\n\nUser: {message}"
        ),
    )


VECTORSTORE = ensure_vectorstore(MENU_PDF, INDEX_DIR)
RETRIEVER = VECTORSTORE.as_retriever(search_kwargs={"k": 4})  #F
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)  #G
PROMPT = build_prompt()


def answer_query(question: str) -> str:  #H
    """Retrieve Chroma chunks then ground the final answer in that context."""

    docs = RETRIEVER.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.format(chunks=context, message=question)
    response = LLM.invoke(final_prompt)
    return response.content


def chat_loop() -> None:
    """Terminal REPL so you can interrogate the menu like the Langflow chat."""

    try:
        while True:
            user_question = input("Your PizzaMate question: ")
            answer = answer_query(user_question)
            print("\nAI:", answer, "\n")
    except KeyboardInterrupt:
        print("\nEnjoy your slice!")


if __name__ == "__main__":
    chat_loop()


#A Chat + embedding wrappers shared across the PizzaMate builds.
#B Chroma vector store replaces the Langflow Chroma component.
#C PDF loader feeds the ingestion pipeline.
#D Same chunking strategy as the Langflow splitter node.
#E PromptTemplate reproduces the context-aware prompt block.
#F Retriever pulls the top 4 chunks per user query.
#G GPT-4o mini generates grounded answers.
#H Single helper encapsulating retrieve + prompt + generate.
