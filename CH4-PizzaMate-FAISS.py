"""
Chapter 4 appendix: PizzaMate™, the mini-RAG from Langflow, rebuilt in Python.
We load the pizzeria PDF, chunk it, embed into a local FAISS index, then answer
questions by retrieving matching menu snippets and feeding them into a guarded
prompt. Install the Langflow-equivalent bits first:

    pip install pypdf faiss-cpu langchain-openai langchain-community python-dotenv

Update `.env` with your OpenAI key or paste it inline as shown below.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  #A
from langchain_community.vectorstores import FAISS  #B
from langchain_community.document_loaders import PyPDFLoader  #C
from langchain.text_splitter import RecursiveCharacterTextSplitter  #D
from langchain_core.prompts import PromptTemplate  #E

BASE_DIR = Path(__file__).resolve().parent
MENU_PDF = BASE_DIR / "Lu Sule Pizzeria - Info and Menu.pdf"
INDEX_DIR = BASE_DIR / "menu"


def ensure_vectorstore(pdf_path: Path, index_dir: Path) -> FAISS:  #G
    """Load the saved FAISS index or rebuild it from the PDF like the Langflow loader."""

    embeddings = OpenAIEmbeddings()

    if index_dir.exists():
        return FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(index_dir))
    return vectorstore


def build_prompt() -> PromptTemplate:  #H
    """Composite RAG prompt mirroring the Langflow PromptTemplate block."""

    return PromptTemplate(
        input_variables=["chunks", "message"],
        template=(
            "Answer the user query by using only the provided context.\n"
            "Context:\n{chunks}\n\nUser: {message}"
        ),
    )


def answer_query(question: str) -> str:  #I
    """Retrieve menu chunks then ask the LLM to ground the answer in that context."""

    docs = RETRIEVER.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.format(chunks=context, message=question)
    response = LLM.invoke(final_prompt)
    return response.content


load_dotenv()  #F
# os.environ["OPENAI_API_KEY"] = "REPLACE_WITH_YOUR_KEY"

if not os.getenv("OPENAI_API_KEY"):  #J
    raise RuntimeError("Set OPENAI_API_KEY before running this script.")

VECTORSTORE = ensure_vectorstore(MENU_PDF, INDEX_DIR)
RETRIEVER = VECTORSTORE.as_retriever(search_kwargs={"k": 4})  #K
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)  #L
PROMPT = build_prompt()  #M


def chat_loop() -> None:  #N
    """Terminal REPL that mirrors the PizzaMate Langflow chat output."""

    try:
        while True:
            user_question = input("Your PizzaMate question: ")
            answer = answer_query(user_question)
            print("\nAI:", answer, "\n")
    except KeyboardInterrupt:
        print("\nEnjoy your slice!")


if __name__ == "__main__":  #O
    chat_loop()  #P


#A Chat + embedding models used by Langflow.
#B Local FAISS store replaces the Langflow vector-store block.
#C PDF loader feeds the ingestion pipeline.
#D Same chunking strategy as the Langflow splitter node.
#E PromptTemplate reproduces the context-aware prompt node.
#F `load_dotenv` pulls secrets from `.env` before we touch the API.
#G Build or load the FAISS index, mirroring Langflow’s ingestion step.
#H Assemble the RAG prompt text once for reuse.
#I Single function for retrieve+generate so it’s easy to test or extend.
#J Fail fast when the OpenAI key is missing.
#K Create a retriever that returns the top 4 chunks per query.
#L GPT-4o mini handles the grounded responses.
#M Store the compiled prompt for repeated formatting.
#N Chat loop so you can interrogate the menu from the terminal.
#O Standard entrypoint guard.
#P Start the chat loop when running the file directly.
