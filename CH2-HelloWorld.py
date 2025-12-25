"""
Chapter 2 appendix: this mirrors the Langflow "Hello World" chatbot by wiring
the same nodes in pure LangChain code—a system prompt plus a chat loop that
lets you speak with a pirate persona from your terminal. Use it to compare the
visual build with the equivalent Python, then extend it as you experiment with
model settings later in the chapter. Install deps once:

    pip install langchain-openai langchain-core python-dotenv

Update `.env` with your OpenAI key or paste it inline as shown below.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  #A
from langchain_core.messages import SystemMessage, HumanMessage  #B


load_dotenv()  #C
# os.environ["OPENAI_API_KEY"] = "REPLACE_WITH_YOUR_KEY"

api_key = os.getenv("OPENAI_API_KEY")  #D
if not api_key:  #E
    raise RuntimeError("Set OPENAI_API_KEY before running this script.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)  #F
system_msg = SystemMessage(content="You only answer in Pirate English.")  #G


def chat_loop() -> None:  #H
    """Replicates the Langflow chat node by streaming prompts from stdin."""

    try:
        while True:
            user_input = input("User: ")
            response = llm.invoke([system_msg, HumanMessage(content=user_input)])  #I
            print("AI:", response.content)  #J
    except KeyboardInterrupt:  #K
        print("\nGoodbye, matey!")


if __name__ == "__main__":  #L
    chat_loop()  #M


#A LangChain’s ChatOpenAI is the Python analog to Langflow’s ChatOpenAI block.
#B System/Human message objects reproduce the prompt nodes from the canvas.
#C `load_dotenv` reads the local .env so you don’t copy keys into code.
#D Pull the OpenAI key once the environment is hydrated.
#E Fail fast if the key is missing, mirroring Langflow’s settings check.
#F Instantiate GPT-4o mini with the chapter’s pirate-chat parameters.
#G Keep the pirate persona text as a reusable system prompt.
#H Wrap the chat loop for reuse in tests or other scripts.
#I Invoke the LLM with the ordered system + user messages.
#J Mirror Langflow’s chat UI by streaming the assistant reply to stdout.
#K Catch CTRL-C so the terminal chat ends cleanly.
#L Standard entrypoint guard so imports stay side-effect free.
#M Start the chat loop when running the file directly.
