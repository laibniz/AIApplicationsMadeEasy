"""
Chapter 3 appendix: extends the pirate chatbot by mirroring the Langflow Prompt
Template + Message History blocks, so every turn injects the running dialog
before the latest user question. Use this version to see how lightweight memory
feels in Python before you move on to richer chains later in the chapter.
Install deps once:

    pip install langchain-openai langchain-core python-dotenv

Update `.env` with your OpenAI key or paste it inline as shown below.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  #A
from langchain_core.prompts import PromptTemplate  #B
from langchain_core.messages import SystemMessage, HumanMessage  #C


load_dotenv()  #D
# os.environ["OPENAI_API_KEY"] = "REPLACE_WITH_YOUR_KEY"

if not os.getenv("OPENAI_API_KEY"):  #E
    raise RuntimeError("Set OPENAI_API_KEY before running this script.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)  #F
system_msg = SystemMessage(content="Always reply in Pirate English.")  #G

prompt = PromptTemplate(
    input_variables=["message", "history"],
    template="{history}\nUser: {message}\nAI: ",
)  #H


def chat_loop() -> None:  #I
    """Chat loop with manual history injection, matching Langflow’s wiring."""

    history_lines: list[str] = []

    try:
        while True:
            history_block = "\n".join(history_lines)
            user_input = input("User: ")
            rendered_prompt = prompt.format(message=user_input, history=history_block)

            response = llm.invoke(
                [system_msg, HumanMessage(content=rendered_prompt)]
            )  #J
            print("AI:", response.content)  #K

            history_lines.extend(
                [f"User: {user_input}", f"AI: {response.content}"]
            )  #L
    except KeyboardInterrupt:
        print("\nGoodbye, matey!")


if __name__ == "__main__":  #M
    chat_loop()


#A Same ChatOpenAI wrapper from Chapter 2, reused without extra ceremony.
#B PromptTemplate mirrors Langflow’s prompt block.
#C System/Human message objects remain our LangChain building blocks.
#D `load_dotenv` brings environment keys into the process automatically.
#E Fail fast if the key is still missing.
#F Instantiate GPT-4o mini with the chapter’s suggested settings.
#G Keep the pirate system persona consistent with the Langflow canvas.
#H Composite prompt that injects `{history}` before the latest user turn.
#I Bundle the chat loop so we can reuse/import it later in the chapter.
#J Invoke the LLM with the rendered prompt to emulate the Prompt+History nodes.
#K Stream responses to stdout, same as the Langflow chat UI.
#L Track alternating “User:/AI:” lines so the next turn has full context.
#M Standard entrypoint guard.
