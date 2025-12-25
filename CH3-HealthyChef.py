"""
Chapter 3 appendix: reproduces the Langflow “Healthy Chef” chain where a Michelin
chef creates a recipe and a Registered Dietitian rewrites it for nutrition. Two
PromptTemplates plus two sequential LLM calls mirror the dual-node pipeline you
assembled visually, so you can tweak either persona directly in code.
Install deps once:

    pip install langchain-openai langchain-core python-dotenv

Update `.env` with your OpenAI key or paste it inline as shown below.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  #A
from langchain_core.prompts import PromptTemplate  #B


load_dotenv()  #C
# os.environ["OPENAI_API_KEY"] = "REPLACE_WITH_YOUR_KEY"

if not os.getenv("OPENAI_API_KEY"):  #D
    raise RuntimeError("Set OPENAI_API_KEY before running this script.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)  #E

chef_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=(
        "Act as a world-renowned Chef. Given the input from the user which might "
        "contain the title of a desired recipe and/or some ingredients available to use:\n\n"
        "{user_input}\n\n"
        "create a detailed recipe following the format:\n\n"
        "TITLE: A recipe title.\n"
        "INGREDIENTS: A list of ingredients with quantities.\n"
        "PREPARATION: Step-by-step preparation instructions.\n\n"
        "Ensure the recipe is creative and appealing. Just generate the recipe using the "
        "given format. If the input is insufficient, output \"Insufficient input\"."
    ),
)  #F

dietitian_prompt = PromptTemplate(
    input_variables=["chef_output"],
    template=(
        "You are a Registered Dietitian reviewing the following recipe:\n"
        "---\n"
        "{chef_output}\n"
        "---\n"
        "Evaluate its health aspects and generate a healthier alternative by:\n"
        "- Reducing added sugars and unhealthy fats.\n"
        "- Suggesting substitutions for processed ingredients.\n"
        "- Suggesting different cooking steps to make it healthier.\n\n"
        "The output must follow the format:\n\n"
        "TITLE: The recipe title. It should be as close as possible to the original. "
        "Keep it the same if it applies.\n"
        "INGREDIENTS: A list of ingredients with quantities.\n"
        "PREPARATION: Step-by-step preparation instructions.\n"
        "HEALTH COMMENTS: A paragraph describing why the recipe is healthy and "
        "suggesting ways to make it even healthier (if applicable).\n\n"
        "Just generate an output using the given format. If the input is insufficient, "
        "output \"Insufficient input\"."
    ),
)  #G


def generate_recipes(user_input: str) -> tuple[str, str]:  #H
    """Runs the Chef -> Dietitian chain exactly like the Langflow graph."""

    chef_response = llm.invoke(chef_prompt.format(user_input=user_input))
    dietitian_response = llm.invoke(
        dietitian_prompt.format(chef_output=chef_response.content)
    )
    return chef_response.content, dietitian_response.content


def chat_loop() -> None:  #I
    """Simple REPL for repeated testing from the terminal."""

    try:
        while True:
            user_input = input("Ingredients or recipe idea: ")
            chef_recipe, healthy_recipe = generate_recipes(user_input)  #J

            print("\n--- Chef's Recipe ---")
            print(chef_recipe)
            print("\n--- Healthier Version ---")
            print(healthy_recipe)
            print("\n")
    except KeyboardInterrupt:
        print("\nHappy cooking!")


if __name__ == "__main__":  #K
    chat_loop()


#A Same ChatOpenAI wrapper from Chapter 2, reused for both personas.
#B PromptTemplate objects map 1:1 with Langflow’s prompt nodes.
#C `load_dotenv` pulls credentials from .env automatically.
#D Guard execution if the key is missing.
#E Single LLM instance handles the Chef and Dietitian turns sequentially.
#F Chef persona prompt exactly as wired in Langflow.
#G Dietitian persona prompt copies the second Langflow block.
#H Helper encapsulating the two-step generation so it’s reusable in tests.
#I Terminal loop to fire the chain repeatedly.
#J Capture both outputs so we can print or reuse them downstream.
#K Standard entrypoint guard.
