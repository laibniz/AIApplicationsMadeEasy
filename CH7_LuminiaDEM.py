"""
Chapter 7 appendix: Luminia™ DEM reappears as a FastAPI
microservice. We load the brand-guidelines PDF, weave it into the prompt, and
expose a POST `/generate-email` endpoint that accepts any JSON customer profile
and returns `{ "subject": "...", "body": "..." }` under 120 words.

Install deps once (pinning versions to match Langflow):

    pip install "langchain==0.2.7" "langchain-openai==0.1.14" \
        "langchain-community==0.2.7" pypdf fastapi uvicorn python-dotenv

Keep `Luminia Guidelines.pdf` next to this script and update `.env` with your
OpenAI key (or uncomment the inline placeholder).

Run the API locally:

    uvicorn CH7_LuminiaDEM:app --reload

Then send a test POST:

    curl -X POST http://localhost:8000/generate-email \
         -H "Content-Type: application/json" \
         -d '{ "profile": { "name": "SANTOSHI", "segment": "LIGHTS", "recent_purchase_days_ago": 30 } }'
"""

import os  #A
import json  #B
from typing import Any, Dict  #C

from dotenv import load_dotenv
from fastapi import FastAPI  #D
from pydantic import BaseModel  #E

from langchain_openai import ChatOpenAI  #F
from langchain_core.prompts import ChatPromptTemplate  #G
from langchain_community.document_loaders import PyPDFLoader  #H


# --- Configuration ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #I
GUIDELINES_PATH = os.path.join(BASE_DIR, "Luminia Guidelines.pdf")  #J

# os.environ["OPENAI_API_KEY"] = "REPLACE_WITH_YOUR_KEY"
load_dotenv()  #K

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)  #L


# --- Load and prepare guidelines text -------------------------------------

loader = PyPDFLoader(GUIDELINES_PATH)  #M
docs = loader.load()
guidelines_text = "\n\n".join(page.page_content for page in docs)  #N


PROMPT_TEMPLATE = """
Act as a senior copywriter for Luminia™.

Your task is to generate a personalised marketing email that includes:
- A short, engaging subject line
- A concise body tailored to the customer profile (under 120 words)

Follow the brand rules strictly and reference the customer data provided.
Just generate the subject and the body as a JSON object.
The names of the keys must be: 'subject' and 'body'.

-------------------------------------------------
### CUSTOMER GUIDELINES
{guidelines}
-------------------------------------------------
""".strip()  #O

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_TEMPLATE),
        ("human", "{customer_json}"),
    ]
)  #P


# --- Core helper: turn a profile into subject+body ------------------------


def generate_email_from_profile(profile: Dict[str, Any]) -> Dict[str, str]:
    """Call the LLM to generate a JSON email (subject+body) for one customer."""
    customer_json = json.dumps(profile, ensure_ascii=False, indent=2)  #Q
    messages = prompt.format_messages(
        guidelines=guidelines_text,
        customer_json=customer_json,
    )  #R
    response = llm.invoke(messages)  #S
    text = response.content.strip()  #T

    # We expect a JSON object like {"subject": "...", "body": "..."}.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # If parsing fails, return everything in the body and a generic subject.
        return {
            "subject": "Luminia™ special offer",
            "body": text,
        }

    subject = data.get("subject", "Luminia™ special offer")
    body = data.get("body", text)
    return {"subject": subject, "body": body}  #U


# --- FastAPI app and request model ----------------------------------------


class CustomerProfile(BaseModel):
    """A generic customer profile wrapper."""
    profile: Dict[str, Any]  #V


app = FastAPI(title="Luminia Email Generator")  #W


@app.post("/generate-email")
async def generate_email(payload: CustomerProfile) -> Dict[str, str]:
    """
    Accepts a JSON customer profile and returns a brand-compliant
    Luminia™ email with 'subject' and 'body' keys.
    """
    return generate_email_from_profile(payload.profile)  #X


# --- Optional: quick local test without HTTP ------------------------------


if __name__ == "__main__":  #Y
    # Simple CLI test: type a minimal profile and see the result.
    sample_profile = {
        "name": "Alex",
        "segment": "loyal_customer",
        "last_purchase_days_ago": 45,
        "preferred_channel": "email",
        "interests": ["smart lighting", "energy savings"],
    }
    print("Sample profile:", json.dumps(sample_profile, indent=2))
    email = generate_email_from_profile(sample_profile)
    print("\nGenerated email JSON:\n", json.dumps(email, indent=2, ensure_ascii=False))

    print(
        "\nTo start the API server, run from the terminal:\n"
        "  uvicorn CH7_Luminia_Email_API:app --reload\n"
        "Then POST to http://localhost:8000/generate-email with a JSON body like:\n"
        '{ "profile": { ... your customer data ... } }'
    )


#A Manage environment variables and file paths.
#B JSON encoding/decoding for customer profiles and model output.
#C Typing hints for dictionary-based profiles.
#D FastAPI framework to expose a simple HTTP endpoint.
#E Pydantic model for validating incoming request bodies.
#F Chat model wrapper used throughout the book.
#G Prompt template builder (system + human messages).
#H Convenience loader to extract text from the Luminia™ guidelines PDF.
#I Folder where this script lives.
#J Full path to the Luminia™ brand guidelines PDF file.
#K Load `.env` so the OpenAI key stays outside the source code.
#L Initialize the LLM “brain” of the email generator.
#M Load the PDF guidelines using the LangChain document loader.
#N Concatenate all PDF pages into one guidelines string.
#O System-level instructions for the copywriter, including the {guidelines} slot.
#P Prompt object combining system message and customer JSON as human input.
#Q Serialize the incoming customer profile as JSON text for the model.
#R Inject guidelines and customer JSON into the prompt.
#S Call the OpenAI chat model to generate the email.
#T Get the raw text output from the model.
#U Parse the expected JSON with 'subject' and 'body', with a simple fallback.
#V Pydantic schema: the API expects {"profile": { ...customer fields... }}.
#W FastAPI application object.
#X API endpoint logic: run the core helper and return its JSON result.
#Y Optional local test and a hint on how to start the FastAPI server.
