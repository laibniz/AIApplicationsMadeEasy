# Python Appendix Guide

This guide extends the “What if we coded this?” sections from *AI Applications Made Easy* so Python developers can see exactly how the Langflow canvases translate into code. Langflow is built on LangChain—the de-facto standard framework for composing LLM “chains” that combine prompt templates, memory, tools, and model calls—so every script here simply recreates the same nodes programmatically.

## Chapter 2 – Hello World Pirate Chat (`CH2-HelloWorld.py`)

Chapter 2 introduces LangChain itself: a modular toolkit that lets you declare prompts, message templates, retrievers, and tool calls, then tie them together into “chains.” Langflow’s canvas is essentially a visual LangChain editor, so learning the Python API gives you the same power with more control. LangChain’s message classes (`SystemMessage`, `HumanMessage`) map 1:1 with the conversation blocks on the canvas, while `ChatOpenAI` is the standard chat model wrapper used across the industry.

The script mirrors the Langflow starter flow by loading the OpenAI key from `.env`, instantiating `ChatOpenAI`, and prepending the pirate persona text as a system prompt. The input loop simply gathers text from `input()`, constructs the `[system, human]` message list, and sends it to `llm.invoke`. That single function call is the textual equivalent of pressing “Send” in the Langflow playground, because LangChain handles the serialization and API call behind the scenes.

If you change the system prompt, temperature, or even swap `ChatOpenAI` for another LangChain-supported provider, you’ll immediately feel the same behavioral shifts you explored in the chapter. This is the simplest example of how LangChain chains let you keep the flow stable while experimenting with individual components.

## Chapter 3 – Memory & Multi-Step Personas

The third chapter focuses on expanding the pirate chatbot with memory and on orchestrating multiple personas. In Langflow that meant wiring Prompt Template nodes, Message History, and a second prompt for the Dietitian. In Python, the `PromptTemplate` class and a few lists get you the exact same structure—no external state store is required for simple conversations, which keeps the script approachable.

`CH3-HelloWorldImproved.py` keeps a `history_lines` list that accumulates “User/AI” pairs, concatenates them into a text block, and renders a LangChain `PromptTemplate` with the latest turn. This mirrors the Message History node feeding a Prompt Template node in Langflow. The rendered text becomes the `HumanMessage` content, so the LLM sees the same stitched transcript that the Langflow node produced.

`CH3-HealthyChef.py` demonstrates sequential reasoning: the Chef prompt feeds an LLM call, and the response slots into the Dietitian prompt before the second call. In Langflow you connected two `ChatOpenAI` nodes, each with its own prompt block. The script captures that pipeline inside `generate_recipes`, rendering each prompt with `PromptTemplate.format` and invoking the shared `ChatOpenAI` instance twice. Because everything stays inside Python, you can forward the Chef output to logs, insert validation steps, or stash intermediate results for auditing—exactly the type of control Chapter 3 encourages.

## Chapter 4 – PizzaMate Mini-RAG (`CH4-PizzaMate.py`)

PizzaMate’s Langflow graph introduces Retrieval-Augmented Generation: ingest a document, chunk it, embed it, store it, then retrieve relevant snippets before prompting the LLM. LangChain exposes each of those blocks as importable modules. `PyPDFLoader` mirrors the document loader node, `RecursiveCharacterTextSplitter` stands in for the chunker, `OpenAIEmbeddings` produces the vectors, and the Chroma vector store component becomes LangChain’s `Chroma.from_documents` call backed by a local `menu_chroma/` folder. Even the retriever wiring (`vectorstore.as_retriever`) is identical to the Langflow retriever configuration.

When the REPL calls `answer_query`, the script uses the Chroma retriever to fetch the top four chunks, concatenates their `page_content`, and injects that text plus the user question into a `PromptTemplate`. The prompt mirrors the context-aware template from the canvas (“Answer the user query by using only the provided context…”), and the final `ChatOpenAI` call plays the role of Langflow’s response node. Because Chroma persists automatically, you only need to run the ingestion step once—exactly like the file-backed vector store block in the chapter diagram.

Prefer FAISS? `CH4-PizzaMate-FAISS.py` keeps the same implementation so you can compare both file-based stores side by side.

If you want to swap the embedding model, change `OpenAIEmbeddings` to another LangChain embedding class; the rest of the script stays untouched. The same goes for retriever parameters, prompt phrasing, or the LLM itself, which is the kind of modular experimentation the chapter advocates.

## Chapter 5 – HistoryTravelBuddy Agent (`CH5-HistoryTravelBuddy.py`)

Chapter 5 introduces Langflow’s agent node wired to external tools such as SerpAPI search, raw URL fetchers, and Wikipedia queries. In LangChain, tools are regular Python callables decorated with `@tool`, and LangChain already ships with wrappers for SerpAPI (`SerpAPIWrapper`) and Wikipedia (`WikipediaQueryRun`). The script makes those explicit: `web_search` calls SerpAPI via the wrapper, `fetch_url` uses `requests.get`, and `wiki_tool` is LangChain’s built-in component. Once you drop the OpenAI and SerpAPI keys into `.env`, `AgentExecutor` orchestrates the reasoning loop exactly like Langflow’s agent node.

The agent prompt in the script is the same block of instructions you pasted into Langflow, so tool selection follows the same logic described in the chapter: gather attraction ideas, open pages, consult Wikipedia, and stitch together the itinerary. Because the Python version exposes the underlying tool functions, you can inspect inputs and outputs at each step, add logging, or introduce retries—handy when you’re debugging API hiccups outside the comfort of the Langflow UI.

This is also your first taste of how LangChain generalizes the visual agent wiring: if you create another `@tool` function and add it to the `tools` list, the agent immediately considers it, just as if you had dropped a new Tool node onto the canvas. That symmetry is what allows you to prototype in Langflow and then port the flow into production code without rewriting the logic.

## Chapter 6 – Agentic Structures

The Agentic Structures chapter explores both sequential and hierarchical orchestrations. `CH6-HealthyChefReimagined.py` mirrors the Type 1 structure from the book: a Chef agent runs retrieval over a cookbook RAG index, then a Dietitian agent fetches nutrition guidelines and rewrites the recipe. LangChain’s `create_tool_calling_agent` helper wraps each prompt-plus-tool bundle into an agent that can call its assigned tool. The Chef agent connects to the `cookbook_search` tool, which in turn pulls from a FAISS retriever built atop `recipe-book.txt`. The Dietitian agent gets a `fetch_url` tool that hits the NHS EatWell guide. Running the script reproduces the same handoff you built in Langflow: Chef generates a historically accurate dish, Dietitian adjusts it based on EatWell, and both messages print to the console.

`CH6-CopyGenie.py` maps the Type 2 hierarchical structure. The planner and copywriter are each defined as LangChain tools with their respective prompts; the planner reads from `Social_Media_Calendar_2025_2027.csv` using simple file I/O, while the copywriter consumes the plan along with `docx2txt`-parsed brand guidelines. The delivery agent is another tool that talks to Gmail through Composio, a hosted connector that exposes SaaS actions via a uniform API. Once those three tools are defined, `llm.bind_tools` produces the Supervisor agent. The Supervisor instructions are copied from the Langflow build (“Delegate planning to planner_agent… check that each plan entry has a corresponding post…”), so the LLM decides which tool to call next exactly like the canvas version.

Because the Supervisor loop is plain Python, you can attach breakpoints, persist the planner output, or mock the delivery agent when running tests. That level of observability is essential when you adapt CopyGenie to real marketing stacks, and it’s the natural next step after proving the design inside Langflow.

## Chapter 7 – Luminia DEM API (`CH7_LuminiaDEM.py`)

The KNIME + Langflow integration chapter ends with a microservice that produces brand-compliant emails for batched data. `CH7_LuminiaDEM.py` brings that service to life with FastAPI. The script loads the Luminia Guidelines PDF via `PyPDFLoader`, stitches the pages together, and drops the text into the `PROMPT_TEMPLATE`. FastAPI handles HTTP routing, while Pydantic validates the `CustomerProfile` model so any malformed payloads fail fast—mirroring the KNIME workflow’s schema enforcement.

When a POST hits `/generate-email`, the handler calls `generate_email_from_profile`, serializes the customer dict with `json.dumps`, and renders the LangChain prompt. The `ChatOpenAI` call returns the same JSON pair (“subject” and “body”) you configured in Langflow, which means KNIME can plug this endpoint into its automation just as it invoked the Langflow node. Because it’s regular FastAPI, you can also deploy it to any Python-friendly environment, add authentication, or capture telemetry—all without changing the prompt logic.

## Chapter 8 – MCP Tools

Chapter 8 focuses on the Model Context Protocol and how Langflow can consume external tool servers. `CH8-MCPserverCreation.py` uses FastMCP—a helper library that wraps the MCP transport and boilerplate—to expose the `add(a, b)` calculator tool. Declaring `@server.tool()` mirrors dropping a Tool node onto the Langflow canvas; FastMCP handles the JSON-RPC plumbing and lets you run the server over STDIO. The script keeps the quick CLI mode (pass two numbers and it prints the sum) so you can test the function outside MCP, just as the chapter suggests.

`CH8-MCPserverCall.py` is the Python analog of Langflow’s MCP client node. It uses the `mcp` library’s `stdio_client` helper to spawn the server script with `sys.executable`, negotiates the MCP handshake through `ClientSession.initialize()`, lists available tools, and calls `session.call_tool("add", {...})`. The resulting structured payload is printed exactly as Langflow would display it. By editing the arguments or adding logging, you can see every step of the MCP exchange, which demystifies the integration before you delegate it to an agent on the Langflow canvas.
