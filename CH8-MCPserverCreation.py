"""
Chapter 8 appendix: wraps the Langflow MCP tutorial by standing up a FastMCP
calculator server. Any agent—including `CH8-MCPserverCall.py`—can start this
process over STDIO and call the exposed `add(a, b)` tool.

Install once:

    pip install fastmcp

Usage:
1. `python CH8-MCPserverCreation.py 4 5` -> prints `9` (quick CLI test).
2. `python CH8-MCPserverCreation.py` -> runs as an MCP server over STDIO.
"""

import sys
from fastmcp import FastMCP  #A


server = FastMCP(
    name="Calculator MCP Server",
    instructions="Simple MCP server exposing basic calculator tools.",
)


@server.tool()  #B
async def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""  #C
    return a + b


if __name__ == "__main__":  #D
    if len(sys.argv) == 3:
        a = float(sys.argv[1])
        b = float(sys.argv[2])
        print(a + b)
    else:
        server.run()


#A Import FastMCP, a small library used to define MCP servers in Python.
#B Mark this function as a tool that other agents can call.
#C A simple calculator tool that adds two numbers and returns the result.
#D If numbers are provided, run as a calculator; otherwise, wait for tool calls.
