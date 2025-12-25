"""
Chapter 8 appendix: demonstrates how the Langflow MCP client talks to our
calculator server (`CH8-MCPserverCreation.py`). We launch the FastMCP server
over STDIO, list its tools, and call `add(a, b)` to prove the connection works.

Install once:

    pip install fastmcp mcp
"""

import asyncio  #A
import sys

from mcp.client.session import ClientSession  #B
from mcp.client.stdio import StdioServerParameters, stdio_client  #C


async def main() -> None:  #D
    server_params = StdioServerParameters(command=sys.executable, args=["CH8-MCPserverCreation.py"])

    async with stdio_client(server_params) as (read, write):  #E
        async with ClientSession(read, write) as session:  #F
            await session.initialize()  #G

            tools_response = await session.list_tools()  #H
            tools = tools_response.tools
            print("Available tools on the server:")
            for tool in tools:
                print(f"- {tool.name}: {tool.description or '(no description)'}")

            result = await session.call_tool("add", {"a": 4.0, "b": 3.0})  #I

            print("\nRaw MCP tool result:")
            print(result)

            value = None  #J
            if getattr(result, "structuredContent", None):
                value = result.structuredContent.get("result")

            print(f"\n4 + 3 = {value}")


if __name__ == "__main__":  #K
    asyncio.run(main())


#A asyncio is used because the MCP client runs asynchronously.
#B ClientSession represents the connection to the MCP server.
#C These utilities start the server script and connect to it.
#D Main function that runs the MCP client logic.
#E Launch the calculator MCP server as a background process.
#F Open a session to talk to the server.
#G Initialize the connection before making requests.
#H Ask the server which tools it exposes.
#I Call the "add" tool with example inputs.
#J Read the numeric result returned by the tool.
#K Run the client when this file is executed directly.
