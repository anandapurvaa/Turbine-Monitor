"""
MCP client wrappers for LangGraph agents.
These functions call MCP servers instead of direct imports.
"""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json

async def call_mcp_tool(server_module: str, tool_name: str, arguments: dict):
    """
    Call an MCP tool asynchronously.
    """
    server_params = StdioServerParameters(
        command="python",
        args=["-m", server_module],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            # Parse the JSON response
            return json.loads(result.content[0].text)


# MCP tool wrappers for agents

def create_work_order_mcp(engine_id: int, priority: str, description: str, failure_mode_id: str = None):
    """Call CMMS server to create work order via MCP."""
    return asyncio.run(call_mcp_tool(
        "src.mcp_servers.cmms_server_standalone",
        "create_work_order",
        {
            "engine_id": engine_id,
            "priority": priority,
            "description": description,
            "failure_mode_id": failure_mode_id
        }
    ))


def check_parts_inventory_mcp(part_id: str):
    """Call CMMS server to check parts inventory via MCP."""
    return asyncio.run(call_mcp_tool(
        "src.mcp_servers.cmms_server_standalone",
        "check_parts_inventory",
        {"part_id": part_id}
    ))