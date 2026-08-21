"""
Simple MCP client to test the CMMS server.
This demonstrates MCP client-server communication without needing the inspector.
"""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_cmms_server():
    # Start the CMMS server as a subprocess
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.mcp_servers.cmms_server"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Call create_work_order
            result = await session.call_tool(
                "create_work_order",
                arguments={
                    "engine_id": 34,
                    "priority": "HIGH",
                    "description": "Test work order from MCP client",
                    "failure_mode_id": "Test failure mode"
                }
            )
            
            print("\nWork order created:")
            print(result.content)
            
            # Call check_parts_inventory
            result = await session.call_tool(
                "check_parts_inventory",
                arguments={"part_id": "HPC-BLADE-001"}
            )
            
            print("\nParts inventory:")
            print(result.content)


if __name__ == "__main__":
    asyncio.run(test_cmms_server())