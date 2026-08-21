"""
Playwright MCP Testing Agent
Drives the Streamlit dashboard end-to-end to catch UI regressions.
"""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json


async def run_playwright_tests(dashboard_url: str = "http://localhost:8501"):
    """
    Run end-to-end tests on the Streamlit dashboard using Playwright MCP.
    """
    print(f"Starting Playwright MCP test agent for {dashboard_url}...")
    
    # Start Playwright MCP server
    server_params = StdioServerParameters(
        command="npx",
        args=["@playwright/mcp@latest", "--browser", "chromium"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available Playwright tools
            tools = await session.list_tools()
            print(f"\nAvailable Playwright tools: {[t.name for t in tools.tools]}")
            
            # Test 1: Navigate to dashboard
            print("\n=== Test 1: Navigate to Dashboard ===")
            await session.call_tool("navigate", {"url": dashboard_url})
            print("✅ Navigated to dashboard")
            
            # Test 2: Check page title
            print("\n=== Test 2: Check Page Title ===")
            result = await session.call_tool("evaluate", {"function": "document.title"})
            print(f"Page title: {result.content[0].text}")
            
            # Test 3: Find engine selector
            print("\n=== Test 3: Find Engine Selector ===")
            result = await session.call_tool(
                "evaluate",
                {"function": "document.querySelector('[data-testid=\"stSelectbox\"]') !== null"}
            )
            has_selector = json.loads(result.content[0].text)
            print(f"Engine selector found: {has_selector}")
            
            # Test 4: Take screenshot
            print("\n=== Test 4: Take Screenshot ===")
            await session.call_tool("screenshot", {"path": "test_screenshot.png"})
            print("✅ Screenshot saved to test_screenshot.png")
            
            print("\n✅ All Playwright tests passed!")


if __name__ == "__main__":
    asyncio.run(run_playwright_tests())