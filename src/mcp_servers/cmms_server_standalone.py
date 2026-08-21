"""
CMMS MCP Server - runs as a standalone MCP service.
Can be called by MCP clients over stdio or HTTP transport.
"""
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("cmms-server")

# In-memory storage for mock CMMS
work_orders = []
parts_inventory = {
    "HPC-BLADE-001": {"part_name": "HPC Blade Set", "quantity": 25, "min_stock": 10},
    "HPC-BLADE-002": {"part_name": "HPC Blade (Single)", "quantity": 150, "min_stock": 50},
    "FAN-BLADE-001": {"part_name": "Fan Blade Set", "quantity": 15, "min_stock": 5},
    "BEARING-001": {"part_name": "Main Bearing", "quantity": 40, "min_stock": 15},
    "SENSOR-TEMP-001": {"part_name": "Temperature Sensor", "quantity": 100, "min_stock": 30},
    "SENSOR-PRESS-001": {"part_name": "Pressure Sensor", "quantity": 100, "min_stock": 30},
    "SEAL-001": {"part_name": "Compressor Seal Kit", "quantity": 60, "min_stock": 20},
    "VALVE-BLEED-001": {"part_name": "Bleed Air Valve", "quantity": 30, "min_stock": 10},
}


@mcp.tool()
def create_work_order(
    engine_id: int,
    priority: str,
    description: str,
    failure_mode_id: str = None
) -> dict:
    """Create a new work order in the CMMS system."""
    work_order_id = len(work_orders) + 1
    from datetime import datetime
    created_at = datetime.now().isoformat()
    
    work_order = {
        "work_order_id": work_order_id,
        "engine_id": engine_id,
        "priority": priority,
        "description": description,
        "failure_mode_id": failure_mode_id,
        "status": "OPEN",
        "created_at": created_at,
    }
    
    work_orders.append(work_order)
    return work_order


@mcp.tool()
def check_parts_inventory(part_id: str) -> dict:
    """Check availability of a part in inventory."""
    if part_id not in parts_inventory:
        return {
            "part_id": part_id,
            "available": False,
            "message": "Part not found in inventory"
        }
    
    part = parts_inventory[part_id]
    in_stock = part["quantity"] >= part["min_stock"]
    
    return {
        "part_id": part_id,
        "part_name": part["part_name"],
        "quantity": part["quantity"],
        "min_stock": part["min_stock"],
        "in_stock": in_stock,
        "stock_status": "ADEQUATE" if in_stock else "LOW"
    }


@mcp.tool()
def get_work_order(work_order_id: int) -> dict:
    """Retrieve details of a specific work order."""
    if work_order_id < 1 or work_order_id > len(work_orders):
        return {"error": "Work order not found"}
    
    return work_orders[work_order_id - 1]


if __name__ == "__main__":
    # Run with stdio transport (default for MCP)
    mcp.run()