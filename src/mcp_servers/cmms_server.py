from pathlib import Path
from datetime import datetime
from mcp.server.fastmcp import FastMCP
import json
import sqlite3

# Initialize FastMCP server
mcp = FastMCP("cmms-server")

# Database setup
DB_PATH = Path("data/cmms.db")


def init_db():
    """Initialize SQLite database for mock CMMS."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Work orders table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS work_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            engine_id INTEGER NOT NULL,
            priority TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT DEFAULT 'OPEN',
            created_at TEXT NOT NULL,
            failure_mode_id TEXT
        )
    """)
    
    # Parts inventory table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parts_inventory (
            part_id TEXT PRIMARY KEY,
            part_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            min_stock INTEGER DEFAULT 10
        )
    """)
    
    # Seed parts inventory
    parts = [
        ("HPC-BLADE-001", "HPC Blade Set", 25, 10),
        ("HPC-BLADE-002", "HPC Blade (Single)", 150, 50),
        ("FAN-BLADE-001", "Fan Blade Set", 15, 5),
        ("BEARING-001", "Main Bearing", 40, 15),
        ("SENSOR-TEMP-001", "Temperature Sensor", 100, 30),
        ("SENSOR-PRESS-001", "Pressure Sensor", 100, 30),
        ("SEAL-001", "Compressor Seal Kit", 60, 20),
        ("VALVE-BLEED-001", "Bleed Air Valve", 30, 10),
    ]
    
    cursor.executemany(
        "INSERT OR REPLACE INTO parts_inventory VALUES (?, ?, ?, ?)",
        parts
    )
    
    conn.commit()
    conn.close()


@mcp.tool()
def create_work_order(
    engine_id: int,
    priority: str,
    description: str,
    failure_mode_id: str = None
) -> dict:
    """
    Create a new work order in the CMMS system.
    
    Args:
        engine_id: Engine unit number
        priority: Priority level (HIGH, MEDIUM, LOW)
        description: Description of required maintenance
        failure_mode_id: Optional failure mode ID from manual search
    
    Returns:
        Created work order details including ID
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    created_at = datetime.now().isoformat()
    
    cursor.execute(
        """
        INSERT INTO work_orders (engine_id, priority, description, created_at, failure_mode_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (engine_id, priority, description, created_at, failure_mode_id)
    )
    
    work_order_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {
        "work_order_id": work_order_id,
        "engine_id": engine_id,
        "priority": priority,
        "description": description,
        "failure_mode_id": failure_mode_id,
        "status": "OPEN",
        "created_at": created_at,
    }


@mcp.tool()
def check_parts_inventory(part_id: str) -> dict:
    """
    Check availability of a part in inventory.
    
    Args:
        part_id: Part identifier (e.g., HPC-BLADE-001)
    
    Returns:
        Part details including quantity and stock status
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT part_id, part_name, quantity, min_stock FROM parts_inventory WHERE part_id = ?",
        (part_id,)
    )
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return {
            "part_id": part_id,
            "available": False,
            "message": "Part not found in inventory"
        }
    
    part_id, part_name, quantity, min_stock = row
    in_stock = quantity >= min_stock
    
    return {
        "part_id": part_id,
        "part_name": part_name,
        "quantity": quantity,
        "min_stock": min_stock,
        "in_stock": in_stock,
        "stock_status": "ADEQUATE" if in_stock else "LOW"
    }


@mcp.tool()
def get_work_order(work_order_id: int) -> dict:
    """
    Retrieve details of a specific work order.
    
    Args:
        work_order_id: Work order ID
    
    Returns:
        Work order details
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, engine_id, priority, description, status, created_at, failure_mode_id FROM work_orders WHERE id = ?",
        (work_order_id,)
    )
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return {
            "error": "Work order not found"
        }
    
    return {
        "work_order_id": row[0],
        "engine_id": row[1],
        "priority": row[2],
        "description": row[3],
        "status": row[4],
        "created_at": row[5],
        "failure_mode_id": row[6],
    }


# Initialize database on module load
init_db()


if __name__ == "__main__":
    mcp.run()