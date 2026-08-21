from pathlib import Path
from mcp.server.fastmcp import FastMCP
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.index import ManualIndex

# Initialize FastMCP server
mcp = FastMCP("manual-rag-server")

# Load the index
INDEX_DIR = Path("data/manuals/index")
indexer = ManualIndex()
indexer.load(INDEX_DIR)


@mcp.tool()
def search_manuals(query: str, top_k: int = 3) -> list[dict]:
    """
    Search maintenance manuals for relevant failure mode documentation.
    
    Args:
        query: Natural language description of the observed symptoms or sensor patterns
        top_k: Number of top results to return (default: 3)
    
    Returns:
        List of manuals with title, id, content, and relevance score
    """
    results = indexer.search(query, top_k=top_k)
    
    return [
        {
            "id": manual["id"],
            "title": manual["title"],
            "content": manual["content"].strip(),
            "relevance_score": score,
        }
        for manual, score in results
    ]


if __name__ == "__main__":
    # Run the server
    mcp.run()