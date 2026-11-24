"""Root stdio launcher for the DrawTool Aseprite MCP server."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.mcp_server import serve_stdio  # noqa: E402

if __name__ == "__main__":
    serve_stdio()
