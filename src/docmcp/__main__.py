"""Allow running with: python -m docmcp.server"""
from docmcp.server import config, mcp

mcp.run(transport="http", host=config.host, port=config.port)
