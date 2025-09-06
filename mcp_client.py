#!/usr/bin/env python3
"""
MCP Client for n8n integration
Handles communication with the MCP server for automated processing
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# MCP Client imports
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

class MCPClient:
    """MCP Client for AI Photo Video Enhancer"""
    
    def __init__(self):
        self.session = None
    
    async def connect(self):
        """Connect to MCP server"""
        # Start the MCP server process
        server_process = await asyncio.create_subprocess_exec(
            sys.executable, "mcp_server.py",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Create client session
        self.session = ClientSession(
            read_stream=server_process.stdout,
            write_stream=server_process.stdin
        )
        
        await self.session.initialize()
        return server_process
    
    async def process_and_upload(self, args):
        """Process images and upload to YouTube"""
        if not self.session:
            raise Exception("MCP client not connected")
        
        # Call the MCP tool
        result = await self.session.call_tool(
            name="process_and_upload",
            arguments=args
        )
        
        return result
    
    async def process_images_complete(self, args):
        """Process images with complete workflow"""
        if not self.session:
            raise Exception("MCP client not connected")
        
        result = await self.session.call_tool(
            name="process_images_complete",
            arguments=args
        )
        
        return result
    
    async def upload_to_youtube(self, args):
        """Upload video to YouTube"""
        if not self.session:
            raise Exception("MCP client not connected")
        
        result = await self.session.call_tool(
            name="upload_to_youtube",
            arguments=args
        )
        
        return result
    
    async def list_output_files(self, filter_type="all"):
        """List output files"""
        if not self.session:
            raise Exception("MCP client not connected")
        
        result = await self.session.call_tool(
            name="list_output_files",
            arguments={"filter_type": filter_type}
        )
        
        return result

async def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Parse arguments from environment variables (set by n8n)
    args = {}
    
    # Get arguments from environment variables
    for key, value in os.environ.items():
        if key.startswith('N8N_'):
            # Remove N8N_ prefix and convert to lowercase
            arg_key = key[4:].lower()
            
            # Try to parse as JSON, fallback to string
            try:
                args[arg_key] = json.loads(value)
            except:
                args[arg_key] = value
    
    # Also accept arguments from stdin (for direct calls)
    if not args and not sys.stdin.isatty():
        try:
            stdin_data = sys.stdin.read().strip()
            if stdin_data:
                args = json.loads(stdin_data)
        except:
            pass
    
    client = MCPClient()
    server_process = None
    
    try:
        # Connect to MCP server
        server_process = await client.connect()
        
        # Execute command
        if command == "process_and_upload":
            result = await client.process_and_upload(args)
        elif command == "process_images_complete":
            result = await client.process_images_complete(args)
        elif command == "upload_to_youtube":
            result = await client.upload_to_youtube(args)
        elif command == "list_output_files":
            result = await client.list_output_files(args.get("filter_type", "all"))
        else:
            raise ValueError(f"Unknown command: {command}")
        
        # Output result
        if result.isError:
            print(json.dumps({
                "status": "error",
                "message": result.content[0].text if result.content else "Unknown error"
            }))
            sys.exit(1)
        else:
            # Extract text content
            output = result.content[0].text if result.content else "{}"
            print(output)
    
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": str(e)
        }))
        sys.exit(1)
    
    finally:
        # Clean up
        if server_process:
            server_process.terminate()
            await server_process.wait()

if __name__ == "__main__":
    asyncio.run(main())
