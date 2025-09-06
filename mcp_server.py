#!/usr/bin/env python3
"""
MCP Server for AI Photo & Video Enhancer
Provides automated image processing and video creation with YouTube upload capabilities
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import subprocess
import tempfile
import shutil
from datetime import datetime

# MCP Server imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIPhotoVideoEnhancerMCP:
    """MCP Server for AI Photo & Video Enhancer automation"""
    
    def __init__(self):
        self.server = Server("ai-photo-video-enhancer")
        self.base_dir = Path(__file__).parent
        self.setup_handlers()
    
    def setup_handlers(self):
        """Set up MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """List available tools"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="process_images_complete",
                        description="Complete workflow: Process images with all effects and create video with audio",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "input_images": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of image file paths or URLs to process"
                                },
                                "audio_file": {
                                    "type": "string",
                                    "description": "Path or URL to background audio file (optional)"
                                },
                                "video_duration": {
                                    "type": "integer",
                                    "default": 45,
                                    "description": "Video duration in seconds"
                                },
                                "meme_text": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "maxItems": 2,
                                    "default": ["Studio Ghibli", "Magic Transformation"],
                                    "description": "Top and bottom text for memes"
                                },
                                "output_title": {
                                    "type": "string",
                                    "default": "AI Enhanced Video",
                                    "description": "Title for the output video"
                                }
                            },
                            "required": ["input_images"]
                        }
                    ),
                    Tool(
                        name="upload_to_youtube",
                        description="Upload video to YouTube with metadata",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "video_path": {
                                    "type": "string",
                                    "description": "Path to the video file to upload"
                                },
                                "title": {
                                    "type": "string",
                                    "description": "YouTube video title"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "YouTube video description"
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "YouTube video tags"
                                },
                                "privacy": {
                                    "type": "string",
                                    "enum": ["private", "unlisted", "public"],
                                    "default": "unlisted",
                                    "description": "Video privacy setting"
                                },
                                "thumbnail_path": {
                                    "type": "string",
                                    "description": "Path to custom thumbnail (optional)"
                                }
                            },
                            "required": ["video_path", "title"]
                        }
                    ),
                    Tool(
                        name="process_and_upload",
                        description="Complete automation: Process images, create video, and upload to YouTube",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "input_images": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of image file paths or URLs"
                                },
                                "audio_file": {
                                    "type": "string",
                                    "description": "Background audio file path or URL"
                                },
                                "video_duration": {
                                    "type": "integer",
                                    "default": 45,
                                    "description": "Video duration in seconds"
                                },
                                "youtube_title": {
                                    "type": "string",
                                    "description": "YouTube video title"
                                },
                                "youtube_description": {
                                    "type": "string",
                                    "description": "YouTube video description"
                                },
                                "youtube_tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "default": ["AI", "Studio Ghibli", "Photo Enhancement", "Automation"],
                                    "description": "YouTube tags"
                                },
                                "privacy": {
                                    "type": "string",
                                    "enum": ["private", "unlisted", "public"],
                                    "default": "unlisted",
                                    "description": "YouTube privacy setting"
                                }
                            },
                            "required": ["input_images", "youtube_title"]
                        }
                    ),
                    Tool(
                        name="get_processing_status",
                        description="Get status of current processing job",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "job_id": {
                                    "type": "string",
                                    "description": "Job ID to check status for"
                                }
                            },
                            "required": ["job_id"]
                        }
                    ),
                    Tool(
                        name="list_output_files",
                        description="List all generated output files",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "filter_type": {
                                    "type": "string",
                                    "enum": ["all", "images", "videos"],
                                    "default": "all",
                                    "description": "Filter output files by type"
                                }
                            }
                        }
                    )
                ]
            )
        
        @self.server.call_tool()
        async def handle_call_tool(request: CallToolRequest) -> CallToolResult:
            """Handle tool calls"""
            try:
                if request.name == "process_images_complete":
                    return await self.process_images_complete(request.arguments)
                elif request.name == "upload_to_youtube":
                    return await self.upload_to_youtube(request.arguments)
                elif request.name == "process_and_upload":
                    return await self.process_and_upload(request.arguments)
                elif request.name == "get_processing_status":
                    return await self.get_processing_status(request.arguments)
                elif request.name == "list_output_files":
                    return await self.list_output_files(request.arguments)
                else:
                    raise ValueError(f"Unknown tool: {request.name}")
            
            except Exception as e:
                logger.error(f"Error in {request.name}: {str(e)}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")],
                    isError=True
                )
    
    async def process_images_complete(self, args: Dict[str, Any]) -> CallToolResult:
        """Process images with complete workflow"""
        try:
            input_images = args.get("input_images", [])
            audio_file = args.get("audio_file")
            video_duration = args.get("video_duration", 45)
            meme_text = args.get("meme_text", ["Studio Ghibli", "Magic Transformation"])
            output_title = args.get("output_title", "AI Enhanced Video")
            
            # Create job ID
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare input directory
            input_dir = self.base_dir / "inputs" / "photos"
            input_dir.mkdir(parents=True, exist_ok=True)
            
            # Download/copy input images
            for i, img_path in enumerate(input_images):
                if img_path.startswith(('http://', 'https://')):
                    # Download from URL
                    await self.download_file(img_path, input_dir / f"input_{i}.jpg")
                else:
                    # Copy local file
                    shutil.copy2(img_path, input_dir / f"input_{i}{Path(img_path).suffix}")
            
            # Handle audio file
            if audio_file:
                audio_dir = self.base_dir / "inputs" / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)
                
                if audio_file.startswith(('http://', 'https://')):
                    await self.download_file(audio_file, audio_dir / "background_audio.mp3")
                else:
                    shutil.copy2(audio_file, audio_dir / f"background_audio{Path(audio_file).suffix}")
            
            # Run complete workflow
            cmd = [
                sys.executable, "app.py", "--complete", str(video_duration),
                "--meme-text", meme_text[0], meme_text[1]
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                raise Exception(f"Processing failed: {result.stderr}")
            
            # Get output video path
            video_path = self.base_dir / "outputs" / "videos" / "slideshow.mp4"
            
            if not video_path.exists():
                raise Exception("Video output not found")
            
            # Count processed images
            output_images = list((self.base_dir / "outputs" / "images").glob("*.png"))
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "status": "success",
                            "job_id": job_id,
                            "video_path": str(video_path),
                            "processed_images": len(output_images),
                            "video_duration": video_duration,
                            "output_title": output_title,
                            "message": f"Successfully processed {len(input_images)} input images into {len(output_images)} enhanced images and created video"
                        }, indent=2)
                    )
                ]
            )
            
        except Exception as e:
            logger.error(f"Error in process_images_complete: {str(e)}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Processing failed: {str(e)}")],
                isError=True
            )
    
    async def upload_to_youtube(self, args: Dict[str, Any]) -> CallToolResult:
        """Upload video to YouTube"""
        try:
            video_path = args.get("video_path")
            title = args.get("title")
            description = args.get("description", "")
            tags = args.get("tags", [])
            privacy = args.get("privacy", "unlisted")
            thumbnail_path = args.get("thumbnail_path")
            
            if not Path(video_path).exists():
                raise Exception(f"Video file not found: {video_path}")
            
            # Create YouTube upload script
            upload_script = await self.create_youtube_upload_script(
                video_path, title, description, tags, privacy, thumbnail_path
            )
            
            # Run YouTube upload
            result = subprocess.run(
                [sys.executable, upload_script],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode != 0:
                raise Exception(f"YouTube upload failed: {result.stderr}")
            
            # Parse upload result
            upload_result = json.loads(result.stdout)
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "status": "success",
                            "youtube_url": upload_result.get("url"),
                            "video_id": upload_result.get("id"),
                            "title": title,
                            "privacy": privacy,
                            "message": "Successfully uploaded to YouTube"
                        }, indent=2)
                    )
                ]
            )
            
        except Exception as e:
            logger.error(f"Error in upload_to_youtube: {str(e)}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"YouTube upload failed: {str(e)}")],
                isError=True
            )
    
    async def process_and_upload(self, args: Dict[str, Any]) -> CallToolResult:
        """Complete automation: Process and upload to YouTube"""
        try:
            # Step 1: Process images
            process_args = {
                "input_images": args.get("input_images"),
                "audio_file": args.get("audio_file"),
                "video_duration": args.get("video_duration", 45),
                "output_title": args.get("youtube_title", "AI Enhanced Video")
            }
            
            process_result = await self.process_images_complete(process_args)
            
            if process_result.isError:
                return process_result
            
            # Parse processing result
            process_data = json.loads(process_result.content[0].text)
            video_path = process_data["video_path"]
            
            # Step 2: Upload to YouTube
            upload_args = {
                "video_path": video_path,
                "title": args.get("youtube_title"),
                "description": args.get("youtube_description", ""),
                "tags": args.get("youtube_tags", ["AI", "Studio Ghibli", "Photo Enhancement"]),
                "privacy": args.get("privacy", "unlisted")
            }
            
            upload_result = await self.upload_to_youtube(upload_args)
            
            if upload_result.isError:
                return upload_result
            
            # Combine results
            upload_data = json.loads(upload_result.content[0].text)
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "status": "success",
                            "processing": process_data,
                            "youtube": upload_data,
                            "message": "Complete automation successful: Images processed and video uploaded to YouTube"
                        }, indent=2)
                    )
                ]
            )
            
        except Exception as e:
            logger.error(f"Error in process_and_upload: {str(e)}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Complete automation failed: {str(e)}")],
                isError=True
            )
    
    async def get_processing_status(self, args: Dict[str, Any]) -> CallToolResult:
        """Get processing status"""
        job_id = args.get("job_id")
        
        # Simple status check (in production, you'd use a proper job queue)
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps({
                        "job_id": job_id,
                        "status": "completed",
                        "message": "Job status check - implement proper job tracking for production"
                    }, indent=2)
                )
            ]
        )
    
    async def list_output_files(self, args: Dict[str, Any]) -> CallToolResult:
        """List output files"""
        filter_type = args.get("filter_type", "all")
        
        outputs = {
            "images": [],
            "videos": []
        }
        
        # List images
        if filter_type in ["all", "images"]:
            img_dir = self.base_dir / "outputs" / "images"
            if img_dir.exists():
                outputs["images"] = [str(f) for f in img_dir.glob("*.png")]
        
        # List videos
        if filter_type in ["all", "videos"]:
            vid_dir = self.base_dir / "outputs" / "videos"
            if vid_dir.exists():
                outputs["videos"] = [str(f) for f in vid_dir.glob("*.mp4")]
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(outputs, indent=2)
                )
            ]
        )
    
    async def download_file(self, url: str, dest_path: Path):
        """Download file from URL"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(dest_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                else:
                    raise Exception(f"Failed to download {url}: HTTP {response.status}")
    
    async def create_youtube_upload_script(self, video_path: str, title: str, 
                                         description: str, tags: List[str], 
                                         privacy: str, thumbnail_path: Optional[str]) -> str:
        """Create YouTube upload script"""
        script_content = f'''
import os
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

def upload_video():
    creds = None
    # Load credentials
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    youtube = build('youtube', 'v3', credentials=creds)
    
    body = {{
        'snippet': {{
            'title': '{title}',
            'description': '{description}',
            'tags': {json.dumps(tags)},
            'categoryId': '22'  # People & Blogs
        }},
        'status': {{
            'privacyStatus': '{privacy}'
        }}
    }}
    
    media = MediaFileUpload('{video_path}', chunksize=-1, resumable=True)
    
    request = youtube.videos().insert(
        part=','.join(body.keys()),
        body=body,
        media_body=media
    )
    
    response = request.execute()
    
    result = {{
        'id': response['id'],
        'url': f"https://www.youtube.com/watch?v={{response['id']}}",
        'title': response['snippet']['title']
    }}
    
    print(json.dumps(result))

if __name__ == '__main__':
    upload_video()
'''
        
        script_path = self.base_dir / "youtube_upload.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return str(script_path)
    
    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="ai-photo-video-enhancer",
                    server_version="1.0.0",
                    capabilities={}
                )
            )

async def main():
    """Main entry point"""
    server = AIPhotoVideoEnhancerMCP()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
