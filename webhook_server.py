#!/usr/bin/env python3
"""
Webhook Server for AI Photo Video Enhancer
Provides REST API endpoints for external integrations
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
from aiohttp import web, ClientSession
import aiofiles

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebhookServer:
    """Webhook server for AI Photo Video Enhancer automation"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.base_dir = Path(__file__).parent
        self.jobs = {}  # Simple in-memory job tracking
        self.setup_routes()
    
    def setup_routes(self):
        """Set up API routes"""
        self.app.router.add_post('/api/process', self.handle_process_request)
        self.app.router.add_post('/api/upload', self.handle_upload_request)
        self.app.router.add_post('/api/process-and-upload', self.handle_complete_workflow)
        self.app.router.add_get('/api/status/{job_id}', self.handle_status_request)
        self.app.router.add_get('/api/jobs', self.handle_list_jobs)
        self.app.router.add_get('/api/outputs', self.handle_list_outputs)
        self.app.router.add_get('/health', self.handle_health_check)
        
        # CORS middleware
        self.app.middlewares.append(self.cors_middleware)
    
    @web.middleware
    async def cors_middleware(self, request, handler):
        """CORS middleware"""
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    async def handle_health_check(self, request):
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "service": "ai-photo-video-enhancer",
            "timestamp": datetime.now().isoformat()
        })
    
    async def handle_process_request(self, request):
        """Handle image processing request"""
        try:
            data = await request.json()
            
            # Validate required fields
            if 'images' not in data:
                return web.json_response(
                    {"error": "Missing required field: images"}, 
                    status=400
                )
            
            # Create job
            job_id = str(uuid.uuid4())
            job_data = {
                "id": job_id,
                "status": "processing",
                "created_at": datetime.now().isoformat(),
                "type": "process_only",
                "input_data": data
            }
            self.jobs[job_id] = job_data
            
            # Start processing in background
            asyncio.create_task(self.process_images_background(job_id, data))
            
            return web.json_response({
                "job_id": job_id,
                "status": "accepted",
                "message": "Processing started"
            })
            
        except Exception as e:
            logger.error(f"Error in process request: {e}")
            return web.json_response(
                {"error": str(e)}, 
                status=500
            )
    
    async def handle_upload_request(self, request):
        """Handle YouTube upload request"""
        try:
            data = await request.json()
            
            # Validate required fields
            required_fields = ['video_path', 'title']
            for field in required_fields:
                if field not in data:
                    return web.json_response(
                        {"error": f"Missing required field: {field}"}, 
                        status=400
                    )
            
            # Create job
            job_id = str(uuid.uuid4())
            job_data = {
                "id": job_id,
                "status": "uploading",
                "created_at": datetime.now().isoformat(),
                "type": "upload_only",
                "input_data": data
            }
            self.jobs[job_id] = job_data
            
            # Start upload in background
            asyncio.create_task(self.upload_video_background(job_id, data))
            
            return web.json_response({
                "job_id": job_id,
                "status": "accepted",
                "message": "Upload started"
            })
            
        except Exception as e:
            logger.error(f"Error in upload request: {e}")
            return web.json_response(
                {"error": str(e)}, 
                status=500
            )
    
    async def handle_complete_workflow(self, request):
        """Handle complete workflow request"""
        try:
            data = await request.json()
            
            # Validate required fields
            required_fields = ['images', 'youtube_title']
            for field in required_fields:
                if field not in data:
                    return web.json_response(
                        {"error": f"Missing required field: {field}"}, 
                        status=400
                    )
            
            # Create job
            job_id = str(uuid.uuid4())
            job_data = {
                "id": job_id,
                "status": "processing",
                "created_at": datetime.now().isoformat(),
                "type": "complete_workflow",
                "input_data": data
            }
            self.jobs[job_id] = job_data
            
            # Start complete workflow in background
            asyncio.create_task(self.complete_workflow_background(job_id, data))
            
            return web.json_response({
                "job_id": job_id,
                "status": "accepted",
                "message": "Complete workflow started"
            })
            
        except Exception as e:
            logger.error(f"Error in complete workflow request: {e}")
            return web.json_response(
                {"error": str(e)}, 
                status=500
            )
    
    async def handle_status_request(self, request):
        """Handle job status request"""
        job_id = request.match_info['job_id']
        
        if job_id not in self.jobs:
            return web.json_response(
                {"error": "Job not found"}, 
                status=404
            )
        
        return web.json_response(self.jobs[job_id])
    
    async def handle_list_jobs(self, request):
        """Handle list jobs request"""
        return web.json_response({
            "jobs": list(self.jobs.values())
        })
    
    async def handle_list_outputs(self, request):
        """Handle list outputs request"""
        try:
            outputs = {
                "images": [],
                "videos": []
            }
            
            # List images
            img_dir = self.base_dir / "outputs" / "images"
            if img_dir.exists():
                outputs["images"] = [str(f) for f in img_dir.glob("*.png")]
            
            # List videos
            vid_dir = self.base_dir / "outputs" / "videos"
            if vid_dir.exists():
                outputs["videos"] = [str(f) for f in vid_dir.glob("*.mp4")]
            
            return web.json_response(outputs)
            
        except Exception as e:
            logger.error(f"Error listing outputs: {e}")
            return web.json_response(
                {"error": str(e)}, 
                status=500
            )
    
    async def process_images_background(self, job_id: str, data: Dict):
        """Process images in background"""
        try:
            self.jobs[job_id]["status"] = "processing"
            
            # Prepare input images
            await self.prepare_input_images(data.get('images', []))
            
            # Prepare audio if provided
            if data.get('audio_url'):
                await self.prepare_audio_file(data['audio_url'])
            
            # Run processing
            import subprocess
            import sys
            
            cmd = [
                sys.executable, "app.py", "--complete", 
                str(data.get('video_duration', 45)),
                "--meme-text", 
                data.get('meme_text', ['Studio Ghibli', 'Magic Transformation'])[0],
                data.get('meme_text', ['Studio Ghibli', 'Magic Transformation'])[1]
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0:
                # Get output video path
                video_path = self.base_dir / "outputs" / "videos" / "slideshow.mp4"
                
                self.jobs[job_id].update({
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "video_path": str(video_path) if video_path.exists() else None,
                    "output": result.stdout
                })
            else:
                self.jobs[job_id].update({
                    "status": "failed",
                    "completed_at": datetime.now().isoformat(),
                    "error": result.stderr
                })
                
        except Exception as e:
            logger.error(f"Error in background processing: {e}")
            self.jobs[job_id].update({
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "error": str(e)
            })
    
    async def upload_video_background(self, job_id: str, data: Dict):
        """Upload video to YouTube in background"""
        try:
            self.jobs[job_id]["status"] = "uploading"
            
            # Import YouTube uploader
            from youtube_setup import YouTubeUploader
            
            uploader = YouTubeUploader()
            result = uploader.upload_video(
                video_path=data['video_path'],
                title=data['title'],
                description=data.get('description', ''),
                tags=data.get('tags', []),
                privacy_status=data.get('privacy', 'unlisted'),
                thumbnail_path=data.get('thumbnail_path')
            )
            
            self.jobs[job_id].update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "youtube_result": result
            })
            
        except Exception as e:
            logger.error(f"Error in background upload: {e}")
            self.jobs[job_id].update({
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "error": str(e)
            })
    
    async def complete_workflow_background(self, job_id: str, data: Dict):
        """Complete workflow in background"""
        try:
            # Step 1: Process images
            await self.process_images_background(job_id, data)
            
            if self.jobs[job_id]["status"] != "completed":
                return  # Processing failed
            
            # Step 2: Upload to YouTube
            self.jobs[job_id]["status"] = "uploading"
            
            video_path = self.jobs[job_id].get("video_path")
            if not video_path:
                raise Exception("No video file generated")
            
            upload_data = {
                "video_path": video_path,
                "title": data['youtube_title'],
                "description": data.get('youtube_description', ''),
                "tags": data.get('youtube_tags', []),
                "privacy": data.get('privacy', 'unlisted')
            }
            
            await self.upload_video_background(job_id, upload_data)
            
            # Send webhook callback if provided
            if data.get('callback_url'):
                await self.send_webhook_callback(data['callback_url'], self.jobs[job_id])
            
        except Exception as e:
            logger.error(f"Error in complete workflow: {e}")
            self.jobs[job_id].update({
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "error": str(e)
            })
    
    async def prepare_input_images(self, image_urls: List[str]):
        """Download and prepare input images"""
        input_dir = self.base_dir / "inputs" / "photos"
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing images
        for f in input_dir.glob("*"):
            f.unlink()
        
        # Download new images
        async with ClientSession() as session:
            for i, url in enumerate(image_urls):
                if url.startswith(('http://', 'https://')):
                    # Download from URL
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.read()
                            file_path = input_dir / f"input_{i}.jpg"
                            async with aiofiles.open(file_path, 'wb') as f:
                                await f.write(content)
                else:
                    # Copy local file
                    import shutil
                    shutil.copy2(url, input_dir / f"input_{i}{Path(url).suffix}")
    
    async def prepare_audio_file(self, audio_url: str):
        """Download and prepare audio file"""
        audio_dir = self.base_dir / "inputs" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        if audio_url.startswith(('http://', 'https://')):
            # Download from URL
            async with ClientSession() as session:
                async with session.get(audio_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        file_path = audio_dir / "background_audio.mp3"
                        async with aiofiles.open(file_path, 'wb') as f:
                            await f.write(content)
        else:
            # Copy local file
            import shutil
            shutil.copy2(audio_url, audio_dir / f"background_audio{Path(audio_url).suffix}")
    
    async def send_webhook_callback(self, callback_url: str, job_data: Dict):
        """Send webhook callback"""
        try:
            async with ClientSession() as session:
                async with session.post(callback_url, json=job_data) as response:
                    logger.info(f"Webhook callback sent to {callback_url}: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send webhook callback: {e}")
    
    async def start_server(self):
        """Start the webhook server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Webhook server started on http://{self.host}:{self.port}")
        logger.info("Available endpoints:")
        logger.info("  POST /api/process - Process images only")
        logger.info("  POST /api/upload - Upload video to YouTube")
        logger.info("  POST /api/process-and-upload - Complete workflow")
        logger.info("  GET /api/status/{job_id} - Get job status")
        logger.info("  GET /api/jobs - List all jobs")
        logger.info("  GET /api/outputs - List output files")
        logger.info("  GET /health - Health check")

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Photo Video Enhancer Webhook Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    
    args = parser.parse_args()
    
    server = WebhookServer(host=args.host, port=args.port)
    await server.start_server()
    
    # Keep server running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Server stopped")

if __name__ == "__main__":
    asyncio.run(main())
