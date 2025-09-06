#!/usr/bin/env python3
"""
Quick start script for AI Photo Video Enhancer automation
Starts all automation components in the correct order
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path
import signal
import os

class AutomationStarter:
    """Start and manage automation components"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.processes = []
        
    def start_webhook_server(self):
        """Start webhook server"""
        print("🌐 Starting webhook server...")
        process = subprocess.Popen([
            sys.executable, "webhook_server.py", "--host", "0.0.0.0", "--port", "8080"
        ], cwd=self.base_dir)
        self.processes.append(("webhook_server", process))
        time.sleep(2)  # Give it time to start
        return process
    
    def start_mcp_server(self):
        """Start MCP server"""
        print("🤖 Starting MCP server...")
        process = subprocess.Popen([
            sys.executable, "mcp_server.py"
        ], cwd=self.base_dir)
        self.processes.append(("mcp_server", process))
        time.sleep(1)
        return process
    
    def setup_youtube_credentials(self):
        """Setup YouTube credentials if not exists"""
        creds_file = self.base_dir / "credentials.json"
        if not creds_file.exists():
            print("📺 Setting up YouTube API credentials template...")
            subprocess.run([sys.executable, "youtube_setup.py", "setup"], cwd=self.base_dir)
            print("⚠️  Please configure YouTube API credentials:")
            print("   1. Go to Google Cloud Console")
            print("   2. Enable YouTube Data API v3")
            print("   3. Create OAuth 2.0 credentials")
            print("   4. Download and rename to 'credentials.json'")
            return False
        return True
    
    def test_components(self):
        """Test if components are running"""
        print("🧪 Testing automation components...")
        
        # Test webhook server
        try:
            import requests
            response = requests.get("http://localhost:8080/health", timeout=5)
            if response.status_code == 200:
                print("✅ Webhook server is running")
            else:
                print("❌ Webhook server test failed")
                return False
        except Exception as e:
            print(f"❌ Webhook server not accessible: {e}")
            return False
        
        print("✅ All components are running!")
        return True
    
    def show_usage_examples(self):
        """Show usage examples"""
        print("\n" + "=" * 60)
        print("🎯 AUTOMATION IS READY!")
        print("=" * 60)
        print("\n📡 Webhook Server: http://localhost:8080")
        print("🤖 MCP Server: Running in background")
        
        print("\n🔗 API Endpoints:")
        print("   GET  /health                    - Health check")
        print("   POST /api/process               - Process images only")
        print("   POST /api/upload                - Upload video to YouTube")
        print("   POST /api/process-and-upload    - Complete workflow")
        print("   GET  /api/status/{job_id}       - Check job status")
        print("   GET  /api/jobs                  - List all jobs")
        print("   GET  /api/outputs               - List output files")
        
        print("\n📝 Example API Call:")
        print("""curl -X POST http://localhost:8080/api/process-and-upload \\
  -H "Content-Type: application/json" \\
  -d '{
    "images": ["path/to/image1.jpg", "path/to/image2.jpg"],
    "audio_url": "path/to/background.mp3",
    "video_duration": 45,
    "youtube_title": "My AI Enhanced Video",
    "youtube_description": "Created with AI automation",
    "youtube_tags": ["AI", "Studio Ghibli", "Automation"],
    "privacy": "unlisted"
  }'""")
        
        print("\n🔄 n8n Integration:")
        print("   1. Import n8n_workflow.json into your n8n instance")
        print("   2. Configure webhook URL: http://localhost:8080/api/process-and-upload")
        print("   3. Set up email/Twitter notifications (optional)")
        
        print("\n⚡ Quick Test:")
        print("   python test_automation.py")
        
        print("\n🛑 To stop automation:")
        print("   Press Ctrl+C")
    
    def cleanup(self):
        """Clean up processes"""
        print("\n🛑 Stopping automation components...")
        for name, process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ Stopped {name}")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"🔪 Force killed {name}")
            except Exception as e:
                print(f"⚠️  Error stopping {name}: {e}")
    
    def run(self):
        """Run the automation starter"""
        print("🚀 AI Photo Video Enhancer - Automation Starter")
        print("=" * 60)
        
        try:
            # Setup YouTube credentials
            youtube_ready = self.setup_youtube_credentials()
            if not youtube_ready:
                print("\n⚠️  YouTube API not configured. Some features will be limited.")
                print("You can still use image processing and local video creation.")
            
            # Start components
            self.start_webhook_server()
            self.start_mcp_server()
            
            # Test components
            if self.test_components():
                self.show_usage_examples()
                
                # Keep running
                print("\n🔄 Automation is running... Press Ctrl+C to stop")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
            else:
                print("❌ Component testing failed")
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    starter = AutomationStarter()
    starter.run()

if __name__ == "__main__":
    main()
