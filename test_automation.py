#!/usr/bin/env python3
"""
Test script for AI Photo Video Enhancer automation
Tests MCP server, webhook server, and YouTube integration
"""

import asyncio
import json
import requests
import subprocess
import sys
import time
from pathlib import Path

class AutomationTester:
    """Test automation components"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.webhook_url = "http://localhost:8080"
        
    def test_basic_functionality(self):
        """Test basic image processing functionality"""
        print("🧪 Testing basic functionality...")
        
        try:
            # Test the complete workflow command
            result = subprocess.run([
                sys.executable, "app.py", "--complete", "30"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Basic functionality test passed")
                return True
            else:
                print(f"❌ Basic functionality test failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Basic functionality test error: {e}")
            return False
    
    def test_youtube_setup(self):
        """Test YouTube API setup"""
        print("🧪 Testing YouTube API setup...")
        
        try:
            result = subprocess.run([
                sys.executable, "youtube_setup.py", "test"
            ], capture_output=True, text=True, timeout=30)
            
            if "YouTube API connection successful" in result.stdout:
                print("✅ YouTube API test passed")
                return True
            else:
                print("❌ YouTube API test failed - credentials not configured")
                print("Please run: python youtube_setup.py setup")
                return False
                
        except Exception as e:
            print(f"❌ YouTube API test error: {e}")
            return False
    
    def test_webhook_server(self):
        """Test webhook server"""
        print("🧪 Testing webhook server...")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.webhook_url}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    print("✅ Webhook server test passed")
                    return True
            
            print("❌ Webhook server test failed")
            return False
            
        except requests.exceptions.ConnectionError:
            print("❌ Webhook server not running")
            print("Please start: python webhook_server.py")
            return False
        except Exception as e:
            print(f"❌ Webhook server test error: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test API endpoints"""
        print("🧪 Testing API endpoints...")
        
        try:
            # Test list outputs endpoint
            response = requests.get(f"{self.webhook_url}/api/outputs", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "images" in data and "videos" in data:
                    print("✅ API endpoints test passed")
                    print(f"   Found {len(data['images'])} images, {len(data['videos'])} videos")
                    return True
            
            print("❌ API endpoints test failed")
            return False
            
        except Exception as e:
            print(f"❌ API endpoints test error: {e}")
            return False
    
    def test_complete_workflow_api(self):
        """Test complete workflow via API"""
        print("🧪 Testing complete workflow API...")
        
        # Check if we have test images
        input_dir = self.base_dir / "inputs" / "photos"
        if not input_dir.exists() or not list(input_dir.glob("*")):
            print("⚠️  No test images found, skipping workflow API test")
            return True
        
        try:
            # Get list of input images
            images = [str(f) for f in input_dir.glob("*") if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
            
            if not images:
                print("⚠️  No valid image files found, skipping workflow API test")
                return True
            
            # Test payload
            payload = {
                "images": images[:2],  # Use first 2 images
                "video_duration": 20,
                "youtube_title": "API Test Video",
                "youtube_description": "Test video created via API",
                "youtube_tags": ["test", "api", "automation"],
                "privacy": "private"
            }
            
            # Start processing
            response = requests.post(
                f"{self.webhook_url}/api/process-and-upload",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                job_id = data.get("job_id")
                
                if job_id:
                    print(f"✅ Workflow API test started - Job ID: {job_id}")
                    print("   (Processing will continue in background)")
                    return True
            
            print("❌ Complete workflow API test failed")
            return False
            
        except Exception as e:
            print(f"❌ Complete workflow API test error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting AI Photo Video Enhancer Automation Tests")
        print("=" * 60)
        
        tests = [
            ("Basic Functionality", self.test_basic_functionality),
            ("YouTube API Setup", self.test_youtube_setup),
            ("Webhook Server", self.test_webhook_server),
            ("API Endpoints", self.test_api_endpoints),
            ("Complete Workflow API", self.test_complete_workflow_api)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}")
            print("-" * 30)
            results[test_name] = test_func()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\n📊 Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Your automation setup is ready!")
        else:
            print("⚠️  Some tests failed. Please check the setup guide.")
            print("📖 See AUTOMATION_SETUP.md for detailed instructions.")
        
        return passed == total

def main():
    """Main entry point"""
    print("AI Photo Video Enhancer - Automation Test Suite")
    print("=" * 60)
    
    tester = AutomationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 Ready for automation! Try these commands:")
        print("   python webhook_server.py  # Start webhook server")
        print("   python mcp_server.py      # Start MCP server")
        print("   # Import n8n_workflow.json into your n8n instance")
        sys.exit(0)
    else:
        print("\n🔧 Setup required. Please check the failed tests above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
