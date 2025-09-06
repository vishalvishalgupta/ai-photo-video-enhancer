#!/usr/bin/env python3
"""
YouTube API Setup and Upload Handler
Handles YouTube API authentication and video uploads
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# YouTube API scopes
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

class YouTubeUploader:
    """YouTube API handler for video uploads"""
    
    def __init__(self, credentials_file: str = "credentials.json", token_file: str = "token.json"):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.youtube = None
        self.setup_credentials()
    
    def setup_credentials(self):
        """Set up YouTube API credentials"""
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"YouTube API credentials file not found: {self.credentials_file}\n"
                        "Please download it from Google Cloud Console and place it in the project directory."
                    )
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                # Try different approaches for local server
                try:
                    creds = flow.run_local_server(host='127.0.0.1', port=8080)
                except Exception as e1:
                    try:
                        creds = flow.run_local_server(host='localhost', port=8080)
                    except Exception as e2:
                        # Fallback to console-based flow
                        creds = flow.run_console()
            
            # Save credentials for next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        # Build YouTube service
        self.youtube = build('youtube', 'v3', credentials=creds)
    
    def upload_video(self, 
                    video_path: str,
                    title: str,
                    description: str = "",
                    tags: List[str] = None,
                    category_id: str = "22",  # People & Blogs
                    privacy_status: str = "unlisted",
                    thumbnail_path: Optional[str] = None) -> Dict[str, Any]:
        """Upload video to YouTube"""
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        tags = tags or []
        
        # Video metadata
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags,
                'categoryId': category_id
            },
            'status': {
                'privacyStatus': privacy_status
            }
        }
        
        # Media upload
        media = MediaFileUpload(
            video_path,
            chunksize=-1,
            resumable=True,
            mimetype='video/mp4'
        )
        
        # Insert video
        insert_request = self.youtube.videos().insert(
            part=','.join(body.keys()),
            body=body,
            media_body=media
        )
        
        # Execute upload with retry logic
        response = self._resumable_upload(insert_request)
        
        # Upload thumbnail if provided
        if thumbnail_path and os.path.exists(thumbnail_path):
            try:
                self.upload_thumbnail(response['id'], thumbnail_path)
            except Exception as e:
                print(f"Warning: Thumbnail upload failed: {e}")
        
        return {
            'id': response['id'],
            'url': f"https://www.youtube.com/watch?v={response['id']}",
            'title': response['snippet']['title'],
            'description': response['snippet']['description'],
            'privacy_status': privacy_status
        }
    
    def upload_thumbnail(self, video_id: str, thumbnail_path: str):
        """Upload custom thumbnail for video"""
        media = MediaFileUpload(thumbnail_path, mimetype='image/jpeg')
        
        request = self.youtube.thumbnails().set(
            videoId=video_id,
            media_body=media
        )
        
        return request.execute()
    
    def _resumable_upload(self, insert_request):
        """Handle resumable upload with retry logic"""
        response = None
        error = None
        retry = 0
        
        while response is None:
            try:
                print("Uploading video to YouTube...")
                status, response = insert_request.next_chunk()
                if response is not None:
                    if 'id' in response:
                        print(f"Video uploaded successfully! Video ID: {response['id']}")
                        return response
                    else:
                        raise Exception(f"Upload failed: {response}")
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    # Retriable error
                    error = f"A retriable HTTP error {e.resp.status} occurred:\n{e.content}"
                    print(error)
                else:
                    raise e
            except Exception as e:
                error = f"An error occurred: {e}"
                print(error)
                raise e
            
            if error is not None:
                retry += 1
                if retry > 3:
                    raise Exception("Maximum retries exceeded")
                
                print(f"Retrying upload (attempt {retry})...")
                import time
                time.sleep(2 ** retry)  # Exponential backoff
        
        return response
    
    def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get video information"""
        request = self.youtube.videos().list(
            part="snippet,status,statistics",
            id=video_id
        )
        
        response = request.execute()
        
        if response['items']:
            return response['items'][0]
        else:
            raise Exception(f"Video not found: {video_id}")
    
    def update_video(self, video_id: str, title: str = None, description: str = None, 
                    tags: List[str] = None, privacy_status: str = None) -> Dict[str, Any]:
        """Update video metadata"""
        # Get current video info
        current_info = self.get_video_info(video_id)
        
        # Update fields
        snippet = current_info['snippet']
        if title:
            snippet['title'] = title
        if description:
            snippet['description'] = description
        if tags:
            snippet['tags'] = tags
        
        status = current_info['status']
        if privacy_status:
            status['privacyStatus'] = privacy_status
        
        # Update video
        body = {
            'id': video_id,
            'snippet': snippet,
            'status': status
        }
        
        request = self.youtube.videos().update(
            part='snippet,status',
            body=body
        )
        
        return request.execute()

def create_credentials_template():
    """Create template for YouTube API credentials"""
    template = {
        "installed": {
            "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
            "project_id": "your-project-id",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": "YOUR_CLIENT_SECRET",
            "redirect_uris": ["http://localhost"]
        }
    }
    
    with open("credentials_template.json", "w") as f:
        json.dump(template, f, indent=2)
    
    print("Created credentials_template.json")
    print("Please:")
    print("1. Go to Google Cloud Console")
    print("2. Enable YouTube Data API v3")
    print("3. Create OAuth 2.0 credentials")
    print("4. Download the credentials JSON file")
    print("5. Rename it to 'credentials.json'")

def main():
    """CLI interface for YouTube operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube API Operations")
    parser.add_argument("command", choices=["upload", "setup", "test"])
    parser.add_argument("--video", help="Video file path")
    parser.add_argument("--title", help="Video title")
    parser.add_argument("--description", help="Video description")
    parser.add_argument("--tags", nargs="*", help="Video tags")
    parser.add_argument("--privacy", choices=["private", "unlisted", "public"], 
                       default="unlisted", help="Privacy setting")
    parser.add_argument("--thumbnail", help="Thumbnail image path")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        create_credentials_template()
        return
    
    if args.command == "test":
        try:
            uploader = YouTubeUploader()
            print("✅ YouTube API connection successful!")
            print("Ready for video uploads.")
        except Exception as e:
            print(f"❌ YouTube API setup failed: {e}")
        return
    
    if args.command == "upload":
        if not args.video or not args.title:
            print("Error: --video and --title are required for upload")
            return
        
        try:
            uploader = YouTubeUploader()
            result = uploader.upload_video(
                video_path=args.video,
                title=args.title,
                description=args.description or "",
                tags=args.tags or [],
                privacy_status=args.privacy,
                thumbnail_path=args.thumbnail
            )
            
            print("✅ Upload successful!")
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    main()
