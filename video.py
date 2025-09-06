from moviepy import ImageClip, concatenate_videoclips, VideoClip, CompositeVideoClip, AudioFileClip, concatenate_audioclips
from PIL import Image
from pathlib import Path
from typing import Union, Optional
import os

def ken_burns_from_folder(images_dir: Union[str, Path], out_path: Union[str, Path], duration=12, fps=30, audio_path: Optional[Union[str, Path]] = None):
    paths = sorted([p for p in Path(images_dir).glob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}])
    if not paths:
        raise RuntimeError('No images found for video.')

    # Calculate duration per image
    per = max(2.0, duration / len(paths))
    
    print(f'Creating video from {len(paths)} images...')
    print(f'Duration per image: {per:.1f}s')
    
    # First, standardize all image sizes to avoid distortions
    target_size = (1920, 1080)  # Standard HD resolution
    clips = []
    
    for i, p in enumerate(paths):
        try:
            # Load image with PIL to check and standardize size
            pil_img = Image.open(p)
            
            # Convert to RGB if needed
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # Calculate aspect ratio and resize to fit target size while maintaining aspect ratio
            img_ratio = pil_img.width / pil_img.height
            target_ratio = target_size[0] / target_size[1]
            
            if img_ratio > target_ratio:
                # Image is wider - fit to width
                new_width = target_size[0]
                new_height = int(target_size[0] / img_ratio)
            else:
                # Image is taller - fit to height
                new_height = target_size[1]
                new_width = int(target_size[1] * img_ratio)
            
            # Resize image
            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a black background and center the image
            background = Image.new('RGB', target_size, (0, 0, 0))
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            background.paste(pil_img, (paste_x, paste_y))
            
            # Save temporary standardized image
            temp_path = f"/tmp/temp_img_{i}.png"
            background.save(temp_path)
            
            # Create ImageClip with standardized image
            img_clip = ImageClip(temp_path, duration=per)
            
            clips.append(img_clip)
            print(f'Processed image {i+1}/{len(paths)}: {p.name} -> {new_width}x{new_height}')
            
        except Exception as e:
            print(f'Warning: Skipping {p.name} due to error: {e}')
            continue

    if not clips:
        raise RuntimeError('No valid images could be processed for video.')

    print('Concatenating standardized clips...')
    # Since all clips now have the same size and format, concatenation should be smooth
    final = concatenate_videoclips(clips, method='chain')
    
    # Add background audio if provided
    if audio_path and Path(audio_path).exists():
        print(f'Adding background audio: {audio_path}')
        try:
            audio_clip = AudioFileClip(str(audio_path))
            
            # Adjust audio duration to match video duration
            video_duration = final.duration
            if audio_clip.duration > video_duration:
                # Trim audio to video length
                audio_clip = audio_clip.subclipped(0, video_duration)
                print(f'Audio trimmed to {video_duration:.1f} seconds')
            elif audio_clip.duration < video_duration:
                # Loop audio to match video length
                loops_needed = int(video_duration / audio_clip.duration) + 1
                audio_clip = concatenate_audioclips([audio_clip] * loops_needed).subclipped(0, video_duration)
                print(f'Audio looped to match video duration of {video_duration:.1f} seconds')
            
            # Set audio to video
            final = final.with_audio(audio_clip)
            print('✅ Background audio successfully added')
            
        except Exception as e:
            print(f'⚠️ Warning: Could not add audio - {e}')
            print('Proceeding with video generation without audio...')
    elif audio_path:
        print(f'⚠️ Warning: Audio file not found: {audio_path}')
        print('Proceeding with video generation without audio...')
    
    print(f'Writing video to {out_path}...')
    # Write the final video with consistent frame rate
    if final.audio is not None:
        # Include audio in output
        print('🎵 Writing video with background audio...')
        final.write_videofile(str(out_path), fps=fps, codec='libx264', audio_codec='aac', bitrate='5000k')
    else:
        # Video only
        print('🎬 Writing video without audio...')
        final.write_videofile(str(out_path), fps=fps, codec='libx264', bitrate='5000k')
    
    # Clean up temporary files
    for i in range(len(clips)):
        temp_path = f"/tmp/temp_img_{i}.png"
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print('🎬✨ Video generation completed with background audio support!')
