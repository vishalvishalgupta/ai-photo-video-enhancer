import argparse
from pathlib import Path
import cv2
from utils import ensure_dir, list_images
from enhancers import enhance_pipeline
from funny import cartoonify, meme_caption, overlay_sticker
from diffusion import stylize_image, stylize_anime_specific, stylize_ghibli_art

IN_DIR = Path('inputs/photos')
OUT_IMG_DIR = ensure_dir('outputs/images')
OUT_VID_DIR = ensure_dir('outputs/videos')

def run_enhance():
    imgs = list_images(IN_DIR)
    if not imgs:
        print('No images in inputs/photos.')
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            print('Skip (unreadable):', p)
            continue
        out = enhance_pipeline(img)
        out_path = OUT_IMG_DIR / f"{p.stem}_enhanced.png"
        cv2.imwrite(str(out_path), out)
        print('Enhanced ->', out_path)

def run_funny_cartoons():
    imgs = list_images(IN_DIR)
    if not imgs:
        print('No images in inputs/photos.')
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None: 
            print('Skip (unreadable):', p); continue
        out = cartoonify(img)
        out_path = OUT_IMG_DIR / f"{p.stem}_cartoon.png"
        cv2.imwrite(str(out_path), out)
        print('Cartoon ->', out_path)

def run_funny_memes(top: str, bottom: str):
    imgs = list_images(IN_DIR)
    if not imgs:
        print('No images in inputs/photos.')
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None: 
            print('Skip (unreadable):', p); continue
        out = meme_caption(img, top_text=top, bottom_text=bottom)
        out_path = OUT_IMG_DIR / f"{p.stem}_meme.png"
        cv2.imwrite(str(out_path), out)
        print('Meme ->', out_path)

def run_funny_stickers(sticker: str, scale=0.3, position='top-left'):
    imgs = list_images(IN_DIR)
    if not imgs:
        print('No images in inputs/photos.')
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None: 
            print('Skip (unreadable):', p); continue
        out = overlay_sticker(img, sticker, scale=scale, position=position)
        out_path = OUT_IMG_DIR / f"{p.stem}_sticker.png"
        cv2.imwrite(str(out_path), out)
        print('Sticker ->', out_path)

def run_stylize(prompt: str):
    imgs = list_images(IN_DIR)
    if not imgs:
        print('No images in inputs/photos.')
    
    # Detect if this is an anime-specific request
    is_anime_request = any(word in prompt.lower() for word in ['anime', 'manga', 'cartoon', '2d'])
    
    for p in imgs:
        try:
            if is_anime_request:
                print(f'Using specialized anime transformation for {p.name}...')
                out_pil = stylize_anime_specific(str(p))
            else:
                out_pil = stylize_image(str(p), prompt)
        except Exception as e:
            print('Stylize failed for', p, '-', e)
            continue
        out_path = OUT_IMG_DIR / f"{p.stem}_styled.png"
        out_pil.save(out_path)
        print('Styled ->', out_path)

def run_ghibli(precision_mode='ultra'):
    """Generate Studio Ghibli art style images with precision control"""
    imgs = list_images(IN_DIR)
    if not imgs:
        print('No images in inputs/photos.')
        return
    
    print(f'🎨✨ Creating Studio Ghibli art style images in {precision_mode.upper()} precision mode...')
    
    for p in imgs:
        try:
            print(f'🖼️ Processing {p.name} with ULTRA-PRECISE Ghibli art style...')
            out_pil = stylize_ghibli_art(str(p), precision_mode=precision_mode)
        except Exception as e:
            print('❌ Ghibli stylization failed for', p, '-', e)
            continue
        
        # Add precision mode to filename for clarity
        suffix = f"_ghibli_{precision_mode}" if precision_mode != 'ultra' else "_ghibli"
        out_path = OUT_IMG_DIR / f"{p.stem}{suffix}.png"
        out_pil.save(out_path)
        print(f'✅ Ghibli art -> {out_path}')

def run_ghibli_batch():
    """Generate Studio Ghibli images in multiple precision modes for comparison"""
    imgs = list_images(IN_DIR)
    if not imgs:
        print('No images in inputs/photos.')
        return
    
    modes = ['ultra', 'balanced']
    print(f'🎨🔄 Creating Studio Ghibli art in multiple precision modes: {", ".join(modes)}')
    
    for mode in modes:
        print(f'\n🎯 Processing in {mode.upper()} mode...')
        for p in imgs:
            try:
                print(f'  🖼️ Processing {p.name} with {mode} precision...')
                out_pil = stylize_ghibli_art(str(p), precision_mode=mode)
                out_path = OUT_IMG_DIR / f"{p.stem}_ghibli_{mode}.png"
                out_pil.save(out_path)
                print(f'  ✅ {mode.capitalize()} Ghibli -> {out_path}')
            except Exception as e:
                print(f'  ❌ {mode.capitalize()} Ghibli failed for {p} - {e}')
                continue

def run_complete_workflow(video_duration: int = 45, meme_top: str = "Studio Ghibli", meme_bottom: str = "Magic Transformation"):
    """Complete workflow: All image effects + video with audio"""
    from pathlib import Path
    
    print("🎨✨🎬 COMPLETE AI PHOTO & VIDEO WORKFLOW 🎬✨🎨")
    print("=" * 60)
    
    # Check for input images
    imgs = list_images(IN_DIR)
    if not imgs:
        print('❌ No images found in inputs/photos/')
        print('Please add images to inputs/photos/ and try again.')
        return
    
    print(f"📸 Found {len(imgs)} input images")
    
    # Find audio file automatically
    audio_dir = Path('inputs/audio')
    audio_file = None
    if audio_dir.exists():
        audio_extensions = {'.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg'}
        audio_files = [f for f in audio_dir.glob('*') if f.suffix.lower() in audio_extensions]
        if audio_files:
            audio_file = str(audio_files[0])  # Use first audio file found
            print(f"🎵 Found audio file: {audio_file}")
        else:
            print("🔇 No audio files found in inputs/audio/")
    else:
        print("🔇 No inputs/audio/ directory found")
    
    print("\n🎨 STEP 1: IMAGE PROCESSING")
    print("-" * 30)
    
    # Step 1: Enhanced Images
    print("🔧 Processing enhanced images...")
    run_enhance()
    
    # Step 2: Cartoon Images  
    print("\n🎭 Processing cartoon images...")
    run_funny_cartoons()
    
    # Step 3: Meme Images
    print(f"\n😂 Processing meme images with text: '{meme_top}' / '{meme_bottom}'...")
    run_funny_memes(meme_top, meme_bottom)
    
    # Step 4: Ghibli Images (both modes)
    print("\n🎨✨ Processing Studio Ghibli images...")
    run_ghibli_batch()
    
    # Count total processed images
    processed_images = list(OUT_IMG_DIR.glob('*.png'))
    print(f"\n✅ IMAGE PROCESSING COMPLETE!")
    print(f"📊 Total processed images: {len(processed_images)}")
    
    # Step 5: Video Creation
    print(f"\n🎬 STEP 2: VIDEO CREATION ({video_duration}s)")
    print("-" * 30)
    
    if audio_file:
        print(f"🎵 Creating video with background audio: {Path(audio_file).name}")
        run_video(video_duration, audio_path=audio_file)
    else:
        print("🔇 Creating video without audio (no audio file found)")
        run_video(video_duration)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 COMPLETE WORKFLOW FINISHED! 🎉")
    print("=" * 60)
    print(f"📸 Input images processed: {len(imgs)}")
    print(f"🎨 Total output images: {len(processed_images)}")
    print(f"🎬 Video duration: {video_duration} seconds")
    if audio_file:
        print(f"🎵 Audio: {Path(audio_file).name}")
    print(f"📁 Video output: outputs/videos/slideshow.mp4")
    print("\n✨ Your magical AI-enhanced video is ready! ✨")

def run_video(duration: int, audio_path: str = None):
    from video import ken_burns_from_folder
    out_path = OUT_VID_DIR / 'slideshow.mp4'
    
    print(f'🎬 Creating {duration}-second video from processed images...')
    if audio_path:
        print(f'🎵 With background audio: {audio_path}')
    
    ken_burns_from_folder(OUT_IMG_DIR, out_path, duration=duration, audio_path=audio_path)
    print(f'🎬✨ Video created -> {out_path}')

def main():
    ap = argparse.ArgumentParser(description='AI Photo & Video Enhancer')
    ap.add_argument('--enhance', action='store_true', help='Run quality enhancement')
    ap.add_argument('--cartoon', action='store_true', help='Funny cartoonify')
    ap.add_argument('--meme', nargs=2, metavar=('TOP','BOTTOM'), help='Funny meme captions')
    ap.add_argument('--sticker', type=str, help='Path to PNG sticker overlay')
    ap.add_argument('--sticker-scale', type=float, default=0.3)
    ap.add_argument('--sticker-pos', type=str, default='top-left', choices=['top-left','top-right','bottom-left','bottom-right'])
    ap.add_argument('--stylize', type=str, help='Stable Diffusion img2img prompt (style)')
    ap.add_argument('--ghibli', action='store_true', help='Generate Studio Ghibli art style images (ultra precision)')
    ap.add_argument('--ghibli-mode', type=str, choices=['ultra', 'balanced'], default='ultra', 
                    help='Ghibli precision mode: ultra (best quality) or balanced (faster)')
    ap.add_argument('--ghibli-batch', action='store_true', help='Generate Ghibli images in all precision modes')
    ap.add_argument('--video', type=int, metavar='SECONDS', help='Make a video from outputs/images (specify duration in seconds)')
    ap.add_argument('--audio', type=str, help='Path to background audio file for video (mp3, wav, etc.)')
    ap.add_argument('--complete', type=int, metavar='SECONDS', help='🌟 COMPLETE WORKFLOW: All image effects + video with auto-detected audio (specify video duration)')
    ap.add_argument('--meme-text', nargs=2, metavar=('TOP','BOTTOM'), default=['Studio Ghibli', 'Magic Transformation'], 
                    help='Custom meme text for complete workflow (default: "Studio Ghibli" "Magic Transformation")')

    args = ap.parse_args()

    if args.enhance:
        run_enhance()
    if args.cartoon:
        run_funny_cartoons()
    if args.meme:
        top, bottom = args.meme
        run_funny_memes(top, bottom)
    if args.sticker:
        run_funny_stickers(args.sticker, scale=args.sticker_scale, position=args.sticker_pos)
    if args.stylize:
        run_stylize(args.stylize)
    if args.ghibli:
        run_ghibli(precision_mode=args.ghibli_mode)
    if args.ghibli_batch:
        run_ghibli_batch()
    if args.complete:
        meme_top, meme_bottom = args.meme_text
        run_complete_workflow(video_duration=args.complete, meme_top=meme_top, meme_bottom=meme_bottom)
    elif args.video:
        run_video(args.video, audio_path=args.audio)

if __name__ == '__main__':
    main()
