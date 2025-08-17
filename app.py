import argparse
from pathlib import Path
import cv2
from utils import ensure_dir, list_images
from enhancers import enhance_pipeline
from funny import cartoonify, meme_caption, overlay_sticker
from diffusion import stylize_image, stylize_anime_specific

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

def run_video(duration: int):
    from video import ken_burns_from_folder
    out_path = OUT_VID_DIR / 'slideshow.mp4'
    ken_burns_from_folder(OUT_IMG_DIR, out_path, duration=duration)
    print('Video ->', out_path)

def main():
    ap = argparse.ArgumentParser(description='AI Photo & Video Enhancer')
    ap.add_argument('--enhance', action='store_true', help='Run quality enhancement')
    ap.add_argument('--cartoon', action='store_true', help='Funny cartoonify')
    ap.add_argument('--meme', nargs=2, metavar=('TOP','BOTTOM'), help='Funny meme captions')
    ap.add_argument('--sticker', type=str, help='Path to PNG sticker overlay')
    ap.add_argument('--sticker-scale', type=float, default=0.3)
    ap.add_argument('--sticker-pos', type=str, default='top-left', choices=['top-left','top-right','bottom-left','bottom-right'])
    ap.add_argument('--stylize', type=str, help='Stable Diffusion img2img prompt (style)')
    ap.add_argument('--video', type=int, metavar='SECONDS', help='Make a 10–20s video from outputs/images')

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
    if args.video:
        run_video(args.video)

if __name__ == '__main__':
    main()
