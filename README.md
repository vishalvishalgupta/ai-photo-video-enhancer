# AI Photo & Video Enhancer — Starter Kit

An end‑to‑end, local-first pipeline that:

1. Ingests your photos.
2. Enhances quality (denoise, sharpen, upscale).
3. Generates “funny” variants (cartoonify, meme captions, sticker overlays, style transfer via Stable Diffusion img2img).
4. Exports short AI videos (10–20s) from the same photos using Ken Burns pans/zooms (optionally extend to AnimateDiff).

> ⚠️ This is a **starter kit** you can run locally with CPU or GPU. Advanced steps (Stable Diffusion / AnimateDiff) benefit hugely from an NVIDIA GPU + CUDA.

## 1) Tech Choices
- Python 3.10+
- OpenCV, Pillow (classic image/fun ops)
- diffusers + torch (Stable Diffusion img2img)
- moviepy + ffmpeg (video rendering)

## 2) Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install -r requirements.txt
```

Install ffmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Windows: download & add to PATH.

(Optional) Hugging Face for gated models (e.g., SDXL):
```bash
huggingface-cli login
```
Copy `.env.example` to `.env` and adjust.

## 3) Usage
1) Put images into `inputs/photos/`.
2) Enhance:
```bash
python app.py --enhance
```
3) Funny:
```bash
python app.py --cartoon
python app.py --meme "when prod works" "but only on my machine"
python app.py --sticker inputs/stickers/sunglasses.png --sticker-scale 0.25 --sticker-pos bottom-right
```
4) Style transfer (Stable Diffusion img2img):
```bash
python app.py --stylize "cute Pixar style, bright soft colors, 3D render"
```
5) Make a short slideshow video (10–20s):
```bash
python app.py --video 15
```

Outputs go to `outputs/images/` and `outputs/videos/`.

## 4) Notes
- For speed/quality on SD img2img, use an NVIDIA GPU with a CUDA torch build.
- Tweak image size in `diffusion.py` (e.g., 512 or 768).
- Fonts: if `Impact.ttf` is missing, place it beside `funny.py` or adjust font selection.

## 5) Extending
- Real-ESRGAN/GFPGAN for upscaling/face restoration.
- ControlNet for pose/edge-guided transformations.
- AnimateDiff for short AI motions.
- FastAPI/Gradio web UI.
- YouTube Data API for auto-upload post render.

## 6) License
MIT
