import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def cartoonify(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=9, C=2)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def meme_caption(img, top_text='', bottom_text=''):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    W, H = pil.size
    try:
        font = ImageFont.truetype('Impact.ttf', size=int(H*0.09))
    except:
        font = ImageFont.load_default()

    def draw_text_center(text, y):
        if not text:
            return
        bbox = draw.textbbox((0,0), text, font=font)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x = (W - w) // 2
        draw.text((x, y), text.upper(), fill=(255,255,255), font=font,
                  stroke_width=3, stroke_fill=(0,0,0))

    if top_text:
        draw_text_center(top_text, int(H*0.03))
    if bottom_text:
        draw_text_center(bottom_text, int(H*0.85))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def overlay_sticker(img, sticker_path: str, scale=0.3, position='top-left', margin=20):
    base = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGBA')
    sticker = Image.open(sticker_path).convert('RGBA')
    w, h = base.size
    sw = int(max(1, w * scale))
    sh = int(max(1, sticker.height * sw / max(1, sticker.width)))
    sticker = sticker.resize((sw, sh), Image.LANCZOS)

    positions = {
        'top-left': (margin, margin),
        'top-right': (w - sw - margin, margin),
        'bottom-left': (margin, h - sh - margin),
        'bottom-right': (w - sw - margin, h - sh - margin),
    }
    pos = positions.get(position, (margin, margin))
    base.alpha_composite(sticker, dest=pos)
    return cv2.cvtColor(np.array(base.convert('RGB')), cv2.COLOR_RGB2BGR)
