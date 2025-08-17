from pathlib import Path
from typing import Union

IMAGES_EXT = {'.jpg', '.jpeg', '.png', '.webp'}

def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def list_images(folder: Union[str, Path]):
    folder = Path(folder)
    return [p for p in folder.glob('*') if p.suffix.lower() in IMAGES_EXT]
