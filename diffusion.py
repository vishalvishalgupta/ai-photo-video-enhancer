import os
from dotenv import load_dotenv
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

load_dotenv()
# Use a more CPU-friendly model
MODEL_ID = os.getenv('MODEL_ID', 'runwayml/stable-diffusion-v1-5')

_pipe_cache = None

def get_img2img_pipe():
    global _pipe_cache
    if _pipe_cache is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # For CPU, use float32 to avoid precision issues
        if not torch.cuda.is_available():
            dtype = torch.float32
            
        _pipe_cache = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            # CPU optimizations
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        
        if torch.cuda.is_available():
            _pipe_cache = _pipe_cache.to('cuda')
        else:
            # CPU optimizations
            _pipe_cache = _pipe_cache.to('cpu')
            # Enable memory efficient attention for CPU
            try:
                _pipe_cache.enable_attention_slicing()
                _pipe_cache.enable_model_cpu_offload()
            except:
                pass  # These might not be available in all versions
                
    return _pipe_cache

def stylize_image(img_path: str, prompt: str, strength=0.15, guidance_scale=4.0, steps=30, size=768):
    pipe = get_img2img_pipe()
    init_image = Image.open(img_path).convert('RGB')
    original_size = init_image.size
    
    # For better quality, work with larger images but maintain aspect ratio
    max_dimension = max(original_size)
    if max_dimension > size:
        # Scale down proportionally
        scale_factor = size / max_dimension
        new_width = int(original_size[0] * scale_factor)
        new_height = int(original_size[1] * scale_factor)
    else:
        # Keep original size if it's already smaller
        new_width, new_height = original_size
    
    # Resize maintaining exact aspect ratio
    init_image = init_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # No padding - work directly with the properly sized image
    processed_image = init_image
    
    # Create very conservative anime prompting
    if "anime" in prompt.lower() or "manga" in prompt.lower():
        # Focus on style transfer while preserving structure
        enhanced_prompt = f"anime art style, {prompt}, preserve original composition, detailed, high quality"
    else:
        enhanced_prompt = f"{prompt}, preserve original composition, detailed, high quality"
    
    # Strong negative prompt to prevent any distortions
    negative_prompt = ("distorted, deformed, disfigured, poorly drawn face, mutation, mutated, "
                      "extra limb, ugly, poorly drawn hands, missing limb, floating limbs, "
                      "disconnected limbs, malformed hands, blur, out of focus, long neck, "
                      "long body, mutated hands and fingers, out of frame, blurry, bad anatomy, "
                      "blurred, watermark, grainy, signature, cut off, draft, duplicate, "
                      "coppy, multi, two faces, disfigured, kitsch, oversaturated, grain, "
                      "low-res, mutation, mutated, extra limb, missing limb, floating limbs, "
                      "disconnected limbs, malformed hands, blur, out of focus, long neck, "
                      "long body, disgusting, poorly drawn, mutilated, mangled, old, "
                      "heterochromia, dots, bad quality, weapons, NSFW, draft")
    
    # Use fixed seed for consistency in testing
    generator = torch.Generator(device=pipe.device)
    generator.manual_seed(123)  # Fixed seed for predictable results
    
    out = pipe(
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        image=processed_image,
        strength=strength,  # Very low strength to preserve structure
        guidance_scale=guidance_scale,  # Low guidance scale
        num_inference_steps=steps,
        generator=generator,
        output_type="pil"
    )
    
    # Ensure the output is valid
    result_image = out.images[0]
    
    # Convert to numpy array to check for issues
    import numpy as np
    arr = np.array(result_image)
    
    # If the image is completely black or has invalid values, return the original
    if arr.max() == 0 or np.isnan(arr).any() or np.isinf(arr).any():
        print(f"Warning: Stylization produced invalid output, returning original")
        return Image.open(img_path).convert('RGB')
    
    # Resize back to original dimensions if needed
    if result_image.size != original_size:
        result_image = result_image.resize(original_size, Image.Resampling.LANCZOS)
    
    # Minimal post-processing to avoid introducing artifacts
    from PIL import ImageEnhance
    
    # Very subtle enhancement only
    enhancer = ImageEnhance.Contrast(result_image)
    result_image = enhancer.enhance(1.02)  # Very minimal contrast boost
    
    return result_image

def stylize_anime_specific(img_path: str, strength=0.12, guidance_scale=3.5, steps=35):
    """
    Specialized function for anime transformation with ultra-conservative settings
    to preserve original structure while achieving anime style
    """
    pipe = get_img2img_pipe()
    init_image = Image.open(img_path).convert('RGB')
    original_size = init_image.size
    
    # Work with original size if reasonable, otherwise scale down minimally
    max_dimension = max(original_size)
    if max_dimension > 1024:
        scale_factor = 1024 / max_dimension
        new_width = int(original_size[0] * scale_factor)
        new_height = int(original_size[1] * scale_factor)
        init_image = init_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Ultra-specific anime prompt focusing on style only
    enhanced_prompt = ("anime art style, manga style, 2D illustration style, "
                      "clean lines, cel shading, vibrant colors, detailed, "
                      "preserve facial features, maintain proportions, high quality")
    
    # Comprehensive negative prompt
    negative_prompt = ("photorealistic, 3D, realistic, photograph, "
                      "distorted face, deformed, disfigured, extra limbs, "
                      "missing limbs, bad anatomy, blurry, low quality, "
                      "mutation, extra fingers, fewer fingers, "
                      "long neck, bad proportions, cropped, cut off")
    
    # Fixed seed for consistency
    generator = torch.Generator(device=pipe.device)
    generator.manual_seed(42)
    
    out = pipe(
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=strength,  # Ultra-low strength
        guidance_scale=guidance_scale,  # Very low guidance
        num_inference_steps=steps,
        generator=generator,
        output_type="pil"
    )
    
    result_image = out.images[0]
    
    # Validation
    import numpy as np
    arr = np.array(result_image)
    if arr.max() == 0 or np.isnan(arr).any() or np.isinf(arr).any():
        print("Warning: Anime stylization failed, returning original")
        return Image.open(img_path).convert('RGB')
    
    # Resize back to original if needed
    if result_image.size != original_size:
        result_image = result_image.resize(original_size, Image.Resampling.LANCZOS)
    
    return result_image
