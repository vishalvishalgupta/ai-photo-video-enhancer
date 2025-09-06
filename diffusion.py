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

def stylize_ghibli_art(img_path: str, precision_mode='ultra'):
    """
    ULTRA-PRECISE Studio Ghibli art transformation with advanced character-aware processing
    Creates authentic Ghibli-style artwork with perfect precision and magical quality
    
    Args:
        img_path: Path to input image
        precision_mode: 'ultra' for maximum precision, 'balanced' for speed/quality balance
    """
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    
    print("🎨✨ Creating ULTRA-PRECISE Studio Ghibli art with advanced character-aware processing...")
    
    # Load the original image
    image = Image.open(img_path).convert('RGB')
    original_size = image.size
    
    print(f"📐 Processing image: {original_size[0]}x{original_size[1]} in {precision_mode} mode")
    
    # Convert to numpy for processing
    img_array = np.array(image).astype(np.float32)
    
    # Advanced portrait/character detection with multiple algorithms
    def is_portrait_image(img, precision_mode='ultra'):
        """Advanced portrait detection with multiple algorithms for perfect precision"""
        h, w = img.shape[:2]
        aspect_ratio = w / h
        
        # Multi-factor portrait detection
        portrait_score = 0.0
        
        # Factor 1: Aspect ratio analysis
        if aspect_ratio < 1.3:  # Portrait or square-ish
            portrait_score += 0.3
        elif aspect_ratio < 1.6:  # Slightly wide but could be portrait
            portrait_score += 0.1
        
        # Factor 2: Enhanced skin tone detection
        center_h, center_w = h // 2, w // 2
        center_region = img[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4]
        
        skin_pixels = 0
        face_pixels = 0
        total_pixels = center_region.shape[0] * center_region.shape[1]
        
        for i in range(center_region.shape[0]):
            for j in range(center_region.shape[1]):
                r, g, b = center_region[i, j]
                
                # Enhanced skin tone detection with multiple ranges
                is_skin = False
                
                # Light skin tones
                if (r > 95 and g > 40 and b > 20 and r > g and r > b and abs(r - g) > 15):
                    is_skin = True
                # Medium skin tones
                elif (r > 80 and g > 50 and b > 30 and r > g and abs(r - g) > 10):
                    is_skin = True
                # Darker skin tones
                elif (r > 60 and g > 35 and b > 20 and r >= g and abs(r - g) < 25):
                    is_skin = True
                # Asian skin tones
                elif (r > 100 and g > 80 and b > 60 and abs(r - g) < 20 and abs(g - b) < 20):
                    is_skin = True
                
                if is_skin:
                    skin_pixels += 1
                
                # Face-like color detection (broader range)
                if ((r > 60 and g > 30 and b > 15) and 
                    (r + g + b > 150) and 
                    (abs(r - g) < 50)):
                    face_pixels += 1
        
        skin_ratio = skin_pixels / total_pixels
        face_ratio = face_pixels / total_pixels
        
        # Factor 3: Skin/face ratio scoring
        if skin_ratio > 0.15:  # Strong skin presence
            portrait_score += 0.4
        elif skin_ratio > 0.08:  # Moderate skin presence
            portrait_score += 0.2
        elif face_ratio > 0.25:  # Face-like colors
            portrait_score += 0.15
        
        # Factor 4: Color distribution analysis (faces have more varied colors)
        color_variance = np.var(center_region.reshape(-1, 3), axis=0).mean()
        if color_variance > 200:  # High color variation typical of faces
            portrait_score += 0.2
        elif color_variance > 100:
            portrait_score += 0.1
        
        # Factor 5: Edge density (faces have more edges than landscapes)
        if precision_mode == 'ultra':
            import cv2
            gray_center = cv2.cvtColor(center_region.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_center, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density > 0.1:  # High edge density
                portrait_score += 0.15
            elif edge_density > 0.05:
                portrait_score += 0.08
        
        # Decision threshold
        is_portrait = portrait_score > 0.5
        
        print(f"🔍 Portrait detection score: {portrait_score:.2f} -> {'Character/Portrait' if is_portrait else 'Landscape/General'}")
        return is_portrait
    
    is_character_photo = is_portrait_image(img_array, precision_mode)
    
    if is_character_photo:
        print("🎭 Applying ULTRA-PRECISE character-specific Ghibli transformation...")
        result_image = ghibli_character_transform(img_array, image, precision_mode)
    else:
        print("🏞️ Applying ULTRA-PRECISE landscape/general Ghibli transformation...")
        result_image = ghibli_landscape_transform(img_array, image, precision_mode)
    
    print("✅ ULTRA-PRECISE Studio Ghibli art transformation completed with perfect precision!")
    return result_image

def ghibli_character_transform(img_array, original_image, precision_mode='ultra'):
    """ULTRA-PRECISE Ghibli transformation - stunning impact with perfect face preservation"""
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
    import numpy as np
    import cv2
    
    print("👧🌟✨ Creating ULTRA-PRECISE Studio Ghibli Character ✨🌟👧")
    print(f"🎯 PERFECT PRECISION: Face preservation FIRST • {precision_mode.upper()} mode • Character identity intact")
    
    # Step 1: Advanced feature detection for microscopic precision
    def detect_ghibli_features_precision(img):
        """Ultra-precise feature detection for WOW-impact transformation"""
        try:
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # High-precision cascades
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # Multiple detection passes for accuracy
            faces = face_cascade.detectMultiScale(gray, 1.05, 6, minSize=(40, 40))
            
            if len(faces) > 0:
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = face
                
                # High-precision eye detection
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 4, minSize=(12, 8))
                eyes_full = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes]
                
                # Calculate precise face landmarks
                face_center = (x + w//2, y + h//2)
                eye_centers = [(ex + ew//2, ey + eh//2) for ex, ey, ew, eh in eyes_full]
                
                return {
                    'face': (x, y, w, h),
                    'face_center': face_center,
                    'eyes': eyes_full,
                    'eye_centers': eye_centers,
                    'detected': True
                }
        except:
            pass
        
        h, w = img.shape[:2]
        return {
            'face': (w//4, h//4, w//2, h//2),
            'face_center': (w//2, h//2),
            'eyes': [(w//3, h//3, w//8, h//12), (2*w//3, h//3, w//8, h//12)],
            'eye_centers': [(w//3 + w//16, h//3 + h//24), (2*w//3 + w//16, h//3 + h//24)],
            'detected': False
        }
    
    # Step 2: Create intelligent face protection zones
    def create_intelligent_protection(img, features):
        """Create intelligent protection zones to preserve character identity"""
        h, w = img.shape[:2]
        
        # Create comprehensive protection map
        protection_map = np.zeros((h, w), dtype=np.float32)
        
        fx, fy, fw, fh = features['face']
        face_center_x, face_center_y = features['face_center']
        
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # MAXIMUM protection for core facial features
        core_face_dist = np.sqrt((x_coords - face_center_x)**2 + (y_coords - face_center_y)**2)
        core_face_radius = min(fw, fh) * 0.35
        core_protection = np.exp(-(core_face_dist / (core_face_radius * 0.8))**2) * 0.9
        protection_map = np.maximum(protection_map, core_protection)
        
        # STRONG protection for eye regions
        for eye_center_x, eye_center_y in features['eye_centers']:
            eye_dist = np.sqrt((x_coords - eye_center_x)**2 + (y_coords - eye_center_y)**2)
            eye_radius = max(fw, fh) * 0.06
            eye_protection = np.exp(-(eye_dist / (eye_radius * 0.7))**2) * 0.8
            protection_map = np.maximum(protection_map, eye_protection)
        
        # MODERATE protection for extended face area
        extended_face_radius = max(fw, fh) * 0.5
        extended_protection = np.exp(-(core_face_dist / extended_face_radius)**2) * 0.4
        protection_map = np.maximum(protection_map, extended_protection)
        
        return protection_map
    
    # Step 3: Apply gentle Ghibli smoothing with protection
    def apply_protected_smoothing(img, protection_map):
        """Apply Ghibli smoothing while protecting important features"""
        result = img.copy().astype(np.float32)
        
        print("  🛡️ Applying protected Ghibli smoothing...")
        
        # Create multiple smoothing levels
        light_smooth = cv2.bilateralFilter(result.astype(np.uint8), 9, 50, 50).astype(np.float32)
        medium_smooth = cv2.bilateralFilter(result.astype(np.uint8), 12, 70, 70).astype(np.float32)
        
        # Apply smoothing based on protection level
        for i in range(3):
            # Where protection is high (0.7-1.0), use original
            # Where protection is medium (0.3-0.7), use light smoothing
            # Where protection is low (0-0.3), use medium smoothing
            high_protection = protection_map > 0.7
            medium_protection = (protection_map > 0.3) & (protection_map <= 0.7)
            low_protection = protection_map <= 0.3
            
            result[high_protection, i] = result[high_protection, i]
            result[medium_protection, i] = (result[medium_protection, i] * 0.7 + 
                                          light_smooth[medium_protection, i] * 0.3)
            result[low_protection, i] = (result[low_protection, i] * 0.4 + 
                                       medium_smooth[low_protection, i] * 0.6)
        
        return result
    
    # Step 4: Enhance eyes carefully with protection
    def enhance_eyes_carefully(img, features, protection_map):
        """Enhance eyes carefully while preserving natural appearance"""
        result = img.copy().astype(np.float32)
        h, w = result.shape[:2]
        
        print("  👁️✨ Carefully enhancing eyes...")
        
        y_coords, x_coords = np.ogrid[:h, :w]
        
        for eye_center_x, eye_center_y in features['eye_centers']:
            fx, fy, fw, fh = features['face']
            eye_radius = max(fw, fh) * 0.05  # Smaller, more conservative
            
            eye_dist = np.sqrt((x_coords - eye_center_x)**2 + (y_coords - eye_center_y)**2)
            
            # Very gentle eye enhancement
            eye_mask = np.exp(-(eye_dist / (eye_radius * 0.8))**2)
            
            # Reduce enhancement where protection is high
            effective_enhancement = eye_mask * (1.0 - protection_map * 0.7)
            
            # Apply very gentle enhancement
            for i in range(3):
                if i == 2:  # Blue for depth
                    result[:,:,i] += effective_enhancement * 12  # Much gentler
                elif i == 1:  # Green for life
                    result[:,:,i] += effective_enhancement * 8
                else:  # Red for warmth
                    result[:,:,i] += effective_enhancement * 6
            
            # Tiny sparkle only in center
            sparkle_dist = np.sqrt((x_coords - eye_center_x)**2 + (y_coords - eye_center_y)**2)
            sparkle_mask = np.exp(-(sparkle_dist / (eye_radius * 0.15))**2) * (1.0 - protection_map * 0.8)
            result += np.stack([sparkle_mask * 15] * 3, axis=2)  # Much gentler sparkle
        
        return np.clip(result, 0, 255)
    
    # Step 5: Add gentle rosy cheeks with protection
    def add_gentle_protected_cheeks(img, features, protection_map):
        """Add gentle rosy cheeks while respecting protection zones"""
        result = img.copy().astype(np.float32)
        h, w = result.shape[:2]
        
        print("  🌸😊 Adding gentle protected cheeks...")
        
        fx, fy, fw, fh = features['face']
        face_center_x, face_center_y = features['face_center']
        
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Calculate cheek positions
        cheek_y = face_center_y + fh//6
        cheek_offset = fw//4
        
        for cheek_x in [face_center_x - cheek_offset, face_center_x + cheek_offset]:
            cheek_dist = np.sqrt((x_coords - cheek_x)**2 + (y_coords - cheek_y)**2)
            cheek_radius = fw * 0.08  # Smaller radius
            
            # Gentle cheek enhancement
            cheek_mask = np.exp(-(cheek_dist / cheek_radius)**2)
            
            # Reduce intensity where protection is high
            effective_cheek = cheek_mask * (1.0 - protection_map * 0.6)
            
            # Apply very gentle blush
            result[:,:,0] += effective_cheek * 12  # Gentle rose
            result[:,:,1] += effective_cheek * 8   # Gentle pink
            result[:,:,2] += effective_cheek * 10  # Gentle warmth
        
        return np.clip(result, 0, 255)
    
    # Step 6: Apply gentle atmospheric glow
    def add_gentle_atmospheric_glow(img):
        """Add gentle atmospheric glow that doesn't overpower the character"""
        result = img.copy().astype(np.float32)
        h, w = result.shape[:2]
        
        print("  ✨🌟 Adding gentle atmospheric glow...")
        
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Very gentle center glow
        center_x, center_y = w//2, h//2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        center_dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        gentle_glow = 1.0 + (1.0 - center_dist / max_dist) * 0.06  # Much gentler
        
        # Subtle upper-left warm light
        ul_x, ul_y = w//4, h//4
        ul_dist = np.sqrt((x_coords - ul_x)**2 + (y_coords - ul_y)**2)
        ul_glow = 1.0 + np.exp(-(ul_dist / (max_dist * 0.8))**2) * 0.04
        
        # Combine gentle lighting
        total_glow = gentle_glow * ul_glow
        
        # Apply very gentle color temperature
        result[:,:,0] *= total_glow * 1.03  # Minimal warm boost
        result[:,:,1] *= total_glow * 1.02  # Minimal natural boost
        result[:,:,2] *= total_glow * 0.99  # Minimal cool reduction
        
        return np.clip(result, 0, 255)
    
    # Step 7: Apply balanced color enhancement
    def apply_balanced_color_enhancement(img):
        """Apply balanced color enhancement that preserves character while adding impact"""
        result = img.copy()
        
        print("  🎨🌈 Applying balanced color enhancement...")
        
        # Conservative color quantization
        data = result.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        k = 22  # More colors to preserve character details
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
        
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()].reshape(result.shape)
        
        # Very conservative blending - mostly preserve original
        result = quantized.astype(np.float32) * 0.4 + result.astype(np.float32) * 0.6
        
        return result
    
    # Execute BALANCED transformation with protection
    print("- 🔍 Precise feature detection...")
    features = detect_ghibli_features_precision(img_array)
    print(f"- Face: {features['detected']}, Eyes: {len(features['eye_centers'])}")
    
    print("- 🛡️ Creating intelligent protection zones...")
    protection_map = create_intelligent_protection(img_array, features)
    
    print("- 🎨 Applying protected smoothing...")
    protected_smooth = apply_protected_smoothing(img_array, protection_map)
    
    print("- 👁️ Carefully enhancing eyes...")
    enhanced_eyes = enhance_eyes_carefully(protected_smooth, features, protection_map)
    
    print("- 🌸 Adding gentle protected cheeks...")
    gentle_cheeks = add_gentle_protected_cheeks(enhanced_eyes, features, protection_map)
    
    print("- ✨ Adding gentle atmospheric glow...")
    gentle_glow = add_gentle_atmospheric_glow(gentle_cheeks)
    
    print("- 🎨 Applying balanced color enhancement...")
    balanced_colors = apply_balanced_color_enhancement(gentle_glow)
    
    # Convert to PIL for final balanced polish
    result_image = Image.fromarray(balanced_colors.astype(np.uint8))
    
    print("- 🌟 Final balanced polish...")
    
    # Balanced enhancements that preserve character while adding impact
    enhancer = ImageEnhance.Color(result_image)
    result_image = enhancer.enhance(1.25)  # Moderate color enhancement
    
    enhancer = ImageEnhance.Brightness(result_image)
    result_image = enhancer.enhance(1.05)  # Gentle brightness
    
    enhancer = ImageEnhance.Contrast(result_image)
    result_image = enhancer.enhance(1.08)  # Subtle contrast
    
    # Light sharpening for definition
    result_image = result_image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=115, threshold=3))
    
    return result_image

def ghibli_landscape_transform(img_array, original_image, precision_mode='ultra'):
    """ULTRA-PRECISE Ghibli transformation for landscape/general photos"""
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    import cv2
    
    print(f"🏞️✨ Creating ULTRA-PRECISE Studio Ghibli Landscape in {precision_mode.upper()} mode")
    
    # Enhanced transformation for landscapes with precision control
    result = img_array.copy()
    
    # Precision-based parameter adjustment
    if precision_mode == 'ultra':
        levels = 12  # More color levels for ultra precision
        blur_iterations = 4
        edge_threshold_low = 30
        edge_threshold_high = 120
    else:
        levels = 10  # Standard levels
        blur_iterations = 3
        edge_threshold_low = 50
        edge_threshold_high = 150
    
    # Precision-controlled color quantization for painted look
    normalized = result / 255.0
    quantized = np.round(normalized * levels) / levels
    result = (quantized * 255).astype(np.float32)
    
    # Enhanced watercolor effect with precision control
    pil_img = Image.fromarray(result.astype(np.uint8))
    
    for iteration in range(blur_iterations):
        # Adaptive blur radius based on iteration and precision
        blur_radius = 1.2 + (iteration * 0.3) if precision_mode == 'ultra' else 1.5
        
        blurred = np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius)))
        
        # Enhanced edge detection with precision parameters
        edges = cv2.Canny(np.array(pil_img.convert('L')), edge_threshold_low, edge_threshold_high)
        edge_strength = edges / 255.0
        
        # Apply Ghibli-style cel shading effect
        for i in range(3):
            # More sophisticated blending for ultra precision
            if precision_mode == 'ultra':
                # Multi-level blending for smoother transitions
                blend_factor = 0.4 * (1.0 - edge_strength) * (1.0 - iteration * 0.1)
                result[:,:,i] = result[:,:,i] * (1 - blend_factor) + blurred[:,:,i] * blend_factor
            else:
                blend_factor = 0.5 * (1.0 - edge_strength)
                result[:,:,i] = result[:,:,i] * (1 - blend_factor) + blurred[:,:,i] * blend_factor
        
        pil_img = Image.fromarray(result.astype(np.uint8))
    
    # Add Ghibli-style atmospheric effects for landscapes
    if precision_mode == 'ultra':
        # Add subtle color temperature shifts typical of Ghibli landscapes
        h, w = result.shape[:2]
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Warm light from upper areas (sky)
        sky_gradient = 1.0 - (y_coords / h) * 0.3
        result[:,:,0] *= sky_gradient * 1.05  # Warm reds
        result[:,:,1] *= sky_gradient * 1.03  # Warm yellows
        
        # Cool shadows in lower areas
        ground_gradient = (y_coords / h) * 0.2 + 0.9
        result[:,:,2] *= ground_gradient * 1.02  # Cool blues in shadows
        
        result = np.clip(result, 0, 255)
    
    # Convert to PIL for final enhancements
    result_image = Image.fromarray(result.astype(np.uint8))
    
    # Precision-controlled landscape-specific enhancements
    if precision_mode == 'ultra':
        print("  🎨 Applying ULTRA-PRECISE landscape enhancements...")
        
        # Ultra-precise color enhancement
        enhancer = ImageEnhance.Color(result_image)
        result_image = enhancer.enhance(1.42)  # More vibrant for ultra mode
        
        # Magical brightness with precision
        enhancer = ImageEnhance.Brightness(result_image)
        result_image = enhancer.enhance(1.12)  # Brighter magical feel
        
        # Enhanced contrast for definition
        enhancer = ImageEnhance.Contrast(result_image)
        result_image = enhancer.enhance(1.08)  # Better contrast
        
        # Ultra-precise sharpening for Ghibli clarity
        result_image = result_image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=125, threshold=2))
        
    else:
        print("  🎨 Applying balanced landscape enhancements...")
        
        # Standard enhancements
        enhancer = ImageEnhance.Color(result_image)
        result_image = enhancer.enhance(1.35)  # Vibrant for landscapes
        
        enhancer = ImageEnhance.Brightness(result_image)
        result_image = enhancer.enhance(1.08)  # Brighter for magical feel
        
        enhancer = ImageEnhance.Contrast(result_image)
        result_image = enhancer.enhance(1.06)  # Gentle contrast
    
    print(f"✅ ULTRA-PRECISE Studio Ghibli landscape transformation completed in {precision_mode.upper()} mode!")
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
