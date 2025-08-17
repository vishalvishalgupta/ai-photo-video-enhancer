import cv2
import numpy as np

def auto_contrast_brightness(img, clip_hist_percent=1.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray],[0],None,[256],[0,256])
    accumulator = histogram.cumsum()
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    minimum_gray = np.searchsorted(accumulator, clip_hist_percent)
    maximum_gray = np.searchsorted(accumulator, maximum-clip_hist_percent)

    alpha = 255 / (maximum_gray - minimum_gray + 1e-5)
    beta = -minimum_gray * alpha
    auto_result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return auto_result

def denoise(img, strength=10):
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)

def sharpen(img, amount=1.0):
    blurred = cv2.GaussianBlur(img, (0,0), 3)
    return cv2.addWeighted(img, 1+amount, blurred, -amount, 0)

def upscale(img, scale=2):
    h, w = img.shape[:2]
    return cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

def enhance_pipeline(img, do_upscale=True):
    x = auto_contrast_brightness(img)
    x = denoise(x, strength=7)
    x = sharpen(x, amount=0.6)
    if do_upscale:
        x = upscale(x, scale=2)
    return x
