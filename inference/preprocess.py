"""
Image preprocessing utilities for OCR pipeline.
Handles letterbox resize, grayscale conversion, and stride-32 padding.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Union, Optional


def letterbox_resize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    stride: int = 32
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with letterbox padding to maintain aspect ratio.
    
    Args:
        image: Input image (H, W, C) or (H, W)
        target_size: Target (height, width)
        color: Padding color
        auto: Minimum rectangle padding
        scale_fill: Stretch to fill (no padding)
        stride: Stride for padding alignment
    
    Returns:
        Tuple of (padded_image, scale_ratio, (pad_w, pad_h))
    """
    shape = image.shape[:2]  # current shape [height, width]
    
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    # Scale ratio (new / old)
    r = min(target_size[0] / shape[0], target_size[1] / shape[1])
    
    # Compute new unpadded size
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = target_size[1] - new_unpad[0], target_size[0] - new_unpad[1]
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (target_size[1], target_size[0])
        r = target_size[1] / shape[1], target_size[0] / shape[0]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=color)
    
    return image, r, (int(dw), int(dh))


def stride32_pad(
    image: np.ndarray,
    stride: int = 32,
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad image to be divisible by stride.
    
    Args:
        image: Input image
        stride: Stride value (default 32 for YOLO)
        color: Padding color
    
    Returns:
        Tuple of (padded_image, original_size)
    """
    h, w = image.shape[:2]
    original_size = (h, w)
    
    new_h = int(np.ceil(h / stride) * stride)
    new_w = int(np.ceil(w / stride) * stride)
    
    pad_h = new_h - h
    pad_w = new_w - w
    
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=color
        )
    
    return image, original_size


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def normalize_for_ocr(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalize image for OCR models (ImageNet normalization).
    
    Args:
        image: Input image (H, W, C) in range [0, 255]
        mean: Channel means
        std: Channel stds
    
    Returns:
        Normalized image in range suitable for model input
    """
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(mean)) / np.array(std)
    return image


def deskew_image(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Detect and correct skew in document image.
    
    Args:
        image: Input grayscale image
        max_angle: Maximum skew angle to correct
    
    Returns:
        Deskewed image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find coordinates of non-zero pixels
    coords = np.column_stack(np.where(binary > 0))
    
    if len(coords) < 10:
        return image
    
    # Get rotation angle from minAreaRect
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    
    # Adjust angle
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    
    # Clamp to max angle
    if abs(angle) > max_angle:
        return image
    
    # Rotate
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    if len(image.shape) == 3:
        rotated = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    else:
        rotated = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    if len(image.shape) == 3:
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)


def binarize_adaptive(
    image: np.ndarray,
    block_size: int = 11,
    c: int = 2
) -> np.ndarray:
    """Apply adaptive thresholding for binarization."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c
    )


# ============================================================================
# ADVANCED PREPROCESSING FUNCTIONS (Phase 1 Optimization)
# ============================================================================

def remove_noise(
    image: np.ndarray,
    method: str = "bilateral",
    strength: int = 10
) -> np.ndarray:
    """
    Remove noise from image using various filtering methods.
    
    Args:
        image: Input image
        method: 'median', 'bilateral', 'gaussian', or 'nlmeans'
        strength: Filter strength (kernel size for median/gaussian, or d for bilateral)
    
    Returns:
        Denoised image
    """
    if method == "median":
        kernel_size = strength if strength % 2 == 1 else strength + 1
        return cv2.medianBlur(image, kernel_size)
    
    elif method == "bilateral":
        # Bilateral filter preserves edges while removing noise
        return cv2.bilateralFilter(image, d=strength, sigmaColor=75, sigmaSpace=75)
    
    elif method == "gaussian":
        kernel_size = strength if strength % 2 == 1 else strength + 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif method == "nlmeans":
        # Non-local means denoising (slower but better quality)
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    
    return image


def deskew_hough(
    image: np.ndarray,
    max_angle: float = 15.0,
    threshold_ratio: float = 0.3
) -> np.ndarray:
    """
    Detect and correct skew using Hough Transform line detection.
    More accurate than minAreaRect for documents with clear text lines.
    
    Args:
        image: Input image
        max_angle: Maximum skew angle to correct
        threshold_ratio: Hough threshold as ratio of image diagonal
    
    Returns:
        Deskewed image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough line detection
    h, w = gray.shape
    threshold = int(min(h, w) * threshold_ratio)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    
    if lines is None or len(lines) == 0:
        return image
    
    # Collect angles near horizontal (text lines)
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90  # Convert to degrees from horizontal
        if abs(angle) <= max_angle:
            angles.append(angle)
    
    if not angles:
        return image
    
    # Use median angle for robustness
    median_angle = np.median(angles)
    
    if abs(median_angle) < 0.5:  # Skip tiny corrections
        return image
    
    # Rotate image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated


def normalize_resolution(
    image: np.ndarray,
    target_dpi: int = 300,
    current_dpi: Optional[int] = None,
    target_height: Optional[int] = None
) -> np.ndarray:
    """
    Normalize image resolution for consistent OCR performance.
    
    Args:
        image: Input image
        target_dpi: Target DPI (if current_dpi is known)
        current_dpi: Current image DPI (if known)
        target_height: Target height in pixels (alternative to DPI-based scaling)
    
    Returns:
        Resolution-normalized image
    """
    h, w = image.shape[:2]
    
    if target_height is not None:
        # Scale to target height
        scale = target_height / h
    elif current_dpi is not None and current_dpi != target_dpi:
        # DPI-based scaling
        scale = target_dpi / current_dpi
    else:
        # Heuristic: assume 72 DPI for low-res images, scale up
        if h < 100:
            scale = 300 / 72  # Assume 72 DPI, scale to 300
        elif h < 200:
            scale = 1.5
        else:
            return image  # Already reasonable resolution
    
    if abs(scale - 1.0) < 0.1:
        return image
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Use INTER_CUBIC for upscaling, INTER_AREA for downscaling
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def horizontal_projection_profile(image: np.ndarray) -> np.ndarray:
    """
    Compute horizontal projection profile for baseline detection.
    
    Args:
        image: Grayscale or binary image
    
    Returns:
        1D array of row sums (projection profile)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Invert if needed (text should be dark)
    if np.mean(gray) > 127:
        gray = 255 - gray
    
    # Sum along rows
    profile = np.sum(gray, axis=1)
    
    return profile


def detect_baselines(
    image: np.ndarray,
    min_line_gap: int = 10,
    smooth_window: int = 5
) -> list:
    """
    Detect text baselines using horizontal projection profiling.
    
    Args:
        image: Input image
        min_line_gap: Minimum gap between detected lines
        smooth_window: Smoothing window for profile
    
    Returns:
        List of baseline y-coordinates
    """
    profile = horizontal_projection_profile(image)
    
    # Smooth profile
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(profile, kernel, mode='same')
    
    # Find peaks (text lines)
    threshold = np.mean(smoothed) + 0.5 * np.std(smoothed)
    
    baselines = []
    in_line = False
    line_start = 0
    
    for i, val in enumerate(smoothed):
        if val > threshold and not in_line:
            in_line = True
            line_start = i
        elif val <= threshold and in_line:
            in_line = False
            line_center = (line_start + i) // 2
            
            # Check minimum gap from previous baseline
            if not baselines or (line_center - baselines[-1]) >= min_line_gap:
                baselines.append(line_center)
    
    return baselines


def segment_text_lines(
    image: np.ndarray,
    padding: int = 5
) -> list:
    """
    Segment image into text lines using projection profiling.
    
    Args:
        image: Input image
        padding: Padding around each line segment
    
    Returns:
        List of (y1, y2) tuples for each text line
    """
    profile = horizontal_projection_profile(image)
    
    # Smooth and threshold
    kernel = np.ones(3) / 3
    smoothed = np.convolve(profile, kernel, mode='same')
    threshold = np.mean(smoothed) * 0.3
    
    lines = []
    in_line = False
    line_start = 0
    
    for i, val in enumerate(smoothed):
        if val > threshold and not in_line:
            in_line = True
            line_start = max(0, i - padding)
        elif val <= threshold and in_line:
            in_line = False
            line_end = min(len(profile), i + padding)
            if line_end - line_start > 10:  # Minimum line height
                lines.append((line_start, line_end))
    
    # Handle last line if still in_line
    if in_line:
        lines.append((line_start, len(profile)))
    
    return lines


def full_preprocessing_pipeline(
    image: np.ndarray,
    denoise: bool = True,
    deskew: bool = True,
    normalize_res: bool = True,
    enhance: bool = True,
    denoise_method: str = "bilateral",
    target_height: Optional[int] = None
) -> np.ndarray:
    """
    Full preprocessing pipeline combining all optimizations.
    
    Args:
        image: Input image
        denoise: Apply noise removal
        deskew: Apply Hough-based skew correction
        normalize_res: Normalize resolution
        enhance: Apply contrast enhancement
        denoise_method: 'median', 'bilateral', 'gaussian', 'nlmeans'
        target_height: Optional target height for normalization
    
    Returns:
        Fully preprocessed image
    """
    result = image.copy()
    
    # 1. Resolution normalization (do first to standardize input)
    if normalize_res:
        result = normalize_resolution(result, target_height=target_height)
    
    # 2. Noise removal
    if denoise:
        result = remove_noise(result, method=denoise_method)
    
    # 3. Skew correction
    if deskew:
        result = deskew_hough(result)
    
    # 4. Contrast enhancement
    if enhance:
        result = enhance_contrast(result)
    
    return result


def preprocess_for_layout(
    image: Union[np.ndarray, Image.Image, str],
    target_size: int = 640,
    stride: int = 32
) -> Tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline for layout detection.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        target_size: Target size for layout model
        stride: Stride for padding
    
    Returns:
        Tuple of (preprocessed_image, metadata_dict)
    """
    # Load image
    if isinstance(image, str):
        img = cv2.imread(image)
    elif isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()
    
    original_shape = img.shape[:2]
    
    # Deskew
    deskewed = deskew_image(img)
    
    # Letterbox resize
    resized, ratio, pad = letterbox_resize(
        deskewed, 
        target_size=(target_size, target_size),
        stride=stride
    )
    
    # Normalize for YOLO (0-1 range)
    normalized = resized.astype(np.float32) / 255.0
    
    # HWC to CHW
    normalized = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension
    normalized = np.expand_dims(normalized, 0)
    
    metadata = {
        "original_shape": original_shape,
        "ratio": ratio,
        "pad": pad,
        "resized_shape": resized.shape[:2]
    }
    
    return normalized, metadata


def preprocess_for_ocr(
    image: Union[np.ndarray, Image.Image, str],
    target_size: Optional[Tuple[int, int]] = None,
    grayscale: bool = False
) -> np.ndarray:
    """
    Preprocessing for TrOCR input.
    
    Args:
        image: Input image
        target_size: Optional resize target (H, W)
        grayscale: Whether to convert to grayscale
    
    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            img = Image.fromarray(image).convert('RGB')
        else:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        img = image.convert('RGB')
    
    if target_size:
        img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
    
    if grayscale:
        img = img.convert('L').convert('RGB')
    
    return np.array(img)


def crop_region(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 5
) -> np.ndarray:
    """
    Crop a region from image with optional padding.
    
    Args:
        image: Source image
        bbox: Bounding box (x1, y1, x2, y2)
        padding: Padding pixels
    
    Returns:
        Cropped image region
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Add padding and clamp
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image[y1:y2, x1:x2]


def remove_underlines(image: np.ndarray) -> np.ndarray:
    """
    Remove horizontal lines from image (grayscale or RGB).
    """
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_color = True
    else:
        gray = image
        is_color = False
        
    h, w = gray.shape
    # Line needs to be at least 20% of width
    line_min_width = int(w * 0.2)
    if line_min_width < 20: line_min_width = 20
    
    # Invert for morphological operations
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_width, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Mask out top half to protect characters
    mask = np.zeros_like(detected_lines)
    mask[int(h*0.4):, :] = 255 
    detected_lines = cv2.bitwise_and(detected_lines, mask)
    
    # Dilation to cover artifacts
    dilated_lines = cv2.dilate(detected_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
    
    # Inpaint
    if is_color:
        inpainted = cv2.inpaint(image, dilated_lines, 3, cv2.INPAINT_TELEA)
    else:
        inpainted = cv2.inpaint(gray, dilated_lines, 3, cv2.INPAINT_TELEA)
        
    return inpainted


def boost_dots(image: np.ndarray) -> np.ndarray:
    """
    Boost small dots in image.
    """
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_color = True
    else:
        gray = image
        is_color = False
        
    # Invert
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    output_mask = np.zeros_like(binary)
    has_dots = False
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Heuristic for "dot"
        if 5 <= area <= 60:
            output_mask[labels == i] = 255
            has_dots = True
            
    if not has_dots:
        return image
        
    # Dilate the dots
    kernel = np.ones((2,2), np.uint8)
    dilated_dots = cv2.dilate(output_mask, kernel, iterations=1)
    
    # Apply to image (burn consistent black)
    result = image.copy()
    
    if is_color:
        # Set all channels to 0 where dilated dots are
        result[dilated_dots == 255] = [0, 0, 0] # Black
    else:
        result[dilated_dots == 255] = 0
    
    return result
