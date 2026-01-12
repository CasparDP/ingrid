"""Image preprocessing for document scans.

Provides deskewing (auto-rotation correction), contrast enhancement, and other
image transformations to improve OCR/HTR results.
"""

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

from .models import PreprocessedImage

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocess document scan images for better OCR/HTR results.

    This class handles image transformations including:
    - Auto-rotation (deskewing) using Hough transform
    - Contrast and sharpness enhancement
    - Optional DPI resampling

    Attributes:
        deskew: Enable automatic deskewing.
        enhance_contrast: Enable contrast enhancement.
        target_dpi: Target DPI for resampling (None = keep original).
        max_angle: Maximum rotation angle to consider (degrees).
    """

    def __init__(
        self,
        deskew: bool = True,
        enhance_contrast: bool = True,
        target_dpi: int | None = None,
        max_angle: float = 45.0,
    ) -> None:
        """Initialize preprocessor.

        Args:
            deskew: Enable automatic deskewing.
            enhance_contrast: Enable contrast enhancement.
            target_dpi: Resample to target DPI (None = keep original).
            max_angle: Maximum rotation angle to consider (degrees).
        """
        self.deskew = deskew
        self.enhance_contrast = enhance_contrast
        self.target_dpi = target_dpi
        self.max_angle = max_angle

    def process(self, image_path: Path | str) -> PreprocessedImage:
        """Preprocess a single image.

        Args:
            image_path: Path to input image.

        Returns:
            PreprocessedImage with enhanced PIL Image.

        Raises:
            FileNotFoundError: If image doesn't exist.
            ValueError: If image cannot be loaded.
        """
        start_time = time.time()
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            # Load image
            image = Image.open(image_path)
            original_size = image.size
            file_size = image_path.stat().st_size

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            result = PreprocessedImage(
                original_path=image_path,
                image=image,
                original_size=original_size,
                processed_size=image.size,
                file_size_bytes=file_size,
            )

            # Apply transformations
            if self.deskew:
                image, angle = self._deskew_image(image)
                result.image = image
                result.deskewed = True
                result.deskew_angle = angle
                logger.debug(f"Deskewed by {angle:.2f} degrees")

            if self.enhance_contrast:
                image = self._enhance_contrast(image)
                result.image = image
                result.contrast_enhanced = True
                logger.debug("Enhanced contrast")

            result.processed_size = image.size
            result.processing_time = time.time() - start_time

            logger.info(f"Preprocessed {image_path.name} in {result.processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Failed to preprocess {image_path}: {e}")
            raise ValueError(f"Image preprocessing failed: {e}") from e

    def _deskew_image(self, image: Image.Image) -> tuple[Image.Image, float]:
        """Detect and correct skew/rotation.

        Uses Hough transform to detect dominant line angles in the image.

        Args:
            image: PIL Image to deskew.

        Returns:
            Tuple of (deskewed image, rotation angle in degrees).
        """
        try:
            # Convert to grayscale numpy array
            gray = rgb2gray(np.array(image))

            # Edge detection
            edges = canny(gray, sigma=3.0)

            # Hough transform to detect lines
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
            h, theta, d = hough_line(edges, theta=tested_angles)

            # Find dominant angle
            _, angles, _ = hough_line_peaks(h, theta, d, num_peaks=5)

            if len(angles) == 0:
                return image, 0.0

            # Calculate median angle (more robust than mean)
            angle_degrees = np.median(np.degrees(angles))

            # Normalize to [-45, 45] range
            if angle_degrees > 45:
                angle_degrees -= 90
            elif angle_degrees < -45:
                angle_degrees += 90

            # Only rotate if angle is significant
            if abs(angle_degrees) < 0.5:
                return image, 0.0

            # Rotate image
            rotated = image.rotate(
                angle_degrees,
                resample=Image.BICUBIC,
                expand=True,
                fillcolor=(255, 255, 255),
            )

            return rotated, angle_degrees

        except Exception as e:
            logger.warning(f"Deskew failed: {e}, using original image")
            return image, 0.0

    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Enhance image contrast for better OCR.

        Args:
            image: PIL Image to enhance.

        Returns:
            Enhanced PIL Image.
        """
        # Increase contrast (factor 1.5)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        # Increase sharpness slightly (factor 1.2)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)

        return image
