# resize_image.py

import cv2
import numpy as np
from scipy import ndimage


class ResizeBitMap:
    """
    This class provides image processing methods for SVG export using Potrace.
    Uses resize-first approach for accurate scaling with filled contours.
    """

    def __init__(self, binary_image, scale):
        self.binary_image = binary_image
        self.scale = scale
        self.svg = None

    def resize_bitmap_to_dpi(self, target_dpi):
        """Improved version with better quality"""
        bitmap = self.binary_image
        current_dpi = self.scale

        # Calculate dimensions (your existing code)
        height, width = bitmap.shape[:2]
        physical_width_inches = width / current_dpi
        physical_height_inches = height / current_dpi
        new_width = int(physical_width_inches * target_dpi)
        new_height = int(physical_height_inches * target_dpi)

        scale_factor = target_dpi / current_dpi

        # Choose interpolation based on scale factor
        if scale_factor >= 1.0:
            # Upscaling: use INTER_LINEAR
            interpolation = cv2.INTER_LINEAR
        else:
            # Downscaling: use INTER_AREA (much better than NEAREST)
            interpolation = cv2.INTER_AREA

        # Resize
        resized_bitmap = cv2.resize(
            bitmap,
            (new_width, new_height),
            interpolation=interpolation
        )

        # Re-binarize to ensure clean black/white values
        _, resized_bitmap = cv2.threshold(
            resized_bitmap, 127, 255, cv2.THRESH_BINARY)

        return resized_bitmap
