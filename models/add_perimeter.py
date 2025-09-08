# add_perimeter.py

import cv2
import numpy as np


class AddPerimeter:
    """
    Class to handle bitmap perimeter expansion with millimeter specification

    Args:
        resized_bitmap: Pre-scaled bitmap at known DPI
        dpi: The DPI of the resized bitmap
    """

    def __init__(self, resized_bitmap, dpi):
        """
        Initialize the perimeter expander

        Args:
            resized_bitmap: Binary image (0s and 255s) already resized to target DPI
            dpi: The DPI of the resized bitmap
        """
        # Input validation
        if resized_bitmap is None:
            raise ValueError("resized_bitmap cannot be None")

        if dpi <= 0:
            raise ValueError(f"DPI must be positive, got: {dpi}")

        # Ensure the image is binary (0 and 255)
        if len(resized_bitmap.shape) == 3:
            resized_bitmap = cv2.cvtColor(resized_bitmap, cv2.COLOR_BGR2GRAY)

        # Convert to binary if not already
        _, self.bitmap = cv2.threshold(
            resized_bitmap, 127, 255, cv2.THRESH_BINARY)
        self.original_bitmap = self.bitmap.copy()
        self.dpi = dpi

    def mm_to_pixels(self, mm):
        """
        Convert millimeters to pixels based on current DPI

        Args:
            mm: Distance in millimeters

        Returns:
            int: Distance in pixels
        """
        if mm < 0:
            print(f"Warning: Negative mm value: {mm}")
            return 0

        # Convert mm to inches, then to pixels
        inches = mm / 25.4  # 25.4 mm per inch
        pixels = int(inches * self.dpi)

        print(f"Converting {mm}mm to {pixels} pixels at {self.dpi} DPI")
        return pixels

    def expand_black_areas(self, buffer_mm):
        """
        Expand black areas by specified millimeter buffer

        Args:
            buffer_mm: Buffer size in millimeters

        Returns:
            numpy.ndarray: Expanded bitmap
        """
        # Convert mm to pixels
        buffer_pixels = self.mm_to_pixels(buffer_mm)

        if buffer_pixels <= 0:
            print("Warning: Buffer too small, returning original bitmap")
            return self.bitmap.copy()  # Return a copy to avoid modifying original

        print(
            f"Expanding black areas by {buffer_mm}mm ({buffer_pixels} pixels)")

        # Create circular kernel for natural expansion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2*buffer_pixels + 1, 2*buffer_pixels + 1))

        # To expand black areas: invert, dilate, then invert back
        inverted = cv2.bitwise_not(self.bitmap)
        expanded_inverted = cv2.dilate(inverted, kernel, iterations=1)
        expanded = cv2.bitwise_not(expanded_inverted)

        return expanded
