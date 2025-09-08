# detect_contours.py

import numpy as np
import cv2
import tkinter
from tkinter import messagebox
from utils.message_bus import MessageBus, Message


class DetectContours:
    """
    This class provides all the functionality that would involve detecting the contours in
    a loaded image, including:

    - Detect contours in the image using either brightness thresholding or Canny edge detection
    - Returning a bitmap image of the contours to be used for svg conversion 
    - Detect Aruco markers in the image 
    - Isolate contours that are not ArUco markers AND are within the perimeter of ArUco markers
    - Calculate the scale of the image based on ArUco markers
    - Return a scale factor in pixels per inch assuming the markers are 1 inch by 1 inch

    NOTE: This version ONLY detects contours within the perimeter defined by ArUco markers.
    At least 4 ArUco markers are required to define the detection area.
    """

    def __init__(self, controller):
        self.image = None
        self.controller = controller
        self.bus = self.controller.bus

        self.bus.subscribe('controller.process_request',
                           self.on_process_request)
        self.bus.subscribe('controller.brightness_changed',
                           self.on_brightness_change)
        self.bus.subscribe('controller.reset_request', self.on_reset_request)
        self.bus.subscribe(
            'controller.detection_method_changed', self.on_method_change)
        self.bus.subscribe('controller.tolerance_changed',
                           self.on_tolerance_change)
        self.bus.subscribe('controller.canny_low_threshold_changed',
                           self.on_canny_low_change)
        self.bus.subscribe('controller.canny_high_threshold_changed',
                           self.on_canny_high_change)

        # Initialize processing parameters
        self.detection_method = "brightness"
        self.brightness_threshold = 100
        self.brightness_blur_kernel = (5, 5)
        self.brightness_min_area = 100
        self.brightness_max_area = None

        self.canny_low_threshold = 15
        self.canny_high_threshold =80
        self.canny_blur_kernel = (5, 5)
        self.canny_min_area = 100
        self.canny_max_area = None

        self.min_contour_area = 100
        self.current_tolerance_mm = 0.0

        # Results storage
        self.contours = []
        self.non_marker_contours = []
        self.bitmap = None
        self.pixels_per_inch = None

        self.marker_corners = None
        self.marker_ids = None
        self.count_markers = 0

        self.tolerance_contours = None
        self.tolerance_bitmap = None

        self.binary_image = None
        self.hierarchy = None
        self.canny_edges = None

        self.dimensions_inches = "0 in x 0 in"
        self.dimensions_mm = "0 mm x 0 mm"
        self.dimensions_pixels = "0 px x 0 px"

    def on_process_request(self, message: Message):
        self.image = message.data
        if self.image is None:
            print("No image received for processing.")
            error_msg = "No image received for processing.\n\nPlease load or capture an image first."
            messagebox.showerror("Image Processing Error", error_msg)
        else:
            self.process()

    def on_reset_request(self, message: Message):
        # Reset all variables
        self.image = None
        self.contours = []
        self.bitmap = None
        self.pixels_per_inch = None
        self.count_markers = 0

        self.non_marker_contours = []
        self.marker_corners = None
        self.marker_ids = None

        # Reset tolerance data
        self.tolerance_contours = []
        self.tolerance_bitmap = None
        

        self.binary_image = None
        self.hierarchy = None
        self.canny_edges = None

        self.dimensions_inches = "0 in x 0 in"
        self.dimensions_mm = "0 mm x 0 mm"
        self.dimensions_pixels = "0 px x 0 px"

    def on_method_change(self, message: Message):
        """Receives a new detection method and reprocesses the bitmap"""
        self.detection_method = message.data
        self.process()

    def on_brightness_change(self, message: Message):
        """Receives a new brightness threshold parameter and reprocesses the bitmap"""
        if self.image is not None and self.detection_method == 'brightness':
            self.brightness_threshold = message.data
            self.process()
        else:
            print("Error Processing Image. Please make sure an image is loaded and ArUco markers are clearly visible.")
    
    def on_canny_low_change(self, message: Message):
        """Handle Canny low threshold change and reprocess the image"""
        if self.image is not None and self.detection_method == 'canny':
            self.canny_low_threshold = message.data
            self.process()

    def on_canny_high_change(self, message: Message):
        """Handle Canny high threshold change and reprocess the image"""
        if self.image is not None and self.detection_method == 'canny':
            self.canny_high_threshold = message.data
            self.process()

    def on_tolerance_change(self, message: Message):
        """Handle tolerance change and reprocess tolerance contours"""
        self.current_tolerance_mm = message.data

        # Only reprocess if we have base contours
        if self.non_marker_contours and self.pixels_per_inch is not None:
            self.update_tolerance_processing()
            self.publish_contour_data()

    def process(self):
        """Main processing method - ALWAYS uses perimeter detection"""
        self.marker_corners, self.marker_ids, _ = self._detect_aruco_markers()
        self.pixels_per_inch = self._calculate_scale()
        

        try:
            if self.pixels_per_inch is None:
                print("Could not calculate scale - no markers detected")
                return None, None, None

            # Always use perimeter detection
            if self.detection_method == "brightness":
                self._process_brightness()
            elif self.detection_method == "canny":
                self._process_canny()
            else:
                raise ValueError(f"Unknown method: {self.detection_method}")
            
            if self.non_marker_contours is None or len(self.non_marker_contours) == 0:
                print("No valid contours found after processing.")
                
                return None, None, None
            
            else:
                # Process tolerance after base processing
                self.update_tolerance_processing()

                # Calculate tool dimensions
                self.calculate_tool_dimensions()

                # Publish combined data
                self.publish_contour_data()

            return self.bitmap, self.pixels_per_inch, self.non_marker_contours

        except Exception as e:
            print(f"Processing error: {e}")
            return None, None, None

    def publish_contour_data(self):
        """Publish both original and tolerance contour data"""
        contour_data = {
            'original_contours': self.non_marker_contours,
            'tolerance_contours': self.tolerance_contours,
            'original_bitmap': self.bitmap,
            'tolerance_bitmap': self.tolerance_bitmap,
            'pixels_per_inch': self.pixels_per_inch,
            'current_tolerance_mm': self.current_tolerance_mm,
            "count_markers": self.count_markers,
            'dimensions_inches': self.dimensions_inches,
            'dimensions_mm': self.dimensions_mm,
            'dimensions_pixels': self.dimensions_pixels
        }

        self.bus.publish('model.contour_data', contour_data, 'DetectContours')

    def _process_brightness(self):
        """Process using brightness thresholding with perimeter constraint"""
        self.contours = self._find_contours_brightness(
            threshold=self.brightness_threshold,
            blur_kernel=self.brightness_blur_kernel,
            min_area=self.brightness_min_area,
            max_area=self.brightness_max_area
        )

        self.non_marker_contours, self.bitmap = self._isolate_non_aruco_contours(
            min_contour_area=self.min_contour_area
        )

    def _process_canny(self):
        """Process using Canny edge detection with perimeter constraint"""
        self.contours = self._find_contours_canny(
            low_threshold=self.canny_low_threshold,
            high_threshold=self.canny_high_threshold,
            blur_kernel=self.canny_blur_kernel,
            min_area=self.canny_min_area,
            max_area=self.canny_max_area
        )

        self.non_marker_contours, self.bitmap = self._isolate_non_aruco_contours(
            min_contour_area=self.min_contour_area
        )


    def _find_contours_brightness(self, threshold=100, blur_kernel=(5, 5),
                                  min_area=100, max_area=None):
        """
        Find contours in the image using binary thresholding.

        Args:
            threshold: Threshold value for binary conversion
            blur_kernel: Gaussian blur kernel size
            min_area: Minimum contour area to keep
            max_area: Maximum contour area to keep (None = no limit)
        """
        # Convert to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(gray_image, blur_kernel, 0)

        # Apply INVERTED threshold to create binary image
        # This makes dark objects (tools) white, and light background black
        _, self.binary_image = cv2.threshold(
            blurred_image, threshold, 255, cv2.THRESH_BINARY_INV
        )

        # Find contours
        contours_all, self.hierarchy = cv2.findContours(
            self.binary_image,
            cv2.RETR_TREE,  # Retrieves all contours and reconstructs hierarchy
            cv2.CHAIN_APPROX_SIMPLE  # Compresses horizontal, vertical, and diagonal segments
        )

        # Filter contours by area
        self.contours = []
        for contour in contours_all:
            area = cv2.contourArea(contour)
            if area >= min_area:
                if max_area is None or area <= max_area:
                    self.contours.append(contour)

        print(
            f"=== DEBUG: After INV threshold, found {len(self.contours)} contours ===")
        return self.contours

    def _find_contours_canny(self, low_threshold=50, high_threshold=150,
                             blur_kernel=(5, 5), min_area=100, max_area=None):
        """
        Find contours using Canny edge detection method.

        Args:
            low_threshold: Lower threshold for Canny edge detection
            high_threshold: Upper threshold for Canny edge detection
            blur_kernel: Gaussian blur kernel size
            min_area: Minimum contour area to keep
            max_area: Maximum contour area to keep (None = no limit)
        """
        # Convert to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(gray_image, blur_kernel, 0)

        # Apply Canny edge detection
        self.canny_edges = cv2.Canny(
            blurred_image, low_threshold, high_threshold)

        # Apply morphological closing to fill gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        self.canny_edges = cv2.dilate(self.canny_edges, kernel, iterations=1)

        # Find contours from Canny edges
        contours_all, hierarchy = cv2.findContours(
            self.canny_edges,
            cv2.RETR_EXTERNAL,  # Only external contours for Canny
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area
        canny_contours = []
        for contour in contours_all:
            area = cv2.contourArea(contour)
            if area >= min_area:
                if max_area is None or area <= max_area:
                    canny_contours.append(contour)

        return canny_contours

    def _isolate_non_aruco_contours(self, min_contour_area=100):
        """
        Unified method that ALWAYS uses perimeter detection.
        Works for both brightness and canny detection methods.
        """
        # Create perimeter mask
        perimeter_mask = self._create_marker_perimeter_mask()

        if perimeter_mask is None:
            print(
                "ERROR: Could not create perimeter mask - need at least 4 ArUco markers")
            return [], None

        # Get marker areas and bounding rectangles for comparison
        marker_corners = self.marker_corners
        contours = self.contours

        marker_areas = []
        marker_bounding_rects = []

        if marker_corners is not None:
            for corner_set in marker_corners:
                marker_area = cv2.contourArea(corner_set[0])
                marker_areas.append(marker_area)
                marker_rect = cv2.boundingRect(corner_set[0])
                marker_bounding_rects.append(marker_rect)

        # Filter contours by perimeter first
        perimeter_contours = self._filter_contours_by_perimeter(
            contours, perimeter_mask)
        print(
            f"Found {len(perimeter_contours)} contours inside perimeter (from {len(contours)} total)")

        # Then apply existing marker filtering
        non_marker_contours = []
        for contour in perimeter_contours:
            contour_area = cv2.contourArea(contour)

            # Skip very small contours
            if contour_area < min_contour_area:
                continue

            # For canny method, skip contours larger than 50% of perimeter area
            if self.detection_method == "canny":
                perimeter_area = cv2.countNonZero(perimeter_mask)
                

            # Check if this contour area matches any marker area
            is_marker_by_area = False
            for marker_area in marker_areas:
                if marker_area > 0 and abs(contour_area - marker_area) / marker_area < 0.1:
                    is_marker_by_area = True
                    break

            # Check if contour is inside any marker bounding rectangle
            is_inside_marker_rect = False
            if marker_bounding_rects:
                is_inside_marker_rect = self._is_contour_inside_marker_rects(
                    contour, marker_bounding_rects)

            # Only include if it's not a marker
            if not is_marker_by_area and not is_inside_marker_rect:
                non_marker_contours.append(contour)

        # Select only the largest contour
        if non_marker_contours:
            largest_contour = max(non_marker_contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            non_marker_contours = [largest_contour]
            print(
                f"Selected largest contour inside perimeter with area {largest_area:.1f}")
        else:
            print("No non-marker contours found inside perimeter")

        # Create bitmap
        height, width = self.image.shape[:2]
        bitmap = np.full((height, width), 255, dtype=np.uint8)

        if non_marker_contours:
            cv2.drawContours(bitmap, non_marker_contours, -1, 0, thickness=-1)
            bitmap = cv2.bitwise_not(bitmap)

        return non_marker_contours, bitmap

    def _create_marker_perimeter_mask(self):
        """
        Create a mask that defines the area inside the ArUco marker perimeter.
        Returns a binary mask where white pixels are inside the perimeter.
        """
        if self.marker_corners is None or len(self.marker_corners) < 4:
            print("Need at least 4 markers to create perimeter")
            error_msg = "Insufficient markers detected to create detection perimeter."
            messagebox.showerror("Marker Detection Error", error_msg)
            return None

        height, width = self.image.shape[:2]

        # Extract marker center points
        marker_centers = []
        for corner_set in self.marker_corners:
            corners = corner_set[0]  # Get the 4 corner points
            # Calculate center of each marker
            center_x = np.mean(corners[:, 0])
            center_y = np.mean(corners[:, 1])
            marker_centers.append([center_x, center_y])

        marker_centers = np.array(marker_centers, dtype=np.float32)

        # Sort markers to form a proper polygon (clockwise or counter-clockwise)
        # Find the convex hull to get the proper order
        hull_indices = cv2.convexHull(marker_centers, returnPoints=False)
        hull_points = marker_centers[hull_indices.flatten()]

        # Create mask from the hull polygon
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [hull_points.astype(np.int32)], 255)

        return mask

    def _filter_contours_by_perimeter(self, contours, perimeter_mask):
        """
        Filter contours to only include those that are completely inside the perimeter mask.

        Args:
            contours: List of contours to filter
            perimeter_mask: Binary mask defining the valid area

        Returns:
            List of contours that are inside the perimeter
        """
        if perimeter_mask is None:
            return contours

        filtered_contours = []

        for contour in contours:
            # Check if contour is inside perimeter
            if self._is_contour_inside_perimeter(contour, perimeter_mask):
                filtered_contours.append(contour)

        return filtered_contours

    def _is_contour_inside_perimeter(self, contour, perimeter_mask):
        """
        Check if a contour is completely inside the perimeter mask.

        Args:
            contour: The contour to check
            perimeter_mask: Binary mask defining the valid area

        Returns:
            True if contour is inside perimeter, False otherwise
        """
        # Method 1: Check if all contour points are inside the mask
        all_points_inside = True
        for point in contour:
            x, y = point[0]
            if x < 0 or y < 0 or x >= perimeter_mask.shape[1] or y >= perimeter_mask.shape[0]:
                all_points_inside = False
                break
            if perimeter_mask[y, x] == 0:  # Point is outside perimeter
                all_points_inside = False
                break

        if not all_points_inside:
            return False

        # Method 2: Additional check - verify contour centroid is inside
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if (0 <= cx < perimeter_mask.shape[1] and
                0 <= cy < perimeter_mask.shape[0] and
                    perimeter_mask[cy, cx] > 0):
                return True

        return False

    def _is_contour_inside_marker_rects(self, contour, marker_bounding_rects):
        """
        Check if a contour is inside any of the marker bounding rectangles

        Args:
            contour: The contour to check
            marker_bounding_rects: List of marker bounding rectangles (x, y, w, h)

        Returns:
            True if contour is inside any marker rectangle, False otherwise
        """
        # Get contour's bounding rectangle
        contour_rect = cv2.boundingRect(contour)
        cx, cy, cw, ch = contour_rect

        # Check against each marker bounding rectangle
        for marker_rect in marker_bounding_rects:
            mx, my, mw, mh = marker_rect

            # Check if contour bounding rectangle is completely inside marker rectangle
            if (cx >= mx and cy >= my and
                    cx + cw <= mx + mw and cy + ch <= my + mh):
                return True

        return False

    def _detect_aruco_markers(self, dictionary_type=cv2.aruco.DICT_6X6_250):
        """
        Detect ArUco markers in the image.

        The library being used is the 6x6_250 dictionary.

        Args:
            dictionary_type: Type of ArUco dictionary to use
        """
        arucoDict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, parameters)

        self.marker_corners, self.marker_ids, rejectedImgPoints = detector.detectMarkers(
            self.image)

        if self.marker_ids is not None:
            print(f"Detected {len(self.marker_ids)} ArUco markers.")
            self.count_markers = len(self.marker_ids)
        else:
            self.count_markers = 0
            print("No ArUco markers detected.")
            error_msg = "No ArUco markers detected in the image.\n\nPlease ensure markers are visible and unobstructed."
            messagebox.showerror("Marker Detection Error", error_msg)

        return self.marker_corners, self.marker_ids, rejectedImgPoints

    def _calculate_scale(self):
        """
        Calculate the scale of the image based on the ArUco markers detected.
        """
        if self.marker_ids is None:
            print("No markers detected for scale calculation")
            return None

        marker_areas = []

        # Calculate area for each detected marker
        for i, corner_set in enumerate(self.marker_corners):
            # corner_set is shape (1, 4, 2) - reshape to (4, 2)
            marker_corners = corner_set[0]  # Get the 4 corner points

            # Using cv2.contourArea (recommended)
            area = cv2.contourArea(marker_corners)
            marker_areas.append(area)

            print(
                f"Marker ID {self.marker_ids[i][0]}: Area = {area:.2f} pixels²")

        # Calculate the average area of the markers
        if marker_areas:
            average_area = np.mean(marker_areas)
            print(f"Average marker area: {average_area:.2f} pixels²")

        # Take the square root of the average area to get the scale
        if average_area > 0:
            size = np.sqrt(average_area)
            print(f"ArUco marker size: {size:.2f} pixels")

        # Compute the scale factor based on the fact that the markers are 1 inch by 1 inch
        pixels_per_inch = size / 1.0
        print(f"Scale factor: {pixels_per_inch:.2f} pixels/inch")

        return pixels_per_inch

    def update_tolerance_processing(self):
        """Update tolerance contours and bitmap based on current settings"""
        if not self.non_marker_contours or self.pixels_per_inch is None:
            self.tolerance_contours = []
            self.tolerance_bitmap = None
            return

        # Apply tolerance to the base contour
        base_contour = self.non_marker_contours[0]  # We only keep the largest

        if self.current_tolerance_mm <= 0:
            # No tolerance - use original contours
            self.tolerance_contours = self.non_marker_contours.copy()
            self.tolerance_bitmap = self.bitmap.copy() if self.bitmap is not None else None
        else:
            # Apply tolerance
            tolerance_contour = self.apply_tolerance_to_contour(
                base_contour, self.current_tolerance_mm)
            self.tolerance_contours = [tolerance_contour]
            self.tolerance_bitmap = self.create_tolerance_bitmap(
                tolerance_contour)

    def apply_tolerance_to_contour(self, contour, tolerance_mm):
        """Apply tolerance by dilating the contour"""
        if tolerance_mm <= 0:
            return contour

        # Create mask from contour
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)

        # Convert tolerance to pixels and apply dilation
        tolerance_pixels = int(tolerance_mm * self.pixels_per_inch / 25.4)
        if tolerance_pixels > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (tolerance_pixels*2+1, tolerance_pixels*2+1))
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)

            # Find new contour from dilated mask
            contours, _ = cv2.findContours(
                dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                return max(contours, key=cv2.contourArea)

        return contour

    def create_tolerance_bitmap(self, tolerance_contour):
        """Create bitmap from tolerance contour"""
        height, width = self.image.shape[:2]
        bitmap = np.full((height, width), 255, dtype=np.uint8)

        if tolerance_contour is not None:
            cv2.drawContours(bitmap, [tolerance_contour], -1, 0, thickness=-1)
            bitmap = cv2.bitwise_not(bitmap)

        return bitmap
    
    def calculate_tool_dimensions(self):

        contour = self.non_marker_contours[0]
        rect = cv2.minAreaRect(contour)
        center, size, angle = rect

        # Calculate tool dimensions in inches
        width_inch = round(size[0] / self.pixels_per_inch, 2)
        height_inch = round(size[1] / self.pixels_per_inch, 2)

        # calculate tool dimensions in mm
        width_mm = round(width_inch * 25.4, 2)
        height_mm = round(height_inch * 25.4, 2)

        # tool dimensions in pixels  

        width_px = round(size[0], 2)
        height_px = round(size[1], 2)

        self.dimensions_inches = f"{width_inch} in x {height_inch} in"
        self.dimensions_mm = f"{width_mm} mm x {height_mm} mm"
        self.dimensions_pixels = f"{width_px} px x {height_px} px"

        print(self.dimensions_inches)
        print(self.dimensions_mm)
        print(self.dimensions_pixels)
