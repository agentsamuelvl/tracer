# undistort.py


import cv2
import numpy as np
import time


class Undistort:

    def __init__(self, dist_coeffs, camera_matrix, reference_width=3840, reference_height=2160):
        """
        Undistortion class for correcting lens distortion in video feeds and images.

        Args:
            dist_coeffs: Numpy array of distortion coefficients
            camera_matrix: Numpy array of the camera matrix obtained at reference resolution
            reference_width: Width of the reference resolution (default: 3840)
            reference_height: Height of the reference resolution (default: 2160)
        """
        self.dist_coeffs = dist_coeffs
        self.camera_matrix = camera_matrix
        self.reference_width = reference_width
        self.reference_height = reference_height

        # Cache for scaled matrices
        self._cached_matrices = {}

    def get_scaled_camera_matrix(self, target_width, target_height):
        """
        Get camera matrix scaled for target resolution.

        Args:
            target_width: Target image/video width
            target_height: Target image/video height

        Returns:
            Scaled camera matrix for the target resolution
        """
        # Use caching to avoid recalculating same matrices
        cache_key = (target_width, target_height)
        if cache_key in self._cached_matrices:
            return self._cached_matrices[cache_key]

        # Calculate scaling factors
        scale_x = target_width / self.reference_width
        scale_y = target_height / self.reference_height

        # If no scaling needed, return original matrix
        if scale_x == 1.0 and scale_y == 1.0:
            scaled_matrix = self.camera_matrix.copy()
        else:
            # Create scaling matrix
            scaling_matrix = np.array([
                [scale_x, 0, 0],
                [0, scale_y, 0],
                [0, 0, 1]
            ], dtype=np.float64)

            scaled_matrix = scaling_matrix @ self.camera_matrix

        # Cache the result
        self._cached_matrices[cache_key] = scaled_matrix
        return scaled_matrix

    def correct_frame(self, frame, frame_resolution=None):
        """
        Correct lens distortion for a frame, automatically detecting or using specified resolution.

        Args:
            frame: Input frame (numpy array)
            frame_resolution: Optional tuple (width, height). If None, detects from frame.

        Returns:
            Undistorted frame
        """
        if frame_resolution is None:
            # Auto-detect frame resolution
            frame_height, frame_width = frame.shape[:2]
        else:
            frame_width, frame_height = frame_resolution

        # Get appropriate camera matrix
        camera_matrix = self.get_scaled_camera_matrix(
            frame_width, frame_height)

        # Apply undistortion
        corrected_frame = cv2.undistort(frame, camera_matrix, self.dist_coeffs)
        return corrected_frame

    def correct_image_distortion(self, image):
        """
        Correct lens distortion for high-resolution images.

        Args:
            image: Input image (numpy array)

        Returns:
            Undistorted image
        """
        return self.correct_frame(image)

    def correct_video_distortion(self, frame, target_resolution=None):
        """
        Correct lens distortion for video frames.

        Args:
            frame: Input video frame
            target_resolution: Optional (width, height) tuple for specific resolution

        Returns:
            Undistorted frame
        """
        return self.correct_frame(frame, target_resolution)

    def get_cache_info(self):
        """Get information about cached matrices."""
        return {
            'cached_resolutions': list(self._cached_matrices.keys()),
            'cache_size': len(self._cached_matrices)
        }


def main():
    """Test the undistortion with live camera feed."""

    # Your calibration data
    dist_coeffs = np.array([
        -3.098348566381915,
        1.9305800045978267,
        -0.0008770414354296032,
        0.00038636963423301795,
        0.9230471476181449,
        -2.68191413745652,
        0.5139948439427104,
        2.1620483642150687,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ], dtype=np.float64)

    camera_matrix = np.array([
        [2639.568422234177, 0.0, 1924.635156665537],
        [0.0, 2643.409206875026, 1112.4588562588956],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Initialize undistorter
    undistorter = Undistort(dist_coeffs, camera_matrix)

    # Setup camera
    cap = cv2.VideoCapture(1)

    # Try to set high resolution (camera will use what it supports)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    # Check actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")

    # Performance monitoring
    frame_count = 0
    start_time = time.time()

    print("Press 'q' to quit, 'c' to capture image, 'i' for cache info")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Apply undistortion (automatically detects frame resolution)
        corrected_frame = undistorter.correct_frame(frame)

        # Performance tracking
        frame_count += 1
        if frame_count % 30 == 0:  # Every 30 frames
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"Processing at {fps:.1f} FPS")

        # Display results
        # Resize for display if too large
        if frame.shape[1] > 1920:  # If wider than 1920px
            display_scale = 1920 / frame.shape[1]
            new_width = int(frame.shape[1] * display_scale)
            new_height = int(frame.shape[0] * display_scale)

            frame_display = cv2.resize(frame, (new_width, new_height))
            corrected_display = cv2.resize(
                corrected_frame, (new_width, new_height))
        else:
            frame_display = frame
            corrected_display = corrected_frame

        cv2.imshow('Original', frame_display)
        cv2.imshow('Corrected', corrected_display)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Capture high-res image
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"undistorted_capture_{timestamp}.jpg"
            cv2.imwrite(filename, corrected_frame, [
                        cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Saved: {filename}")
        elif key == ord('i'):
            # Show cache info
            cache_info = undistorter.get_cache_info()
            print(f"Cache info: {cache_info}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Final performance report
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    print(f"\nFinal stats:")
    print(f"Processed {frame_count} frames in {elapsed:.1f} seconds")
    print(f"Average FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    main()
