# models/camera_manager.py

import cv2
import numpy as np
import gc
import time
import threading
from threading import Thread
from .undistort import Undistort
from utils.message_bus import MessageBus, Message
from utils.config_manager import ConfigManager
from tkinter import messagebox


class CameraManager:
    def __init__(self, controller):
        self.controller = controller
        self.bus = controller.bus
        
        # Subscribe to messages
        self.bus.subscribe('controller.capture_request', self.capture_frame)
        self.bus.subscribe('controller.camera_parameters_updated', self.reload_camera_parameters)
        self.bus.subscribe('controller.start_camera', self._initialize_camera)
        self.bus.subscribe('controller.stop_camera', self.stop_camera)
        self.bus.subscribe('controller.camera_index_changed', self.on_camera_index_change)

        # Camera state
        self.cap = None
        self.running = False
        self.processing_thread = None
        self.frame_lock = threading.Lock()
        
        # Frame storage
        self.latest_frame_data = None
        self.latest_full_frame = None

        # Load camera parameters

        self.camera_index = 1  # Default camera index
        self.config_manager = ConfigManager()
        self.load_camera_parameters()
        
        # Initialize undistorter (but don't cache maps yet)
        ref_width, ref_height = self.config_manager.get_reference_resolution()
        self.undistorter = Undistort(self.dist_coeffs, self.camera_matrix, ref_width, ref_height)
        self.undistortion_maps = None

        self.black_frame_threshold = 10  # Average brightness below this = black frame
        self.consecutive_black_frames = 0
        self.max_consecutive_black_frames = 4  # Trigger error after this many
        self.black_frame_error_shown = False
        self.initial_frames_processed = 0
        self.min_initial_frames = 5  # Check after processing at least this many frames


    def on_camera_index_change(self, message: Message):
        """Handle camera index change"""
        try:
            new_index = message.data['new_value']
            self.camera_index = new_index
            print(f"Camera index updated to: {new_index}")
            
            # Note: The actual camera change will happen when start_camera is called next
            
        except Exception as e:
            print(f"Error updating camera index: {e}")


    def load_camera_parameters(self):
        """Load camera parameters from configuration"""
        try:
            self.dist_coeffs = self.config_manager.get_distortion_coefficients()
            self.camera_matrix = self.config_manager.get_camera_matrix()
            
            config_summary = self.config_manager.get_config_summary()
            print(f"Loaded camera parameters: {config_summary['camera_model']}")
            print(f"Focal length: {config_summary['focal_length']}")
            print(f"Principal point: {config_summary['principal_point']}")
            
        except Exception as e:
            print(f"Error loading camera parameters: {e}")
            self._load_fallback_parameters()

    def _load_fallback_parameters(self):
        """Fallback camera parameters"""
        print("Using fallback camera parameters")
        
        self.dist_coeffs = np.array([
            -0.1, 0.05, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ], dtype=np.float64)

        self.camera_matrix = np.array([
            [800.0, 0.0, 960.0],
            [0.0, 800.0, 540.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def _initialize_camera(self, message: Message):
        """Initialize camera and start frame processing"""
        
        self.consecutive_black_frames = 0
        self.black_frame_error_shown = False
        self.initial_frames_processed = 0

        if not self.running:
            print("Initializing camera...")
            self.running = True

            # Open camera
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.running = False
                messagebox.showerror("Camera Error", "Failed to open camera. Please make sure a camera is connected and the proper index is selected.")
                raise RuntimeError(f"Camera at index {self.camera_index} failed to open")
            
            else:
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3480)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                self.cap.set(cv2.CAP_PROP_FPS, 10)  
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Get actual resolution and update config
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Camera initialized at: {actual_width}x{actual_height}")
                
                self.config_manager.update_reference_resolution(actual_width, actual_height)
                
                # Cache undistortion maps for this resolution
                self._cache_undistortion_maps()

                # Start frame processing thread
                self.processing_thread = Thread(target=self._process_frames_loop, daemon=False) 
                self.processing_thread.start()
                
                # Give camera a moment to stabilize
                time.sleep(1.0)
                
                # CHANGE THIS: Only publish success if camera is still running
                if self.running:  # ADD THIS CHECK
                    self.bus.publish('model.camera_initialized', {}, 'CameraManager')
                    print("Camera initialization complete")
                else:
                    print("Camera initialization failed - camera stopped during startup")

        else:
            print("Camera is already running")

    def _cache_undistortion_maps(self):
        """Pre-calculate undistortion maps"""
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        try:
            scaled_camera_matrix = self.undistorter.get_scaled_camera_matrix(w, h)
            self.undistortion_maps = cv2.initUndistortRectifyMap(
                scaled_camera_matrix, self.dist_coeffs,
                None, scaled_camera_matrix, (w, h), cv2.CV_32FC1
            )
            print(f"Undistortion maps cached for {w}x{h}")
        except Exception as e:
            print(f"Error caching undistortion maps: {e}")
            self.undistortion_maps = None

    def _process_frames_loop(self):
        """Main frame processing loop with black frame and disconnection detection"""
        print("Starting frame processing...")
        frame_count = 0
        consecutive_failed_reads = 0
        max_failed_reads = 1  # Camera considered disconnected after this many failures
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                # Check for camera disconnection first
                if not ret or frame is None:
                    consecutive_failed_reads += 1
                    print(f"Failed to read frame ({consecutive_failed_reads} consecutive failures)")
                    
                    if consecutive_failed_reads >= max_failed_reads:
                        self._handle_camera_disconnection()
                        break  # Exit the loop immediately
                    
                    time.sleep(0.1)
                    continue
                
                # Reset failed read counter on successful read
                consecutive_failed_reads = 0
                
                frame_count += 1
                self.initial_frames_processed += 1
                
                # Check for black frames (only if camera is connected and reading frames)
                if self._is_black_frame(frame):
                    self.consecutive_black_frames += 1
                    print(f"Black frame detected ({self.consecutive_black_frames} consecutive)")
                    
                    # Only check for errors after initial frames are processed
                    if (self.initial_frames_processed >= self.min_initial_frames and 
                        self.consecutive_black_frames >= self.max_consecutive_black_frames and 
                        not self.black_frame_error_shown):
                        
                        self._handle_black_frame_error()
                        break  # Exit after black frame error
                        
                else:
                    # Reset counter on valid frame
                    if self.consecutive_black_frames > 0:
                        print(f"Valid frame received, resetting black frame counter")
                    self.consecutive_black_frames = 0
                
                # Process frame normally
                undistorted = self.undistort_frame(frame)
                cropped = self.crop_frame(undistorted)
                resized = self.resize_frame(cropped)

                # Store processed frame
                frame_data = {
                    'display_frame': resized.copy(),
                    'full_frame': cropped.copy(),
                    'timestamp': time.time(),
                    'is_black_frame': self.consecutive_black_frames > 0
                }

                with self.frame_lock:
                    self.latest_frame_data = frame_data
                    self.latest_full_frame = cropped.copy()

                # Cleanup
                del frame, undistorted, cropped, resized
                
                # Periodic status and cleanup
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames")
                    gc.collect()
                
                time.sleep(0.04)  # ~25 FPS
                
            except Exception as e:
                print(f"Frame processing error: {e}")
                consecutive_failed_reads += 1
                
                if consecutive_failed_reads >= max_failed_reads:
                    self._handle_camera_disconnection()
                    break
                
                time.sleep(0.1)
        
        print("Frame processing stopped")
        
        # Clean up resources when loop exits
        if self.cap:
            self.cap.release()
            self.cap = None
        
        with self.frame_lock:
            self.latest_frame_data = None
            self.latest_full_frame = None
        
        # Only publish camera_stopped if we haven't already published an error
        if self.running:  # If running is still True, it was a normal stop
            self.bus.publish('model.camera_stopped', {}, 'CameraManager')

    def _handle_camera_disconnection(self):
        """Handle camera disconnection - different from black frames"""
        print("Camera disconnection detected")
        
        error_msg = (
            "Camera has been disconnected or is no longer accessible. This could indicate:\n"
            "• Camera was unplugged\n"
            "• USB connection was lost\n"
            "• Camera is being used by another application\n"
            "• Camera hardware failure"
        )
        
        # Stop processing
        self.running = False
        
        # Publish disconnection error (different message type)
        self.bus.publish('model.camera_disconnection_error', {
            'error': error_msg,
            'error_type': 'disconnection'
        }, 'CameraManager')


    def _is_black_frame(self, frame):
        """Check if frame is predominantly black"""
        try:
            # Convert to grayscale for brightness calculation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate average brightness
            avg_brightness = np.mean(gray)
            
            # Also check if frame is too uniform (all pixels similar value)
            std_dev = np.std(gray)
            is_uniform = std_dev < 5  # Very low standard deviation = uniform image
            
            is_black = avg_brightness < self.black_frame_threshold
            
            if is_black or is_uniform:
                print(f"Frame analysis: avg_brightness={avg_brightness:.1f}, std_dev={std_dev:.1f}")
                return True
                
            return False
            
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return False

    def _handle_black_frame_error(self):
        """Handle black frame error condition"""
        
        self.black_frame_error_shown = True
        error_msg = (
            "Camera is producing black or invalid frames. This could indicate:\n"
            "• Camera lens is covered or obstructed\n"
            "• Camera is not receiving enough light\n"
            "• Camera hardware malfunction\n"
            "• USB connection issues\n"
            "Try unplugging the camera, plugging it back in, and starting the camera again."
        )
        
        print(f"Black frame error: {error_msg}")
        
        # ADD THIS: Stop the processing loop immediately
        self.running = False
        
        # Publish error through message bus
        self.bus.publish('model.camera_black_frame_error', {
            'error': error_msg,
            'error_type': 'black_frames',
            'consecutive_count': self.consecutive_black_frames
        }, 'CameraManager')
        

    def stop_camera(self, message = None):
        """Stop camera and cleanup"""

        if self.running:
            self.running = False
            print("Stopping camera...")
            self.release()
            self.bus.publish('model.camera_stopped', {}, 'CameraManager')
            print("Camera stopped")

        else:
            print("Camera is not running")

    def undistort_frame(self, frame):
        """Apply undistortion"""
        if self.undistortion_maps is not None:
            map1, map2 = self.undistortion_maps
            return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        return self.undistorter.correct_frame(frame)

    def crop_frame(self, frame):
        """Crop to 14:10 aspect ratio"""
        height, width = frame.shape[:2]
        target_aspect = 14 / 10
        crop_height = height
        crop_width = int(crop_height * target_aspect)

        if crop_width > width:
            crop_width = width
            crop_height = int(width / target_aspect)

        start_x = (width - crop_width) // 2
        start_y = (height - crop_height) // 2

        return frame[start_y:start_y + crop_height, start_x:start_x + crop_width]

    def resize_frame(self, frame):
        """Resize for display"""
        return cv2.resize(frame, (800, 571))

    def capture_frame(self, message: Message):
        """Capture current full-size frame"""
        with self.frame_lock:
            if self.latest_full_frame is not None:
                frame_copy = self.latest_full_frame.copy()
                self.bus.publish('model.full_frame_captured', frame_copy, 'CameraManager')
                return frame_copy
            else:
                print("No frame available for capture")
                return None

    def get_latest_frame_data(self):
        """Get latest frame data for GUI"""
        try:
            with self.frame_lock:
                if self.latest_frame_data is not None:
                    return self.latest_frame_data.copy()
                return None
        except Exception as e:
            print(f"Error getting frame data: {e}")
            return None

    def reload_camera_parameters(self, message=None):
        """Reload camera parameters"""
        print("Reloading camera parameters...")
        
        if message and message.data:
            self._apply_parameter_change(message.data)

        self.load_camera_parameters()
        self.undistorter = Undistort(self.dist_coeffs, self.camera_matrix)
        
        if self.cap is not None:
            self._cache_undistortion_maps()

    def _apply_parameter_change(self, parameter_data):
        """Apply parameter change"""
        param_type = parameter_data.get('parameter_type')
        param_name = parameter_data.get('parameter_name')
        new_value = parameter_data.get('new_value')

        if param_type == 'camera_matrix':
            kwargs = {param_name: new_value}
            self.config_manager.update_camera_matrix(**kwargs)
        elif param_type == 'distortion_coefficients':
            kwargs = {param_name: new_value}
            self.config_manager.update_distortion_coefficients(**kwargs)

    def release(self):
        """Clean shutdown"""
        print("Shutting down camera...")
        
        # Stop processing thread
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3.0)
        
        # Clear frame data
        with self.frame_lock:
            self.latest_frame_data = None
            self.latest_full_frame = None
        
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Cleanup
        cv2.destroyAllWindows()
        self.undistortion_maps = None
        
        print("Camera shutdown complete")