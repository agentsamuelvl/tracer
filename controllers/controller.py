# controllers/controller.py


from models.camera_manager import CameraManager
from models.detect_contours import DetectContours
from models.resize_image import ResizeBitMap
from models.bitmap_to_svg import BitmapToSVG
from views.main_gui import View as MainView
from utils.message_bus import MessageBus, Message
import cv2
from tkinter import messagebox
import numpy as np
import threading
import queue
import time


class Controller:

    def __init__(self, parent, message_bus: MessageBus):
        self.bus = message_bus
        self.camera_manager = CameraManager(self)
        self.detect_contours = DetectContours(self)

        self.root = parent

        self.bus.subscribe('model.full_frame_captured',
                           self.on_full_frame_captured)
        self.bus.subscribe('view.user_clicked_capture',
                           self.on_user_clicked_capture)
        self.bus.subscribe('view.user_clicked_process',
                           self.on_user_clicked_process)
        self.bus.subscribe('view.user_clicked_reset',
                           self.on_user_clicked_reset)
        self.bus.subscribe('model.contour_data', self.receive_data)
        self.bus.subscribe('view.brightness_threshold_changed',
                           self.on_brightness_change)
        self.bus.subscribe('view.canny_low_threshold_changed',
                           self.on_canny_low_change)
        self.bus.subscribe('view.canny_high_threshold_changed',
                           self.on_canny_high_change)
        self.bus.subscribe('view.detection_method_changed',
                           self.on_detection_method_change)
        self.bus.subscribe('view.tolerance_changed', self.on_tolerance_change)
        self.bus.subscribe('view.camera_parameter_changed',
                           self.on_camera_parameter_change)
        self.bus.subscribe('view.export_pressed', self.on_export_pressed)
        self.bus.subscribe('view.user_clicked_start_camera', self.on_user_clicked_start_camera)
        self.bus.subscribe('view.user_clicked_stop_camera', self.on_user_clicked_stop_camera)
        self.bus.subscribe('model.camera_initialized', self.on_camera_initialized)
        self.bus.subscribe('model.camera_stopped', self.on_camera_stopped)
        self.bus.subscribe('model.camera_black_frame_error', self.on_black_frame_error)
        self.bus.subscribe('model.camera_disconnection_error', self.on_camera_disconnection_error)
        self.bus.subscribe('model.status_message', self.on_status_message)
        self.bus.subscribe('view.camera_index_changed', self.on_camera_index_change)

    # Bitmap and tolerance settings
        self.tolerance_contours = None
        self.tolerance_bitmap = None
        self.current_tolerance_mm = None
        self.pixels_per_inch = None
        self.bitmap = None
        self.non_marker_contours = None
        self.count_markers = 0

    # Camera intialization settings
        self.interval = 38
        self.display_on = True
        self.image = None
        self.image_captured = False
        self.camera_initialized = False

    # Processing intialization settings
        self.detection_method = "Brightness"
        self.threshold = 128
        self.canny_low_threshold = 30
        self.canny_high_threshold = 90

        print("Camera controller initialized")

    # export initialization settings

        self.dpi = 96


# ------------------- Camera related methods ---------------------


    def on_user_clicked_capture(self, message: Message):

        if not self.camera_initialized:
            print("Camera is not initialized.")
            messagebox.showerror("Error", "Camera is not initialized.")
            return
        else:

            self._hide_video_feed()
            self.image_captured = True

            # send a request to the model to capture the latest frame
            self.bus.publish('controller.capture_request', {}, "Controller")

    def on_user_clicked_reset(self, message: Message):
        """Handle reset button click"""
        self._show_video_feed()
        self.image = None
        self.bitmap = None
        self.image_captured = False

        # Send reset to all models
        self.bus.publish('controller.reset_request', {}, "Controller")

    def on_user_clicked_start_camera(self, message: Message):
        if not self.camera_initialized:
            print("Requesting camera start...")
            
            # Reset any error states
            self.display_on = False  # Will be set to True when camera initializes successfully
            
            self.bus.publish('controller.start_camera', {}, 'Controller')
            
        else:
            print("Camera is already running")

    def on_user_clicked_stop_camera(self, message: Message):
        if self.camera_initialized:
            print("Requesting camera stop...")
            self.bus.publish('controller.stop_camera', {}, 'Controller')
            self.camera_initialized = False
            self.display_on = False
        else:
            print("Camera not running")

    def on_camera_initialized(self, message: Message):
        # Publishes to the view to start updating the camera display
        self.display_on = True
        self.camera_initialized = True
        self.bus.publish('controller.camera_initialized', {}, 'Controller')

    def on_camera_stopped(self, message: Message):

        # Publishes to the view to stop updating the camera display
        self.bus.publish('controller.camera_stopped', {}, 'Controller')
    
    def on_camera_index_change(self, message: Message):
        """Handle camera index changes from UI"""
        print(f"Controller received camera index change: {message.data}")
        
        # Forward to camera manager 
        self.bus.publish('controller.camera_index_changed', message.data, 'Controller')

    def on_camera_disconnection_error(self, message: Message):
        """Handle camera disconnection errors"""
        error_data = message.data
        error_msg = error_data.get('error', 'Camera was disconnected')
        
        print("Camera disconnection detected in controller")
        
        # Reset camera state
        self.camera_initialized = False
        self.display_on = False
        
        # Show disconnection-specific error message
        try:
            messagebox.showerror("Camera Disconnected", error_msg)
        except Exception as e:
            print(f"Could not show disconnection error dialog: {e}")
        
        # Notify view about the disconnection
        self.bus.publish('controller.camera_error', {
            'error': 'Camera was disconnected',
            'error_type': 'disconnection'
        }, 'Controller')

    def on_black_frame_error(self, message: Message):
        """Handle black frame errors from camera manager"""
        error_data = message.data
        error_msg = error_data.get('error', 'Camera producing black frames')
        consecutive_count = error_data.get('consecutive_count', 0)
        
        print(f"Black frame error: {consecutive_count} consecutive black frames")
        
        # Reset camera state (camera will already be stopped by the time this runs)
        self.camera_initialized = False
        self.display_on = False
        
        # Show error message 
        messagebox.showerror("Black Frame Error", f"{error_msg}")

        # Notify view about the black frame issue (for UI updates)
        self.bus.publish('controller.camera_error', {
            'error': 'Camera stopped due to black frames',
            'error_type': 'black_frames'
        }, 'Controller')
        
        # Note: No need to call stop_camera here since the camera manager 
        # already stopped the processing thread and cleaned up resources
        

    def get_latest_frame(self):
        """Get the latest frame data - updated for new thread-safe interface"""
        try:
            
            
            if not self.display_on:
                
                return None
            
            if not hasattr(self, 'camera_manager') or self.camera_manager is None:
                print("DEBUG CTRL: No camera manager available")
                return None
                
            frame_data = self.camera_manager.get_latest_frame_data()
            
            if frame_data and 'display_frame' in frame_data:
                
                return frame_data['display_frame']
            else:
                print("DEBUG CTRL: No frame data from camera manager")
                return None
            
        except Exception as e:
            
            return None

    def on_full_frame_captured(self, message: Message):

        # receive a full resolution frame from the model

        self.image = message.data  # this image can be used for later processing

        self.bus.publish('controller.full_frame_captured',
                         self.image, "Controller")

        self.get_display_image()

    def on_camera_parameter_change(self, message: Message):
        """Handle camera parameter changes from UI"""
        print(f"Controller received camera parameter change: {message.data}")

        # Forward to camera manager (following your existing pattern)
        self.bus.publish('controller.camera_parameters_updated',
                         message.data, 'Controller')

    def _prep_image_for_display(self):
        """Prepare the image for display (e.g., resizing, normalization)"""
        if self.image is not None:
            # Perform any necessary preprocessing on self.image
            display_image = self.image.copy()
            display_image = cv2.resize(display_image, (800, 571))

            display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

            return display_image_rgb

    def get_display_image(self):

        display_image = self._prep_image_for_display()
        self.bus.publish('controller.display_image',
                         display_image, 'Controller')

    def _show_video_feed(self):
        self.display_on = True
        self.bus.publish('controller.restart_display', {}, 'Controller')

    def _hide_video_feed(self):
        self.display_on = False

    def _take_picture(self):
        """Capture image and switch to display mode"""
        self._hide_video_feed()
    

    def reset(self, message: Message):
        """Reset to live camera feed"""
        print("Reset called")
        self.image = None
        self.image_captured = False
        self._show_video_feed()

        self.bus.publish('controller.restart_display', {}, 'Controller')
        print("Reset completed")

    def cleanup(self):
        """Enhanced cleanup method with better error handling"""
        print("Starting controller cleanup...")
        
        try:
            # 1. Stop display first
            self.display_on = False
            print("✓ Display stopped")
        except Exception as e:
            print(f"Error stopping display: {e}")
        
        try:
            # 2. Clean up camera manager with timeout
            if hasattr(self, 'camera_manager') and self.camera_manager:
                print("Releasing camera manager...")
                self.camera_manager.release()
                print("✓ Camera manager released")
        except Exception as e:
            print(f"Error releasing camera: {e}")
        
        try:
            # 3. Clean up message bus subscriptions
            if hasattr(self, 'bus') and self.bus:
                # Unsubscribe from all topics
                topics_to_unsubscribe = [
                    'model.full_frame_captured',
                    'view.user_clicked_capture',
                    'view.user_clicked_process',
                    'view.user_clicked_reset',
                    'model.contour_data',
                    'view.brightness_threshold_changed',
                    'view.detection_method_changed',
                    'view.tolerance_changed',
                    'view.camera_parameter_changed',
                    'view.export_pressed'
                ]
                
                for topic in topics_to_unsubscribe:
                    try:
                        # This assumes your MessageBus has an unsubscribe method
                        # You may need to implement this if it doesn't exist
                        if hasattr(self.bus, '_subscribers') and topic in self.bus._subscribers:
                            self.bus._subscribers[topic].clear()
                    except Exception as e:
                        print(f"Error unsubscribing from {topic}: {e}")
                
                print("✓ Message bus cleaned up")
        except Exception as e:
            print(f"Error cleaning up message bus: {e}")
        
        try:
            # 4. Clear all image data and references
            self.image = None
            self.bitmap = None
            self.tolerance_bitmap = None
            self.non_marker_contours = None
            self.tolerance_contours = None
            
            # Clear detect_contours if it exists
            if hasattr(self, 'detect_contours'):
                self.detect_contours = None
            
            print("✓ Image data cleared")
        except Exception as e:
            print(f"Error clearing image data: {e}")
        
        print("Controller cleanup completed")


# -------------- Processing related methods ---------------------

    def on_user_clicked_process(self, message: Message):

        self.bus.publish('controller.process_request',
                         self.image, 'Controller')

    def on_brightness_change(self, message: Message):

        print(f"controller received brightness threshold {message.data}")
        self.bus.publish('controller.brightness_changed',
                         message.data, 'Controller')

    def on_canny_low_change(self, message: Message):
        
        print(f"controller received canny low threshold {message.data}")
        self.bus.publish('controller.canny_low_threshold_changed',
                         message.data, 'Controller')

    def on_canny_high_change(self, message: Message):
        
        print(f"controller received canny high threshold {message.data}")
        self.bus.publish('controller.canny_high_threshold_changed',
                         message.data, 'Controller')
    

    

    def on_tolerance_change(self, message: Message):

        self.current_tolerance_mm = message.data
        print(f"controller received tolerance {message.data}")
        self.bus.publish('controller.tolerance_changed',
                         message.data, 'Controller')

    def on_detection_method_change(self, message: Message):

        print(f"changing detection method to {message.data}")
        self.bus.publish('controller.detection_method_changed',
                         message.data, 'Controller')

    def receive_data(self, message: Message):
        """Handle processed bitmap and extract contours"""

        self.non_marker_contours = message.data['original_contours']
        self.tolerance_contours = message.data['tolerance_contours']
        self.bitmap = message.data['original_bitmap']
        self.tolerance_bitmap = message.data['tolerance_bitmap']
        self.pixels_per_inch = message.data['pixels_per_inch']
        self.current_tolerance_mm = message.data['current_tolerance_mm']
        self.count_markers = message.data['count_markers']
        self.dimensions_inches = message.data['dimensions_inches']
        self.dimensions_mm = message.data['dimensions_mm']
        self.dimensions_pixels = message.data['dimensions_pixels']

        # Forward contours to any components that need bitmap data
        self.bus.publish('controller.contour_data', message.data, 'Controller')

    def receive_detection_method(self, message: Message):

        self.bus.publish('controller.method_changed',
                         message.data, 'Controller')

    def on_perimeter_change(self, message: Message):
        """Handle perimeter change"""
        self.bus.publish('controller.perimeter_changed',
                         message.data, 'Controller')
        
    def on_status_message(self, message: Message):
        """Handles status messages from models and passes them to the 
            view to be displayed in the status bar. The data will always be a string."""
        
        status_text = message.data # Expecting a string message

        if status_text:
            # Forward status messages to the view
            self.bus.publish('controller.status_message', status_text, 'Controller')
        else:
            pass # Ignore empty status messages



# ================= export related methods ==================

    def on_export_pressed(self, message: Message):
        """Handle export request - delegates to ExportManager"""
        try:
            # Import the export manager
            from models.export_manager import ExportManager
            
            # Prepare bitmap data
            bitmap_data = {
                'bitmap': self.tolerance_bitmap,
                'pixels_per_inch': self.pixels_per_inch,
                'tolerance_mm': self.current_tolerance_mm,
                'metadata': {
                    'detection_method': getattr(self, 'detection_method', 'Unknown'),
                    'threshold': getattr(self, 'threshold', 'Unknown'),
                    'original_resolution': getattr(self, 'original_resolution', 'Unknown')
                }
            }
            
            # Prepare export settings
            export_settings = {
                'dpi': self.dpi,
                'tolerance_mm': self.current_tolerance_mm or 0.0,
                'export_format': 'svg'
            }
            
            # Create export manager and export
            export_manager = ExportManager()
            result = export_manager.export_with_dialog(bitmap_data, export_settings)
            
            if result and result['success']:
                print(f"✅ Export completed: {result['svg_path']}")
                
                # Optionally store the result for future reference
                self.last_export_result = result
                
            else:
                print("Export cancelled or failed")
                
        except ImportError as e:
            print(f"❌ Missing export module: {e}")
            try:
                from tkinter import messagebox
                messagebox.showerror("Export Error", 
                    f"Export functionality not available: {e}")
            except:
                pass
                
        except Exception as e:
            print(f"❌ Export error: {e}")
            import traceback
            traceback.print_exc()
