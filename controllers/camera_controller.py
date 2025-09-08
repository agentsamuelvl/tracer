# camera_controller.py


from models.camera_manager import CameraManager
from views.camera_view import CameraFeed
import cv2
import threading
import queue
import time


class CameraController:
    def __init__(self, parent):
        self.interval = 33
        self.display_on = True
        self.image = None
        self.tk_image = None
        self.model = CameraManager()
        self.view = CameraFeed(parent, self)

        # Threading components
        self.frame_queue = queue.Queue(maxsize=2)  # Holds max 2 frames
        self.camera_thread = None
        self.running = False
        self.thread_lock = threading.Lock()

        self.image_captured_callbacks = []

        self.start_camera_thread()
        self.start_gui_updates()
        print("Camera controller initialized")

    def add_image_captured_callback(self, callback):
        """Add a function to call when image is captured"""
        self.image_captured_callbacks.append(callback)
        print(
            f"Added callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")

    def start_camera_thread(self):
        """Start the background thread that reads camera frames"""
        self.running = True
        self.camera_thread = threading.Thread(
            target=self._camera_worker, daemon=True)
        self.camera_thread.start()

    def _camera_worker(self):
        """This function runs in the background thread"""
        print("Camera worker thread started")
        while self.running:
            try:
                if self.display_on:

                    tk_frame = self.model.get_tkinter_frame()

                    if tk_frame:
                        try:
                            self.frame_queue.put_nowait(tk_frame)

                        except queue.Full:
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait(tk_frame)
                            except queue.Empty:
                                pass

                time.sleep(0.03)

            except Exception as e:
                print(f"Camera thread error: {e}")
                time.sleep(0.1)
        print("Camera worker thread ended")

    def start_gui_updates(self):
        """Start GUI updates on main thread"""
        self.view.show_camera_display()

    def get_latest_frame(self):
        """Called by GUI thread to get latest frame from queue"""
        try:
            # Get frame from queue without blocking
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def show_video_feed(self):
        self.display_on = True

    def hide_video_feed(self):
        self.display_on = False

    def take_picture(self):
        """Capture image and switch to display mode"""
        self.hide_video_feed()
        captured_image = self.model.capture_frame()

        if captured_image is not None:
            resized_frame = self.model.resize_frame(captured_image)
            self.tk_image = self.model.convert_frame_to_tkinter(resized_frame)
            self.image = captured_image  # Store full-quality image

            del resized_frame
            self.view.show_tk_image()

            # ADD THIS: Notify all callbacks with the captured image
            print(f"Notifying {len(self.image_captured_callbacks)} callbacks")
            for callback in self.image_captured_callbacks:
                try:
                    callback(self.image)
                except Exception as e:
                    print(f"Error in callback: {e}")

            print("image captured and callbacks notified")
            return self.tk_image
        else:
            print("Failed to capture Frame")
            return None

    def reset(self):
        """Reset to live camera feed"""
        print("Reset called")
        self.image = None
        self.tk_image = None
        self.show_video_feed()
        print("Reset completed")

    def cleanup(self):
        """Clean shutdown of threads"""
        self.running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
