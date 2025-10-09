# views/main_gui.py

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import sv_ttk
import numpy as np
import cv2
from utils.message_bus import MessageBus, Message


class View:

    def __init__(self, parent, controller, window_title="Tracer", theme="dark"):
        """Initialize the main application"""
        self.root = parent
        self.window_title = window_title
        self.theme = theme
        self.controller = controller

        # set gui colors
        self.window_color = "#181818"
        self.frame_color = "#1c1c1c"
        self.border_color = "#2f2f2f"

        # Initialize UI components
        self._setup_window()
        self._setup_theme()
        self._create_layout()
        self._create_widgets()
        self._configure_styles()

    def _setup_window(self):
        """Configure main window properties"""
        screen_width = self.root.winfo_screenwidth() - 100
        screen_height = self.root.winfo_screenheight() - 100

        self.root.configure(bg=self.window_color)
        self.root.title(self.window_title)
        self.root.geometry(f"{screen_width}x{screen_height}")
        self.root.minsize(1000, 700)

        self._set_window_icon()

    def _set_window_icon(self):
        """Set the window icon"""
        try:
            import sys
            from pathlib import Path
            
            # Get the correct path whether running as script or executable
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                base_path = Path(sys._MEIPASS)
            else:
                # Running as script - go up from views/ to project root
                base_path = Path(__file__).parent.parent
            
            icon_path = base_path / "app_icon.ico"
            
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
                print(f"✓ Icon loaded from: {icon_path}")
            else:
                print(f"⚠ Icon not found at: {icon_path}")
                
        except Exception as e:
            print(f"Could not load icon: {e}")
            # Silently fail - app will just use default icon

    def _setup_theme(self):
        """Setup the UI theme"""
        self.style = ttk.Style()
        sv_ttk.set_theme(self.theme)

    def _create_layout(self):
        """Configure the grid layout for the main window"""
        self.root.grid_columnconfigure(0, weight=0, minsize=571)
        self.root.grid_columnconfigure(1, weight=1, minsize=700)
        self.root.grid_columnconfigure(2, weight=2, minsize=300)
        self.root.grid_rowconfigure(0, weight=1, minsize=800)
        self.root.grid_rowconfigure(1, weight=0, minsize=150)
        self.root.grid_rowconfigure(2, weight=0, minsize=25)

    def _create_widgets(self):
        """Create all UI components"""
        self.camera_feed = CameraFeed(
            self.root,           # parent first
            self.controller,
            self.window_color,
            self.frame_color,
            self.border_color)

        self.camera_controls = CameraControls(
            self.root,           # parent first
            self.controller,
            self.window_color,
            self.frame_color,
            self.border_color)

        self.preview_window = PreviewWindow(
            self.root,           # parent first
            self.controller,
            self.window_color,
            self.frame_color,
            self.border_color)

        self.control_panel = ControlPanel(
            self.root,           # parent first
            self.controller,
            self.window_color,
            self.frame_color,
            self.border_color)

        self.status_bar = StatusBar(
            self.root,           # parent first
            self.controller,
            self.window_color,
            self.frame_color,
            self.border_color)

    def _configure_styles(self):
        """Configure custom styles for widgets"""
        self.style.configure("TButton", focuscolor='none')
        self.style.configure("TNotebook.Tab", focuscolor="none")
        self.style.configure("TNotebook", focuscolor="none")
        self.style.configure("TScale", focuscolor="none")
        self.style.configure("TRadiobutton", focuscolor="none")
        self.style.configure("TCheckbutton", focuscolor="none")
        

    def cleanup(self):
        """Cleanup GUI resources"""
        try:
            # Stop camera feed display
            if hasattr(self, 'camera_feed') and self.camera_feed:
                self.camera_feed.stop_display()
        except Exception as e:
            pass
        
        try:
            # Clear any tkinter references
            if hasattr(self, 'camera_label'):
                self.camera_label = None
        except Exception as e:
            pass


class CameraFeed:
    def __init__(self, parent, controller, window_color, frame_color, border_color):
        self.controller = controller
        self.parent = parent
        self.window_color = window_color
        self.frame_color = frame_color
        self.border_color = border_color

        self.controller.bus.subscribe('controller.display_image', self.show_captured_image)
        self.controller.bus.subscribe('controller.restart_display', self.restart_display)
        self.controller.bus.subscribe('controller.camera_initialized', self._start_safe_display)
        self.controller.bus.subscribe('controller.camera_stopped', self.stop_display)
        
        

        self._create_camera_frame()
        self._create_widgets()

        self._display_running = False
        self._update_job = None
        self._loading_job = None  # NEW: For loading animation
        self._loading_dots = 0    # NEW: For loading animation counter

    def _create_camera_frame(self):
        """Create the camera display frame with consistent sizing"""
        self.camera_border_frame = tk.Frame(self.parent, bg=self.border_color, bd=1)
        self.camera_border_frame.grid(row=0, column=0, sticky='nsew', padx=(8, 4), pady=(8, 4))
        
        # FIXED: Ensure consistent frame size - flipped to vertical
        self.camera_frame = tk.Frame(
            self.camera_border_frame, 
            bg=self.frame_color, 
            width=571, 
            height=800
        )
        self.camera_frame.pack(fill='both', expand=True)
        self.camera_frame.grid_propagate(False)  # IMPORTANT: Prevent size changes
        self.camera_frame.pack_propagate(False)  # ADDED: Also prevent pack from changing size

    def _create_widgets(self):
        """Create the camera display label"""
        self.camera_label = tk.Label(
            self.camera_frame,
            text="Camera Off",
            bg="#121212",
            fg="white",
            font=("Arial", 12),
            compound='center'
        )
        self.camera_label.pack(expand=True, fill='both')
    

    def _start_safe_display(self, message=None):
        """Start the display loop safely"""
        
        
        if not self._display_running and self.controller.display_on:
            self._display_running = True
            self.parent.after(500, self._safe_display_loop)

    def _safe_display_loop(self):
        """FIXED: Create tkinter objects HERE in GUI thread"""
        try:
            # Cancel any pending update
            if self._update_job:
                self.parent.after_cancel(self._update_job)
                self._update_job = None

            if not self.controller.display_on or not self._display_running:
                self._display_running = False
                return

            # Get RAW frame data from camera (no tkinter objects)
            frame_data = self._get_frame_data_safe()
            
            if frame_data and 'display_frame' in frame_data:
                try:
                    display_frame = frame_data['display_frame']
                    # Rotate the frame 90 degrees clockwise
                    display_frame = cv2.rotate(display_frame, cv2.ROTATE_90_CLOCKWISE)
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    tk_image = ImageTk.PhotoImage(pil_image)
                    
                    self.camera_label.config(image=tk_image, text="")
                    self.camera_label.image = tk_image
                    
                except Exception as e:
                    pass
            else:
                self.camera_label.config(image="", text="Camera Off", fg="white")

            # Schedule next update
            self._update_job = self.parent.after(40, self._safe_display_loop)  # ~25 FPS
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # Try to recover
            self.camera_label.config(image="", text="Camera Error - Recovering...", fg="white")
            self._update_job = self.parent.after(1000, self._safe_display_loop)

    def _get_frame_data_safe(self):
        """Safely get RAW frame data from camera manager"""
        try:
            if hasattr(self.controller, 'camera_manager'):
                frame_data = self.controller.camera_manager.get_latest_frame_data()
                if frame_data:
                    return frame_data
            return None
        except Exception as e:
            return None
        
    def show_captured_image(self, message):
        """Handle captured image display"""
        try:
            display_image = message.data
            if display_image is not None:
                # Rotate the captured image 90 degrees clockwise
                display_image = cv2.rotate(display_image, cv2.ROTATE_90_CLOCKWISE)
                # Convert in GUI thread (SAFE!)
                tk_image = ImageTk.PhotoImage(Image.fromarray(display_image))
                self.camera_label.config(image=tk_image, text="")
                self.camera_label.image = tk_image
        except Exception as e:
            pass

    def restart_display(self, message=None):
        """Restart display immediately without delay"""
        
        # Stop current display
        if self._display_running:
            self._display_running = False
            
        # Cancel any pending updates
        if self._update_job:
            self.parent.after_cancel(self._update_job)
            self._update_job = None
        
        # Start immediately if controller display is on
        if self.controller.display_on:
            self._display_running = True
            self._safe_display_loop()
        else:
            self.camera_label.config(image="", text="Camera Off", fg="white")


    def stop_display(self, message=None):
        """Stop display cleanly"""
        
        if self._display_running:
            self._display_running = False
            
        
        
        # FIXED: Ensure frame maintains size when showing "Camera Off"
        self.camera_label.config(image="", text="Camera Off", fg="white")
        
        if self._update_job:
            self.parent.after_cancel(self._update_job)
            self._update_job = None
            
        # IMPORTANT: Re-enforce frame size constraints
        self.camera_frame.grid_propagate(False)
        self.camera_frame.pack_propagate(False)


class CameraControls:
    def __init__(self, parent, controller, window_color, frame_color, border_color):
        self.parent = parent
        self.controller = controller
        self.bus = controller.bus
        self.window_color = window_color
        self.frame_color = frame_color
        self.border_color = border_color
        self._create_camera_controls_frame()
        self._create_widgets()

        self.bus.subscribe('controller.contour_data', self.update_aruco_count)
        self.bus.subscribe('controller.contour_data', self.update_tool_dimensions)
        self.bus.subscribe('view.user_clicked_reset', self.reset)


    def _create_camera_controls_frame(self):
        # Create the main border frame for all controls under the camera
        self.camera_border_frame = tk.Frame(self.parent, bg=self.border_color, bd=1)
        self.camera_border_frame.grid(
            row=1,
            column=0,
            sticky='nsew',
            padx=(8, 4),
            pady=(4, 4)
        )

        # Create the main container frame
        self.main_container_frame = tk.Frame(
            self.camera_border_frame,
            bg=self.frame_color,
            width=500,
        )
        self.main_container_frame.pack(fill='both', expand=True)

    def _create_widgets(self):
        # Create two sub-frames within the main container
        self.main_container_frame.grid_rowconfigure(0, weight=1)
        self.main_container_frame.grid_columnconfigure(0, weight=1)
        self.main_container_frame.grid_columnconfigure(1, weight=1)

        self.controls_frame = tk.Frame(
            self.main_container_frame,
            bg=self.window_color,
            width=250,
            height=100
        )
        self.controls_frame.grid(row=0, column=0, padx=(6, 3), pady=6, sticky="nsew")
        self.controls_frame.grid_propagate(False)  # keeps fixed size

        self.status_frame = tk.Frame(
            self.main_container_frame,
            bg=self.window_color,
            width=250,
            height=100
        )
        self.status_frame.grid(row=0, column=1, padx=(3, 6), pady=6, sticky="nsew")
        self.status_frame.grid_propagate(False)

        # --- CONTROLS frame widgets (grid them left to right) ---
        self.camera_settings_button = ttk.Button(self.controls_frame, text="Camera Settings", takefocus=False, command=self.open_camera_settings)
        self.start_camera_button = ttk.Button(self.controls_frame, text="Start Camera", takefocus=False, command=self.start_camera)
        self.stop_camera_button = ttk.Button(self.controls_frame, text="Stop Camera", takefocus=False, command=self.stop_camera)

        # Layout in a single row, evenly spaced
        self.camera_settings_button.pack(side='top', fill='x', padx=5, pady=5)
        self.start_camera_button.pack(side='top', fill='x', padx=5, pady=5)
        self.stop_camera_button.pack(side='top', fill='x', padx=5, pady=5)

        # --- STATUS frame widgets (force left alignment) ---
        self.aruco_label = ttk.Label(self.status_frame, text="Aruco Markers: 0", anchor="w")
        self.dimensions_label_inches = ttk.Label(self.status_frame, text="Tool Dimensions (Inches): 0 in x 0 in", anchor="w")
        self.dimensions_label_pixels = ttk.Label(self.status_frame, text="Tool Dimensions (Pixels): 0 px x 0 px", anchor="w")
        self.dimensions_label_mm = ttk.Label(self.status_frame, text="Tool Dimensions (mm): 0 mm x 0 mm", anchor="w")

        # Configure one expanding column so labels hug the left
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.aruco_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.dimensions_label_inches.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.dimensions_label_pixels.grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.dimensions_label_mm.grid(row=3, column=0, sticky="w", padx=5, pady=2)
    
    def start_camera(self):
        self.bus.publish('view.user_clicked_start_camera', {}, 'CameraControls')

    def stop_camera(self):
        self.bus.publish('view.user_clicked_stop_camera', {}, 'CameraControls')
    
    def update_aruco_count(self, message: Message):
        count = message.data['count_markers']
        self.aruco_label.config(text=f"Aruco Markers: {count}")

    def update_tool_dimensions(self, message: Message):
        dimensions_inches = message.data['dimensions_inches']
        dimensions_mm = message.data['dimensions_mm']
        dimensions_pixels = message.data['dimensions_pixels']
        self.dimensions_label_inches.config(text=f"Tool Dimensions (Inches): {dimensions_inches}")
        self.dimensions_label_mm.config(text=f"Tool Dimensions (mm): {dimensions_mm}")
        self.dimensions_label_pixels.config(text=f"Tool Dimensions (Pixels): {dimensions_pixels}")

    def reset(self, message: Message):
        self.aruco_label.config(text=f"Aruco Markers: 0") 
        self.dimensions_label_inches.config(text=f"Tool Dimensions (Inches): 0 in x 0 in")
        self.dimensions_label_mm.config(text=f"Tool Dimensions (mm): 0 mm x 0 mm")
        self.dimensions_label_pixels.config(text=f"Tool Dimensions (Pixels): 0 px x 0 px")

    def open_camera_settings(self):
        """Open camera settings window"""

        try:
            # Create and show the camera settings window
            settings_window = CameraSettingsWindow(self.parent, self.controller)
        except Exception as e:
            # Show error dialog
            try:
                from tkinter import messagebox
                messagebox.showerror("Error", f"Failed to open camera settings: {e}")
            except:
                pass


class PreviewWindow:
    def __init__(self, parent, controller, window_color, frame_color, border_color):
        self.parent = parent
        self.controller = controller
        self.bus = controller.bus
        self.window_color = window_color
        self.frame_color = frame_color
        self.border_color = border_color

        self._create_preview_frame()
        self._create_preview_canvas()

        # Subscribe to enhanced contour data and fallback
        self.bus.subscribe('controller.contour_data',
                           self.update_preview_with_enhanced_data)
        self.bus.subscribe('controller.contours',
                           self.update_preview_with_contours)  # Fallback
        self.bus.subscribe('controller.reset_request', self.clear_preview)

        # Store current data for resize events
        self.current_contour_data = None
        self.current_contours = None

    def _create_preview_frame(self):
        """Create the preview frame"""
        self.preview_border_frame = tk.Frame(
            self.parent, bg=self.border_color, bd=1)
        self.preview_border_frame.grid(
            row=0, column=1, rowspan=2, sticky='nsew',
            padx=(4, 4), pady=(8, 4)
        )

        self.preview_frame = tk.Frame(
            self.preview_border_frame,
            bg=self.frame_color, bd=0, highlightthickness=0, width=250
        )
        self.preview_frame.pack(fill='both', expand=True)
        self.preview_frame.grid_propagate(False)

    def _create_preview_canvas(self):
        """Create canvas for polygon drawing"""
        self.preview_canvas = tk.Canvas(
            self.preview_frame,
            bg='white',
            highlightthickness=0, bd=0
        )
        self.preview_canvas.pack(fill='both', expand=True, padx=5, pady=5)

        # Bind to canvas resize events
        self.preview_canvas.bind('<Configure>', self._on_canvas_resize)

        # Initial message
        self._show_status_message("Waiting to process image")

    def _on_canvas_resize(self, event):
        """Handle canvas resize - redraw current contours if available"""
        if self.current_contour_data is not None:
            # Use enhanced data if available
            contour_data = self.current_contour_data
            original_contours = contour_data['original_contours']
            tolerance_contours = contour_data['tolerance_contours']

            if original_contours:
                self._draw_enhanced_contours(
                    original_contours[0], tolerance_contours[0] if tolerance_contours else None)
        elif self.current_contours is not None:
            # Fallback to basic contours
            self._draw_basic_contours(self.current_contours[0])

    def update_preview_with_enhanced_data(self, message: Message):
        """Update preview with enhanced contour data"""
        try:
            contour_data = message.data
            self.current_contour_data = contour_data
            self.current_contours = None  # Clear fallback data

            original_contours = contour_data['original_contours']
            tolerance_contours = contour_data['tolerance_contours']

            # Check if we have any contours to display
            if not original_contours or len(original_contours) == 0:
                self._show_status_message("No contours detected")
                return

            # Always draw both contours - tolerance_contours will be the same as original if tolerance = 0
            self._draw_enhanced_contours(
                original_contours[0],
                tolerance_contours[0] if tolerance_contours else None
            )

        except Exception as e:
            self._show_status_message(f"Display Error: {e}", "red")

    def update_preview_with_contours(self, message: Message):
        """Fallback update method for basic contours"""
        try:
            contours = message.data

            # Only use this if we don't have enhanced data
            if self.current_contour_data is not None:
                return  # Enhanced data takes priority

            self.current_contours = contours

            if not contours:
                self._show_status_message("No contours detected")
                return

            self._draw_basic_contours(contours[0])

        except Exception as e:
            self._show_status_message(f"Display Error: {e}", "red")

    def _draw_enhanced_contours(self, original_contour, tolerance_contour):
        """Draw contours using pre-processed tolerance data"""
        self.preview_canvas.delete("all")

        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.parent.after(50, lambda: self._draw_enhanced_contours(
                original_contour, tolerance_contour))
            return

        # Use tolerance contour if available, otherwise use original
        display_tolerance_contour = tolerance_contour if tolerance_contour is not None else original_contour

        # Auto-rotate both contours
        rect = cv2.minAreaRect(original_contour)
        center, (width, height), angle = rect
        rotation_angle = angle + 90 if width > height else angle

        rotated_original = self._rotate_contour(
            original_contour, center, rotation_angle)
        rotated_tolerance = self._rotate_contour(
            display_tolerance_contour, center, rotation_angle)

        # Scale based on tolerance contour bounds (larger of the two)
        original_points = self._scale_and_center_contour(
            rotated_original, canvas_width, canvas_height, rotated_tolerance)
        tolerance_points = self._scale_and_center_contour(
            rotated_tolerance, canvas_width, canvas_height, rotated_tolerance)

        # Always draw tolerance contour first (red), then original (black)
        if len(tolerance_points) >= 6:
            self.preview_canvas.create_polygon(
                tolerance_points, outline='red', fill='red', width=1)

        if len(original_points) >= 6:
            self.preview_canvas.create_polygon(
                original_points, outline='black', fill='black', width=1)

    def _draw_basic_contours(self, tool_contour):
        """Fallback drawing method for basic contours"""
        self.preview_canvas.delete("all")

        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.parent.after(
                50, lambda: self._draw_basic_contours(tool_contour))
            return

        # Basic drawing without tolerance (since we don't have the data)
        rect = cv2.minAreaRect(tool_contour)
        center, (width, height), angle = rect
        rotation_angle = angle + 90 if width > height else angle

        rotated_contour = self._rotate_contour(
            tool_contour, center, rotation_angle)
        points = self._scale_and_center_contour(
            rotated_contour, canvas_width, canvas_height)

        # Draw filled polygon
        if len(points) >= 6:
            self.preview_canvas.create_polygon(
                points, outline='black', fill='black', width=1
            )

    def _scale_and_center_contour(self, contour, canvas_width, canvas_height, bounds_contour=None):
        """Scale contour to fit canvas using optional bounds contour"""
        if bounds_contour is None:
            bounds_contour = contour

        # Get bounding box from bounds contour
        rx, ry, rw, rh = cv2.boundingRect(bounds_contour)
        scale_x = (canvas_width - 40) / rw
        scale_y = (canvas_height - 40) / rh
        scale = min(scale_x, scale_y, 1.0)

        # Calculate offset to center
        offset_x = (canvas_width - rw * scale) / 2 - rx * scale
        offset_y = (canvas_height - rh * scale) / 2 - ry * scale

        # Convert contour to canvas points
        points = []
        for point in contour.reshape(-1, 2):
            x = point[0] * scale + offset_x
            y = point[1] * scale + offset_y
            points.extend([x, y])

        return points

    def _rotate_contour(self, contour, center, angle_degrees):
        """Rotate a contour around a center point by the given angle in degrees"""
        # Convert angle to radians
        angle_rad = np.radians(angle_degrees)

        # Create rotation matrix
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # Get center coordinates
        cx, cy = center

        # Reshape contour for easier manipulation
        points = contour.reshape(-1, 2).astype(np.float32)

        # Translate to origin
        points[:, 0] -= cx
        points[:, 1] -= cy

        # Apply rotation
        rotated_points = np.zeros_like(points)
        rotated_points[:, 0] = points[:, 0] * \
            cos_angle - points[:, 1] * sin_angle
        rotated_points[:, 1] = points[:, 0] * \
            sin_angle + points[:, 1] * cos_angle

        # Translate back
        rotated_points[:, 0] += cx
        rotated_points[:, 1] += cy

        # Convert back to contour format
        return rotated_points.astype(np.int32).reshape(-1, 1, 2)

    def _show_status_message(self, message, color="black"):
        """Show a status message in the center of the canvas"""
        self.preview_canvas.delete("all")

        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        # If canvas dimensions are too small, schedule a retry
        if canvas_width <= 10 or canvas_height <= 10:
            self.parent.after(50, lambda: self._show_status_message(message, color))
            return

        # Clear any stored contour data when showing a message
        self.current_contour_data = None
        self.current_contours = None

        self.preview_canvas.create_text(
            canvas_width // 2, canvas_height // 2,
            text=message, fill=color, font=("Arial", 10)
        )

    def clear_preview(self, message: Message):
        """Clear the preview canvas"""
        self.current_contour_data = None
        self.current_contours = None
        self._show_status_message("Waiting to process image")


class ControlPanel:
    def __init__(self, parent, controller, window_color, frame_color, border_color):
        self.parent = parent
        self.controller = controller
        self.bus = controller.bus
        self.window_color = window_color
        self.frame_color = frame_color
        self.border_color = border_color
        self._brightness_update_job = None
        self._canny_high_update_job = None
        self._canny_low_update_job = None
        self._tolerance_update_job = None
        self.canny_low_threshold_value = 30
        self.canny_high_threshold_value = 90
        self.brightness_threshold_value = 165
        self._create_control_frame()
        self._create_control_widgets()

    def _create_control_frame(self):
        # create a frame for the controls with a border around it
        self.control_border_frame = tk.Frame(
            self.parent, bg=self.border_color, bd=1)
        self.control_border_frame.grid(
            row=0,
            column=2,
            rowspan=2,
            sticky='nsew',
            padx=(4, 8),
            pady=(8, 4)
        )

        self.control_frame = tk.Frame(
            self.control_border_frame,
            bg=self.frame_color,
            bd=0,
            highlightthickness=0,
            width=250
        )
        self.control_frame.pack(fill='both', expand=True)

        # Title
        title_label = ttk.Label(self.control_frame, text="Controls", font=("Arial", 12, "bold"))
        title_label.pack(pady=(10, 15))

    def _create_control_widgets(self):
        # Process Image button
        self.process_button = ttk.Button(self.control_frame, text="Process Image", takefocus=False, command=self.process_image)
        self.process_button.pack(fill='x', padx=10, pady=5)

        # Reset button
        self.reset_button = ttk.Button(self.control_frame, text="Reset", takefocus=False, command=self.reset)
        self.reset_button.pack(fill='x', padx=10, pady=5)

        # Detection method selection
        method_frame = ttk.LabelFrame(self.control_frame, text="Detection Method")
        method_frame.pack(fill='x', padx=10, pady=15)

        self.method_var = tk.StringVar(value="brightness")

        brightness_radio = ttk.Radiobutton(
            method_frame,
            text="Brightness Thresholding",
            variable=self.method_var,
            value="brightness",
            command=self._select_brightness

        )
        brightness_radio.pack(anchor='w', padx=10, pady=5)

        canny_radio = ttk.Radiobutton(
            method_frame,
            text="Canny Edge Detection",
            variable=self.method_var,
            value="canny",
            command=self._select_canny
        )
        canny_radio.pack(anchor='w', padx=10, pady=5)

        # Parameter controls
        params_frame = ttk.LabelFrame(self.control_frame, text="Parameters")
        params_frame.pack(fill='x', padx=10, pady=15)

        # Brightness threshold slider
        ttk.Label(params_frame, text="Brightness Threshold").pack(
            anchor='w', padx=10, pady=(10, 0))
        self.brightness_label = ttk.Label(params_frame, text=f"{self.brightness_threshold_value}")
        self.brightness_label.pack(anchor='w', padx=10)
        self.brightness_threshold = ttk.Scale(
            params_frame,
            from_=0, to=255,
            command=self._on_brightness_slider_move
        )
        self.brightness_threshold.set(self.brightness_threshold_value)
        self.brightness_threshold.pack(fill='x', padx=10, pady=5)

        # canny lower threshold
        ttk.Label(params_frame, text="Canny Low Threshold").pack(
            anchor='w', padx=10, pady=(15, 0))
        self.canny_low_label = ttk.Label(params_frame, text=f"{self.canny_low_threshold_value}")
        self.canny_low_label.pack(anchor='w', padx=10)
        self.canny_low_threshold = ttk.Scale(
            params_frame,
            from_=0, to=255,
            command=self._on_low_threshold_slider_move
        )
        self.canny_low_threshold.set(self.canny_low_threshold_value)
        self.canny_low_threshold.pack(fill='x', padx=10, pady=5)

        # canny high threshold
        ttk.Label(params_frame, text="Canny High Threshold").pack(
            anchor='w', padx=10, pady=(15, 0))
        self.canny_high_label = ttk.Label(params_frame, text=f"{self.canny_high_threshold_value}")
        self.canny_high_label.pack(anchor='w', padx=10)
        self.canny_high_threshold = ttk.Scale(
            params_frame,
            from_=0, to=255,
            command=self._on_high_threshold_slider_move
        )
        self.canny_high_threshold.set(self.canny_high_threshold_value)
        self.canny_high_threshold.pack(fill='x', padx=10, pady=5)

        # Tolerance control
        ttk.Label(params_frame, text="Spacing (mm)").pack(
            anchor='w', padx=10, pady=(15, 0))
        self.tolerance_label = ttk.Label(params_frame, text="0.0")
        self.tolerance_label.pack(anchor='w', padx=10)
        self.tolerance_scale = ttk.Scale(
    params_frame, from_=0, to=4, command=self._on_tolerance_slider_move)  
        self.tolerance_scale.set(0.0)
        self.tolerance_scale.pack(fill='x', padx=10, pady=(5, 10))

        # Export button at the bottom
        self.export_button = ttk.Button(
            self.control_frame, text="Export", takefocus=False, command=self._on_export_pressed)
        self.export_button.pack(side='bottom', fill='x', padx=10, pady=(20, 10))

    def process_image(self):
        self.controller.bus.publish('view.user_clicked_capture', {}, 'ControlPanel')
        self.controller.bus.publish('view.user_clicked_process', {}, 'ControlPanel')

    def reset(self):
        self.controller.bus.publish('view.user_clicked_reset', {}, 'ControlPanel')

    def _on_brightness_slider_move(self, value):
        """Called on every slider move — schedules a throttled publish."""
        if self._brightness_update_job is not None:
            self.parent.after_cancel(self._brightness_update_job)
        # schedule an update after 20ms; if more moves happen, this gets reset
        self._brightness_update_job = self.parent.after(
            20, self._publish_brightness_threshold)

    def _publish_brightness_threshold(self):
        """Actually sends the brightness value through the bus."""
        self._brightness_update_job = None
        self.brightness_threshold_value = int(self.brightness_threshold.get())
        self.brightness_label.config(text=f"{self.brightness_threshold_value}")
        self.bus.publish("view.brightness_threshold_changed",
                         self.brightness_threshold_value, "ControlPanel")
        
    def _on_low_threshold_slider_move(self, value):
        """Called on every slider move — schedules a throttled publish."""
        if self._canny_low_update_job is not None:
            self.parent.after_cancel(self._canny_low_update_job)
        # schedule an update after 20ms; if more moves happen, this gets reset
        self._canny_low_update_job = self.parent.after(
            20, self._publish_canny_low_threshold)

    def _publish_canny_low_threshold(self):
        """Actually sends the canny low value through the bus."""
        self._canny_low_update_job = None
        self.canny_low_threshold_value = int(self.canny_low_threshold.get())
        self.canny_low_label.config(text=f"{self.canny_low_threshold_value}")
        self.bus.publish("view.canny_low_threshold_changed",
                         self.canny_low_threshold_value, "ControlPanel")

    def _on_high_threshold_slider_move(self, value):
        """Called on every slider move — schedules a throttled publish."""
        if self._canny_high_update_job is not None:
            self.parent.after_cancel(self._canny_high_update_job)
        # schedule an update after 20ms; if more moves happen, this gets reset
        self._canny_high_update_job = self.parent.after(
            20, self._publish_canny_high_threshold)

    def _publish_canny_high_threshold(self):
        """Actually sends the canny high value through the bus."""
        self._canny_high_update_job = None
        self.canny_high_threshold_value = int(self.canny_high_threshold.get())
        self.canny_high_label.config(text=f"{self.canny_high_threshold_value}")
        self.bus.publish("view.canny_high_threshold_changed",
                         self.canny_high_threshold_value, "ControlPanel")

    def _on_tolerance_slider_move(self, value):
        """Called on every slider move — schedules a throttled publish."""
        # Update the label immediately for visual feedback
        tolerance = float(value)
        self.tolerance_label.config(text=f"{tolerance:.1f}")
        
        # Cancel any pending update
        if self._tolerance_update_job is not None:
            self.parent.after_cancel(self._tolerance_update_job)
        
        # Schedule an update after 100ms delay
        self._tolerance_update_job = self.parent.after(
            10, lambda: self._publish_tolerance_change(tolerance)
        )

    def _publish_tolerance_change(self, tolerance_value):
        """Actually sends the tolerance value through the bus."""
        self._tolerance_update_job = None
        self.bus.publish("view.tolerance_changed", tolerance_value, "ControlPanel")

    def _select_brightness(self):
        """publish brightness as detection method to the controller"""
        self.bus.publish("view.detection_method_changed",
                         "brightness", "ControlPanel")

    def _select_canny(self):
        """publish canny as detection method to the controller"""
        self.bus.publish("view.detection_method_changed",
                         "canny", "ControlPanel")

    def _on_export_pressed(self):
        """Handle export button press"""
        self.bus.publish("view.export_pressed", {}, "ControlPanel")


class CameraSettingsWindow:
    """Simplified camera distortion parameter settings window"""

    def __init__(self, parent, controller):
        self.parent = parent
        self.controller = controller
        self.bus = controller.bus

        # Create the popup window
        self.window = tk.Toplevel(parent)
        self.window.title("Camera Settings")
        self.window.geometry("600x700")
        self.window.resizable(True, True)

        # Set window icon if available
        self._set_window_icon()

        # configure styles
        self._configure_focus_styles()

        # Make it modal
        self.window.transient(parent)
        self.window.grab_set()

        # Initialize variables
        self.matrix_vars = {}
        
        # Load current configuration
        self._load_current_config()

        # Create UI
        self._create_ui()

        # Center the window
        self._center_window()

        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self._close_window)

    def _set_window_icon(self):
        """Set the popup window icon"""
        try:
            import sys
            from pathlib import Path
            
            # Get the correct path whether running as script or executable
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                base_path = Path(sys._MEIPASS)
            else:
                # Running as script - go up from views/ to project root
                base_path = Path(__file__).parent.parent
            
            icon_path = base_path / "settings.ico"
            
            if icon_path.exists():
                self.window.iconbitmap(str(icon_path))
                print(f"✓ Settings window icon loaded from: {icon_path}")
            else:
                print(f"⚠ Icon not found at: {icon_path}")
                
        except Exception as e:
            print(f"Could not load settings window icon: {e}")
    def _load_current_config(self):
        """Load current camera configuration"""
        try:
            # Get current config from camera manager
            config_mgr = self.controller.camera_manager.config_manager
            self.config = config_mgr.config.copy()
        except Exception as e:
            self.config = {}

    def _configure_focus_styles(self):
        pass
    

    def _create_ui(self):
        """Create the user interface"""
        # Main container
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.configure(takefocus=0)
        self.notebook.pack(fill='both', expand=True, pady=(0, 10))

        # Camera Matrix Tab
        self._create_camera_matrix_tab()

        # Distortion Coefficients Tab
        self._create_distortion_tab()

        # Info Tab
        self._create_info_tab()

        # Bottom button frame
        self._create_button_frame(main_frame)

    def _create_camera_matrix_tab(self):
        """Create camera matrix parameter tab"""
        # Create main frame for this tab
        matrix_tab_frame = ttk.Frame(self.notebook)
        matrix_tab_frame.configure(takefocus=0)


        # Create scrollable frame
        canvas = tk.Canvas(matrix_tab_frame, 
                   highlightthickness=0,
                   highlightcolor="black",
                   highlightbackground="black",
                   takefocus=0)
        scrollbar = ttk.Scrollbar(
            matrix_tab_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Camera matrix parameters
        matrix_frame = ttk.LabelFrame(
            scrollable_frame, text="Camera Matrix Parameters")
        matrix_frame.pack(fill='x', padx=5, pady=5)

        # Focal length parameters
        focal_frame = ttk.LabelFrame(matrix_frame, text="Focal Length")
        focal_frame.pack(fill='x', padx=5, pady=5)

        self._create_parameter_input(focal_frame, "fx", "Focal Length X",
                                     self.config.get("camera_matrix", {}).get("fx", 0))
        self._create_parameter_input(focal_frame, "fy", "Focal Length Y",
                                     self.config.get("camera_matrix", {}).get("fy", 0))

        # Principal point parameters
        principal_frame = ttk.LabelFrame(matrix_frame, text="Principal Point")
        principal_frame.pack(fill='x', padx=5, pady=5)

        self._create_parameter_input(principal_frame, "cx", "Principal Point X",
                                     self.config.get("camera_matrix", {}).get("cx", 0))
        self._create_parameter_input(principal_frame, "cy", "Principal Point Y",
                                     self.config.get("camera_matrix", {}).get("cy", 0))

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.notebook.add(matrix_tab_frame, text="Camera Matrix")

    def _create_distortion_tab(self):
        """Create distortion coefficients tab"""
        # Create main frame for this tab
        distortion_tab_frame = ttk.Frame(self.notebook)
        distortion_tab_frame.configure(takefocus=0)

        # Create scrollable frame
        canvas = tk.Canvas(distortion_tab_frame, 
                   highlightthickness=0,
                   highlightcolor="black",
                   highlightbackground="black",
                   takefocus=0)
        scrollbar = ttk.Scrollbar(
            distortion_tab_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Radial distortion
        radial_frame = ttk.LabelFrame(
            scrollable_frame, text="Radial Distortion")
        radial_frame.pack(fill='x', padx=5, pady=5)

        dist_coeffs = self.config.get("distortion_coefficients", {})

        self._create_parameter_input(
            radial_frame, "k1", "Radial K1", dist_coeffs.get("k1", 0))
        self._create_parameter_input(
            radial_frame, "k2", "Radial K2", dist_coeffs.get("k2", 0))
        self._create_parameter_input(
            radial_frame, "k3", "Radial K3", dist_coeffs.get("k3", 0))
        self._create_parameter_input(
            radial_frame, "k4", "Radial K4", dist_coeffs.get("k4", 0))
        self._create_parameter_input(
            radial_frame, "k5", "Radial K5", dist_coeffs.get("k5", 0))
        self._create_parameter_input(
            radial_frame, "k6", "Radial K6", dist_coeffs.get("k6", 0))

        # Tangential distortion
        tangential_frame = ttk.LabelFrame(
            scrollable_frame, text="Tangential Distortion")
        tangential_frame.pack(fill='x', padx=5, pady=5)

        self._create_parameter_input(
            tangential_frame, "p1", "Tangential P1", dist_coeffs.get("p1", 0))
        self._create_parameter_input(
            tangential_frame, "p2", "Tangential P2", dist_coeffs.get("p2", 0))

        # Thin prism distortion
        prism_frame = ttk.LabelFrame(
            scrollable_frame, text="Thin Prism Distortion")
        prism_frame.pack(fill='x', padx=5, pady=5)

        self._create_parameter_input(
            prism_frame, "s1", "Thin Prism S1", dist_coeffs.get("s1", 0))
        self._create_parameter_input(
            prism_frame, "s2", "Thin Prism S2", dist_coeffs.get("s2", 0))
        self._create_parameter_input(
            prism_frame, "s3", "Thin Prism S3", dist_coeffs.get("s3", 0))
        self._create_parameter_input(
            prism_frame, "s4", "Thin Prism S4", dist_coeffs.get("s4", 0))

        # Tilting distortion
        tilt_frame = ttk.LabelFrame(
            scrollable_frame, text="Tilting Distortion")
        tilt_frame.pack(fill='x', padx=5, pady=5)

        self._create_parameter_input(
            tilt_frame, "tx", "Tilt TX", dist_coeffs.get("tx", 0))
        self._create_parameter_input(
            tilt_frame, "ty", "Tilt TY", dist_coeffs.get("ty", 0))

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.notebook.add(distortion_tab_frame, text="Distortion")

    def _create_info_tab(self):
        """Create information and presets tab"""
        info_frame = ttk.Frame(self.notebook)
        info_frame.configure(takefocus=0)

        # Current configuration info
        current_frame = ttk.LabelFrame(
            info_frame, text="Current Configuration")
        current_frame.pack(fill='x', padx=5, pady=5)

        # Get config summary
        try:
            config_mgr = self.controller.camera_manager.config_manager
            summary = config_mgr.get_config_summary()

            info_text = f"""Focal Length: {summary['focal_length']}
    Principal Point: {summary['principal_point']}
    Calibration Date: {summary['calibration_date']}
    Config File: {summary['config_file']}"""

            info_label = ttk.Label(
                current_frame, text=info_text, justify='left')
            info_label.pack(padx=10, pady=10)

        except Exception as e:
            error_label = ttk.Label(
                current_frame, text=f"Error loading info: {e}")
            error_label.pack(padx=10, pady=10)

        # Camera selection section
        camera_frame = ttk.LabelFrame(info_frame, text="Camera Selection")
        camera_frame.pack(fill='x', padx=5, pady=5)

        # Camera index selection
        camera_selection_frame = ttk.Frame(camera_frame)
        camera_selection_frame.pack(padx=10, pady=10)

        ttk.Label(camera_selection_frame, text="Camera Index:").pack(side='left', padx=(0, 10))

        # Get current camera index from controller (default to 0 if not available)
        try:
            current_index = getattr(self.controller.camera_manager, 'camera_index', 0)
        except:
            current_index = 0

        self.camera_index_var = tk.StringVar(value=str(current_index))
        camera_options = [str(i) for i in range(6)]  # 0-5
        self.camera_index_combo = ttk.Combobox(
            camera_selection_frame, 
            textvariable=self.camera_index_var,
            values=camera_options,
            state="readonly",
            width=10
        )
        self.camera_index_combo.pack(side='left', padx=(0, 10))

        # Apply camera index button
        ttk.Button(
            camera_selection_frame, 
            text="Apply Camera Index", 
            takefocus=False,
            command=self._apply_camera_index
        ).pack(side='left', padx=(10, 0))

        # Note about camera selection
        note_label = ttk.Label(
            camera_frame, 
            text="Note: Camera must be stopped and restarted for changes to take effect.",
            font=("Arial", 8),
            foreground="gray"
        )
        note_label.pack(padx=10, pady=(0, 10))

        # Presets section
        presets_frame = ttk.LabelFrame(info_frame, text="Presets")
        presets_frame.pack(fill='x', padx=5, pady=5)

        preset_buttons_frame = ttk.Frame(presets_frame)
        preset_buttons_frame.pack(padx=5, pady=5)

        ttk.Button(preset_buttons_frame, text="Reset to Defaults", takefocus=False,
                command=self._reset_to_defaults).pack(side='left', padx=5)

        self.notebook.add(info_frame, text="Info & Presets")
    def _create_parameter_input(self, parent, param_key, label_text, current_value):
        """Create a parameter input field"""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=5, pady=2)

        # Label
        label = ttk.Label(frame, text=f"{label_text}:", width=15)
        label.pack(side='left')

        # Entry field
        var = tk.StringVar(value=str(current_value))
        entry = ttk.Entry(frame, textvariable=var, width=20)
        entry.pack(side='left', padx=(5, 0))

        # Store the variable for later access
        self.matrix_vars[param_key] = var

    def _create_button_frame(self, parent):
        """Create bottom button frame"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill='x', pady=5)

        # Left side buttons
        left_frame = ttk.Frame(button_frame)
        left_frame.pack(side='left')

        ttk.Button(left_frame, text="Apply Changes", takefocus=False,
                   command=self._apply_changes).pack(side='left', padx=5)

        # Right side buttons
        right_frame = ttk.Frame(button_frame)
        right_frame.pack(side='right')

        ttk.Button(right_frame, text="Close", takefocus=False,
                   command=self._close_window).pack(side='left', padx=5)

    def _apply_changes(self):
        """Apply all parameter changes"""
        try:
            changes_applied = False

            # Send camera matrix changes
            for param in ['fx', 'fy', 'cx', 'cy']:
                if param in self.matrix_vars:
                    try:
                        value = float(self.matrix_vars[param].get())
                        parameter_data = {
                            'parameter_type': 'camera_matrix',
                            'parameter_name': param,
                            'new_value': value
                        }
                        self.bus.publish(
                            'view.camera_parameter_changed', parameter_data, 'CameraSettings')
                        changes_applied = True
                    except ValueError:
                        pass

            # Send distortion coefficient changes
            dist_params = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'p1', 'p2',
                           's1', 's2', 's3', 's4', 'tx', 'ty']
            for param in dist_params:
                if param in self.matrix_vars:
                    try:
                        value = float(self.matrix_vars[param].get())
                        parameter_data = {
                            'parameter_type': 'distortion_coefficients',
                            'parameter_name': param,
                            'new_value': value
                        }
                        self.bus.publish(
                            'view.camera_parameter_changed', parameter_data, 'CameraSettings')
                        changes_applied = True
                    except ValueError:
                        pass

            if changes_applied:
                messagebox.showinfo(
                    "Success", "Camera parameters applied successfully!")
            else:
                messagebox.showwarning(
                    "Warning", "No valid parameter changes to apply.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply parameters: {e}")

    def _apply_camera_index(self):
        """Apply the selected camera index"""
        try:
            new_index = int(self.camera_index_var.get())
            
            # Send the camera index change through the message bus
            camera_index_data = {
                'parameter_type': 'camera_index',
                'new_value': new_index
            }
            
            self.bus.publish('view.camera_index_changed', camera_index_data, 'CameraSettings')
            
            messagebox.showinfo(
                "Camera Index Updated", 
                f"Camera index set to {new_index}.\n\nPlease stop and restart the camera for changes to take effect."
            )
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please select a valid camera index.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply camera index: {e}")

    def _reset_to_defaults(self):
        """Reset all parameters to defaults"""
        try:
            config_mgr = self.controller.camera_manager.config_manager
            config_mgr.reset_to_defaults()

            # Reload the UI with default values
            self._reload_ui_values()

            messagebox.showinfo(
                "Reset", "Parameters reset to defaults. Click 'Apply Changes' to save.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset to defaults: {e}")

    def _reload_ui_values(self):
        """Reload UI with current configuration values"""
        try:
            config_mgr = self.controller.camera_manager.config_manager
            self.config = config_mgr.config.copy()

            # Update all UI fields
            camera_matrix = self.config.get('camera_matrix', {})
            dist_coeffs = self.config.get('distortion_coefficients', {})

            for param, var in self.matrix_vars.items():
                if param in camera_matrix:
                    var.set(str(camera_matrix[param]))
                elif param in dist_coeffs:
                    var.set(str(dist_coeffs[param]))

        except Exception as e:
            pass

    def _close_window(self):
        """Close the settings window"""
        self.window.grab_release()
        self.window.destroy()

    def _center_window(self):
        """Center the window on the parent"""
        self.window.update_idletasks()

        # Get window dimensions
        window_width = self.window.winfo_width()
        window_height = self.window.winfo_height()

        # Get parent position and size
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Calculate center position
        x = parent_x + (parent_width // 2) - (window_width // 2)
        y = parent_y + (parent_height // 2) - (window_height // 2)

        self.window.geometry(f"+{x}+{y}")


class StatusBar:
     
    def __init__(self, parent, controller, window_color, frame_color, border_color):
        self.parent = parent
        self.controller = controller
        self.bus = controller.bus
        self.window_color = window_color
        self.frame_color = frame_color
        self.border_color = border_color
        self._create_status_frame()

        self.bus.subscribe('controller.status_message', self.update_status)

    def _create_status_frame(self):
        # Create the status bar frame that spans the full bottom width
        self.status_border_frame = tk.Frame(
            self.parent, bg=self.border_color, bd=1)
        self.status_border_frame.grid(
            row=2,
            column=0,
            columnspan=3,
            sticky='nsew',
            padx=0,
            pady=(0, 0)
        )

        self.status_frame = tk.Frame(
            self.status_border_frame,
            bg="black",
            bd=0,
            highlightthickness=0,
            height=25
        )
        self.status_frame.pack(fill='both', expand=True)
        self.status_frame.pack_propagate(False)
        
        # Temporary label to make the status bar visible
        self.status_label = ttk.Label(self.status_frame, text="Tracer 1.0.0")
        self.status_label.pack(pady=4, side = 'left', padx=10)

    def update_status(self, message: Message):
        """Recieve status as a string from the controller and display it in the status 
            bar label.
            Subscription: controller.status_message"""
        
        status_text = message.data # always a string

        if status_text:
            self.status_label.config(text=f"Status: {status_text}")
        else:
            pass



if __name__ == "__main__":

    main_window = tk.Tk()
    app = View(main_window)
    main_window.mainloop()