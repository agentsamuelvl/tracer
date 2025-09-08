# camera_feed.py

import tkinter as tk
from tkinter import ttk


class CameraFeed:
    """Handles camera display and controls"""

    def __init__(self, parent, controller):
        """Initialize camera panel"""
        self.parent = parent
        self.controller = controller
        self.current_image = None

        self._create_camera_frame()
        self._create_camera_controls()

    def _create_camera_frame(self):
        """Create the main camera display frame"""
        self.camera_frame = tk.Frame(
            self.parent,
            bg="#2d2d2d",
            bd=2,
            width=500,
            height=347,
            highlightbackground="#242424",
            highlightthickness=1
        )
        self.camera_frame.grid(
            row=0,
            column=0,
            sticky='nsew',
            padx=(12, 6),
            pady=(12, 6)
        )

        # Prevent frame from shrinking
        self.camera_frame.grid_propagate(False)

        # Create a label inside the frame
        self.camera_label = tk.Label(
            self.camera_frame,
            text="Starting Camera...",  # Changed initial text
            bg="#2d2d2d",
            fg="white",
            font=("Arial", 12),
            compound='center'  # Added for image centering
        )
        self.camera_label.pack(expand=True, fill='both')  # Added fill='both'

    def show_tk_image(self):
        if self.controller.tk_image is not None:
            self.current_image = None
            self.current_image = self.controller.tk_image
            self.camera_label.config(
                image=self.current_image,
                text="",
                compound="center"
            )

    def show_camera_display(self):
        """Update camera display with latest frame from background thread"""

        try:
            if not self.controller.display_on:
                # Check if we have a captured image to display
                if self.controller.tk_image is not None:

                    # Don't overwrite the captured image - just return
                    pass
                else:
                    self.camera_label.config(
                        image="",
                        text="Video Display Off",
                        fg="orange"
                    )
            else:
                tk_frame = self.controller.get_latest_frame()

                if tk_frame:
                    self.current_image = None
                    self.current_image = tk_frame
                    self.camera_label.config(
                        image=self.current_image,
                        text="",
                        compound='center'
                    )
                else:
                    # Keep current display - don't change anything
                    pass

        except Exception as e:
            print(f"GUI update error: {e}")
            self.camera_label.config(
                image="",
                text="Display Error",
                fg="red"
            )

        self.parent.after(self.controller.interval, self.show_camera_display)

    def _create_camera_controls(self):
        """Create camera control buttons and settings"""
        self.controls_frame = tk.Frame(
            self.parent,
            bg="#2d2d2d",
            bd=2,
            width=500,
            highlightbackground="#242424",
            highlightthickness=1
        )
        self.controls_frame.grid(
            row=1,
            column=0,
            sticky='nsew',
            padx=(12, 6),
            pady=(6, 12)
        )

        # Prevent frame from shrinking
        self.controls_frame.grid_propagate(False)

        # create button to capture image
        self.capture_button = tk.Button(
            self.controls_frame, text="take image", command=self.controller.take_picture)
        self.capture_button.pack(padx=40, pady=40)

        # create button to reset gui
        self.reset_button = tk.Button(
            self.controls_frame, text="reset", command=self.controller.reset)
        self.reset_button.pack(padx=40, pady=40)

    # Your existing methods...

    def add_camera_display(self, widget):
        widget.pack(fill='both', expand=True, padx=5, pady=5)

    def add_control_widget(self, widget, row=0, column=0, **kwargs):
        widget.grid(row=row, column=column, **kwargs)

    def get_camera_frame(self):
        return self.camera_frame

    def get_controls_frame(self):
        return self.controls_frame
