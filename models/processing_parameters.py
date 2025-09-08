# models/processing_parameters.py


class ProcessingParameters:
    def __init__(self):
        # Detection method selection
        self.detection_method = "brightness"  # "brightness" or "canny"

        # Brightness method parameters
        self.brightness_threshold = 80
        self.brightness_blur_kernel = (5, 5)
        self.brightness_min_area = 100
        self.brightness_max_area = None

        # Canny method parameters
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.canny_blur_kernel = (5, 5)
        self.canny_min_area = 100
        self.canny_max_area = None

        # Common parameters
        self.min_contour_area = 100

        # Added Perimeter Values

        self.perimeter = 0

    def set_parameter(self, name, value):
        """Set a single parameter"""
        if hasattr(self, name):
            setattr(self, name, value)
            return True
        return False

    def get_parameters(self):
        """Get all parameters as a dictionary"""
        return {attr: getattr(self, attr) for attr in dir(self)
                if not attr.startswith('_') and not callable(getattr(self, attr))}
