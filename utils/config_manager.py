# utils/config_manager.py

import json
import os
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime
import sys


class ConfigManager:
    """Manages camera configuration parameters"""

    def __init__(self, config_dir=None):

        if config_dir is None:
            config_dir = self.get_config_dir()
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "camera_config.json"
        self.backup_file = self.config_dir / "camera_config_backup.json"

        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)

        # Load or create default config
        self.config = self._load_or_create_config()

    def get_config_dir(self):
        """Get config directory that works both in development and as executable"""
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            base_path = Path(sys._MEIPASS)  # PyInstaller temp folder
            # But we want to save config in user's directory, not temp
            import os
            user_config = Path.home() / '.tool_outline_app' / 'config'
            user_config.mkdir(parents=True, exist_ok=True)
            return user_config
        else:
            # Running in normal Python
            return Path('config')

    def _get_default_config(self):
        """Return default camera configuration using your exact original values"""
        return {
            "camera_matrix": {
                "fx": 800,
                "fy": 800,
                "cx": 960,
                "cy": 540
            },
            "distortion_coefficients": {
                "k1": -0.1,       # Mild barrel distortion
                "k2": 0.05,       # Small correction
                "p1": 0.0,        # No tangential distortion
                "p2": 0.0,
                "k3": 0.0,        # No higher-order radial
                "k4": 0.0, "k5": 0.0, "k6": 0.0,
                "s1": 0.0, "s2": 0.0, "s3": 0.0, "s4": 0.0,
                "tx": 0.0, "ty": 0.0
            
            },
            "reference_resolution": {
                "width": 1920,
                "height": 1080
            },
            "metadata": {
                "camera_model": "Default Camera",
                "calibration_date": datetime.now().strftime("%Y-%m-%d"),
                "notes": "Original hardcoded calibration parameters"
            }
        }

    def _load_or_create_config(self):
        """Load existing config or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                print(f"Loaded camera config from {self.config_file}")
                return config
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading config: {e}. Using defaults.")
                return self._create_default_config()
        else:
            return self._create_default_config()

    def _create_default_config(self):
        """Create and save default configuration"""
        config = self._get_default_config()
        self.save_config(config)
        print(f"Created default camera config at {self.config_file}")
        return config

    def get_camera_matrix(self):
        """Get camera matrix as numpy array"""
        try:
            cm = self.config["camera_matrix"]
            return np.array([
                [cm["fx"], 0.0, cm["cx"]],
                [0.0, cm["fy"], cm["cy"]],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
        except KeyError as e:
            print(
                f"Missing camera matrix parameter: {e}. Resetting to defaults.")
            self.reset_to_defaults()
            return self.get_camera_matrix()
        
    def update_reference_resolution(self, width, height):
        """Update reference resolution and scale camera matrix accordingly"""
        old_width, old_height = self.get_reference_resolution()
        
        if old_width == width and old_height == height:
            return  # No change needed
        
        # Calculate scaling factors
        scale_x = width / old_width
        scale_y = height / old_height
        
        # Scale camera matrix
        cm = self.config["camera_matrix"]
        cm["fx"] *= scale_x
        cm["fy"] *= scale_y  
        cm["cx"] *= scale_x
        cm["cy"] *= scale_y
        
        # Update reference resolution
        self.config["reference_resolution"]["width"] = width
        self.config["reference_resolution"]["height"] = height
        
        # Update metadata
        self.config["metadata"]["notes"] = f"Scaled from {old_width}x{old_height} to {width}x{height}"
        self._update_metadata()
        
        self.save_config(self.config)
        print(f"Camera parameters scaled for {width}x{height}")

    def get_distortion_coefficients(self):
        """Get distortion coefficients as numpy array"""
        try:
            dc = self.config["distortion_coefficients"]
            return np.array([
                dc["k1"], dc["k2"], dc["p1"], dc["p2"], dc["k3"],
                dc["k4"], dc["k5"], dc["k6"], dc["s1"], dc["s2"],
                dc["s3"], dc["s4"], dc["tx"], dc["ty"]
            ], dtype=np.float64)
        except KeyError as e:
            print(
                f"Missing distortion coefficient: {e}. Resetting to defaults.")
            self.reset_to_defaults()
            return self.get_distortion_coefficients()

    def get_reference_resolution(self):
        """Get reference resolution as tuple (width, height)"""
        res = self.config["reference_resolution"]
        return (res["width"], res["height"])

    def update_camera_matrix(self, fx=None, fy=None, cx=None, cy=None):
        """Update camera matrix parameters"""
        cm = self.config["camera_matrix"]

        if fx is not None:
            if fx <= 0 or fx > 10000:
                print(f"Warning: fx={fx} seems unrealistic. Using anyway.")
            cm["fx"] = float(fx)
        if fy is not None:
            if fy <= 0 or fy > 10000:
                print(f"Warning: fy={fy} seems unrealistic. Using anyway.")
            cm["fy"] = float(fy)
        if cx is not None:
            cm["cx"] = float(cx)
        if cy is not None:
            cm["cy"] = float(cy)

        self._update_metadata()
        self.save_config(self.config)

    def update_distortion_coefficients(self, **kwargs):
        """Update distortion coefficients"""
        dc = self.config["distortion_coefficients"]

        # Valid coefficient names
        valid_keys = ["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2",
                      "s1", "s2", "s3", "s4", "tx", "ty"]

        for key, value in kwargs.items():
            if key in valid_keys:
                try:
                    dc[key] = float(value)
                except ValueError:
                    print(
                        f"Error: {key}={value} is not a valid number. Ignoring.")
                    continue
            else:
                print(
                    f"Warning: Unknown distortion coefficient '{key}' ignored")

        self._update_metadata()
        self.save_config(self.config)

    def _update_metadata(self):
        """Update metadata when parameters change"""
        self.config["metadata"]["calibration_date"] = datetime.now().strftime(
            "%Y-%m-%d")

    def save_config(self, config=None):
        if config is None:
            config = self.config

        print(f"Attempting to save config to: {self.config_file}")
        print(f"Config has keys: {list(config.keys()) if config else 'None'}")

        try:
            # Ensure directory exists
            self.config_dir.mkdir(exist_ok=True)

            # Test JSON serialization first
            json_str = json.dumps(config, indent=4)
            print(f"JSON serialization successful, {len(json_str)} characters")

            # Write to file
            with open(self.config_file, 'w') as f:
                f.write(json_str)

            # Verify
            if self.config_file.exists():
                size = self.config_file.stat().st_size
                print(f"✅ Config saved successfully, file size: {size} bytes")
            else:
                print("❌ File was not created!")

        except Exception as e:
            print(f"❌ Save config failed: {e}")
            import traceback
            traceback.print_exc()

    def reset_to_defaults(self):
        """Reset configuration to default values"""
        self.config = self._get_default_config()
        self.save_config(self.config)
        print("Reset camera configuration to defaults")

    def get_config_summary(self):
        """Get a summary of current configuration"""
        cm = self.config["camera_matrix"]
        meta = self.config["metadata"]

        return {
            "camera_model": meta["camera_model"],
            "focal_length": f"fx={cm['fx']:.1f}, fy={cm['fy']:.1f}",
            "principal_point": f"cx={cm['cx']:.1f}, cy={cm['cy']:.1f}",
            "calibration_date": meta["calibration_date"],
            "config_file": str(self.config_file)
        }
