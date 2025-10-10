# models/export_manager.py

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from tkinter import filedialog, messagebox
from models.bitmap_to_svg import BitmapToSVG
from models.resize_image import ResizeBitMap


class ExportManager:
    """Handles all export operations for tool outlines"""
    
    def __init__(self):
        self.svg_converter = BitmapToSVG()
        self.last_export_path = None
        self.export_history = []
        self.use_potrace = True  # Enable/disable Potrace usage
    
    def get_potrace_path(self):
        """Get path to Potrace executable"""
        try:
            if getattr(sys, 'frozen', False):
                # Running as executable
                base_path = Path(sys._MEIPASS)
                potrace_path = base_path / "utils" / "potrace.exe"
            else:
                # Running in development
                script_dir = Path(__file__).parent.parent
                tools_dir = script_dir / "utils"
                potrace_path = tools_dir / "potrace.exe"
            
            if potrace_path.exists():
                return str(potrace_path)
            else:
                print(f"Potrace not found at: {potrace_path}")
                return None
        except Exception as e:
            print(f"Error locating Potrace: {e}")
            return None
    
    def export_with_dialog(self, bitmap_data, export_settings):
        """
        Export tool outline with user file selection
        
        Args:
            bitmap_data: Dict containing bitmap and metadata
            export_settings: Dict containing export parameters
            
        Returns:
            Dict with export results or None if failed/cancelled
        """
        try:
            # Validate inputs
            validation_result = self._validate_export_data(bitmap_data, export_settings)
            if not validation_result['valid']:
                self._show_error("Export Validation Failed", validation_result['message'])
                return None
            
            # Prepare bitmap for export
            processed_bitmap = self._prepare_bitmap_for_export(bitmap_data, export_settings)
            if processed_bitmap is None:
                self._show_error("Bitmap Processing Failed", "Could not process bitmap for export")
                return None
            
            # Generate suggested filename
            suggested_filename = self._generate_filename(export_settings)
            
            # Show file dialog
            file_path = self._show_save_dialog(suggested_filename)
            if not file_path:
                return None  # User cancelled
            
            # Convert to SVG
            svg_content = self.svg_converter.bitmap_to_svg_simple(
                processed_bitmap, 
                export_settings['dpi'], 
                None  # Pass None for output_path since we'll save it ourselves
            )

            if not svg_content:
                self._show_error("SVG Conversion Failed", "Could not convert bitmap to SVG")
                return None
            
            # Save SVG file only
            export_result = self._save_svg_file(
                file_path, 
                svg_content, 
                export_settings
            )
            
            if export_result['success']:
                # Update history and settings
                self.last_export_path = export_result['svg_path']
                self.export_history.append(export_result)
                
                # Show success message
                self._show_success(export_result)
                
            return export_result
            
        except Exception as e:
            error_msg = f"Unexpected error during export: {e}"
            print(f"ERROR: {error_msg}")
            self._show_error("Export Error", error_msg)
            import traceback
            traceback.print_exc()
            return None
    
    def _convert_with_potrace(self, bitmap, export_settings):
        """Convert bitmap to SVG using Potrace"""
        try:
            potrace_path = self.get_potrace_path()
            if not potrace_path:
                print("Potrace not available, falling back to built-in converter")
                return self._convert_with_builtin(bitmap, export_settings)
            
            # Create temporary files
            temp_dir = Path.cwd() / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_bmp = temp_dir / f"temp_{timestamp}.pbm"
            temp_svg = temp_dir / f"temp_{timestamp}.svg"
            
            try:
                cv2.imwrite(str(temp_bmp), bitmap) 
                
                # Run Potrace
                potrace_result = self._run_potrace(potrace_path, temp_bmp, temp_svg, export_settings)
                
                if potrace_result and temp_svg.exists():
                    # Read the generated SVG
                    with open(temp_svg, 'r', encoding='utf-8') as f:
                        svg_content = f.read()
                    
                    print("Successfully converted with Potrace")
                    return svg_content
                else:
                    print("Potrace conversion failed, falling back to built-in")
                    return self._convert_with_builtin(bitmap, export_settings)
                    
            finally:
                # Clean up temporary files
                for temp_file in [temp_bmp, temp_svg]:
                    if temp_file.exists():
                        temp_file.unlink()
                        
        except Exception as e:
            print(f"Error in Potrace conversion: {e}")
            return self._convert_with_builtin(bitmap, export_settings)
    

    def _run_potrace(self, potrace_path, input_pbm, output_svg, export_settings):
        """Run Potrace executable"""
        try:
            import subprocess
            import sys
            
            # Potrace command line options
            cmd = [
                potrace_path,
                str(input_pbm),
                "-s",  # SVG output
                "-o", str(output_svg),
                "--tight",  # Tight bounding box
                "-t", "2",  # Suppress speckles (2 pixel threshold)
                "--invert",  # Invert colors (black on white)
            ]
            
            # Add DPI scaling if needed
            dpi = export_settings.get('dpi', 96)
            if dpi != 72:
                cmd.extend(["-r", str(int(dpi))])
            
            print(f"Running Potrace: {' '.join(cmd)}")
            
            # Run Potrace without showing window
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            
            if result.returncode == 0:
                print("Potrace completed successfully")
                return True
            else:
                print(f"Potrace failed with return code {result.returncode}")
                print(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Potrace timed out")
            return False
        except Exception as e:
            print(f"Error running Potrace: {e}")
            return False
    
    def _convert_with_builtin(self, bitmap, export_settings):
        """Fallback: Convert using built-in SVG converter"""
        try:
            print("Using built-in SVG converter")
            return self.svg_converter.bitmap_to_svg_simple(
                bitmap, 
                export_settings['dpi'], 
                None  # Don't save, just return content
            )
        except Exception as e:
            print(f"Built-in converter failed: {e}")
            return None
    
    # ... rest of your existing methods stay the same ...
    
    def _validate_export_data(self, bitmap_data, export_settings):
        """Validate that we have all required data for export"""
        
        # Check bitmap data
        if not bitmap_data or 'bitmap' not in bitmap_data:
            return {'valid': False, 'message': 'No bitmap data available for export.\n\nPlease capture and process an image first.'}
        
        if bitmap_data['bitmap'] is None:
            return {'valid': False, 'message': 'Bitmap is empty.\n\nPlease capture and process an image first.'}
        
        # Check scale information
        if 'pixels_per_inch' not in bitmap_data or bitmap_data['pixels_per_inch'] is None:
            return {'valid': False, 'message': 'No scale information available.\n\nMake sure ArUco markers are detected properly.'}
        
        # Check export settings
        required_settings = ['dpi', 'tolerance_mm']
        for setting in required_settings:
            if setting not in export_settings:
                return {'valid': False, 'message': f'Missing export setting: {setting}'}
        
        return {'valid': True, 'message': 'Validation passed'}
    
    def _prepare_bitmap_for_export(self, bitmap_data, export_settings):
        """Prepare and resize bitmap for export"""
        try:
            bitmap = bitmap_data['bitmap']
            pixels_per_inch = bitmap_data['pixels_per_inch']
            target_dpi = export_settings['dpi']
            
            print(f"Preparing bitmap for export:")
            print(f"  Original shape: {bitmap.shape}")
            print(f"  Pixels per inch: {pixels_per_inch}")
            print(f"  Target DPI: {target_dpi}")
            
            # Create resizer and resize to target DPI
            resizer = ResizeBitMap(bitmap, pixels_per_inch)
            resized_bitmap = resizer.resize_bitmap_to_dpi(target_dpi)
            
            print(f"  Resized shape: {resized_bitmap.shape}")
            
            return resized_bitmap
            
        except Exception as e:
            print(f"Error preparing bitmap: {e}")
            return None
    
    def _generate_filename(self, export_settings):
        """Generate a descriptive filename with timestamp and settings"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include tolerance if specified
        tolerance = export_settings.get('tolerance_mm', 0)
        if tolerance > 0:
            tolerance_str = f"_tol{tolerance:.1f}mm"
        else:
            tolerance_str = ""
        
        # Include DPI
        dpi = export_settings.get('dpi', 96)
        dpi_str = f"_{dpi}dpi"
        
        return f"tool_outline{tolerance_str}{dpi_str}_{timestamp}.svg"
    
    def _show_save_dialog(self, suggested_filename):
        """Show file save dialog"""
        try:
            return filedialog.asksaveasfilename(
                title="Export Tool Outline as SVG",
                defaultextension=".svg",
                initialfile=suggested_filename,
                filetypes=[
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ],
                confirmoverwrite=True
            )
        except Exception as e:
            print(f"Error showing save dialog: {e}")
            return None
    
    def _save_svg_file(self, file_path, svg_content, export_settings):
        """Save SVG file only"""
        export_result = {
            'success': False,
            'svg_path': None,
            'timestamp': datetime.now(),
            'settings': export_settings.copy()
        }
        
        try:
            file_path = Path(file_path)
            
            # Save main SVG file
            svg_path = self.svg_converter._save_svg_file(svg_content, file_path)
            if not svg_path:
                return export_result
            
            export_result['svg_path'] = svg_path
            export_result['success'] = True
            
            print(f"SVG exported successfully: {svg_path}")
            return export_result
            
        except Exception as e:
            print(f"Error saving SVG file: {e}")
            return export_result
    
    def _show_error(self, title, message):
        """Show error message to user"""
        try:
            messagebox.showerror(title, message)
        except:
            print(f"Error: {title} - {message}")
    
    def _show_success(self, export_result):
        """Show success message to user"""
        try:
            svg_path = Path(export_result['svg_path'])
            file_size = svg_path.stat().st_size
            
            message = (
                f"Export completed successfully!\n\n"
                f"File saved: {svg_path.name} ({file_size:,} bytes)\n"
                f"Location: {svg_path.parent}"
            )
            
            messagebox.showinfo("Export Successful", message)
            
        except Exception as e:
            print(f"Could not show success message: {e}")
    
    def get_export_history(self):
        """Get list of recent exports"""
        return self.export_history.copy()
    
    def get_last_export_directory(self):
        """Get directory of last export for convenience"""
        if self.last_export_path:
            return str(Path(self.last_export_path).parent)
        return None