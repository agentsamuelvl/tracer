# bitmap_to_svg.py - SHAPE-PRESERVING VERSION

import cv2
import numpy as np
import subprocess
import os
import tempfile
import xml.etree.ElementTree as ET
import re
import math
import sys


class BitmapToSVG:
    """Simple SVG converter - just Potrace, no simplification"""

    def __init__(self):
        pass

    def bitmap_to_svg_simple(self, bitmap, dpi, output_path):
        """
        Simple SVG conversion that handles color inversion
        """
        import tempfile
        import subprocess

        # Debug: Check the bitmap colors
        unique_values = np.unique(bitmap)
        print(f"Bitmap unique values: {unique_values}")
        print(f"Bitmap shape: {bitmap.shape}")
        print(f"Bitmap dtype: {bitmap.dtype}")

        # Count black vs white pixels
        if len(unique_values) >= 2:
            black_pixels = np.sum(bitmap == 0)
            white_pixels = np.sum(bitmap == 255)
            print(
                f"Black pixels: {black_pixels}, White pixels: {white_pixels}")

            # If more black than white, the background is probably black
            if black_pixels > white_pixels:
                print("Detected: Black background, white tool - INVERTING for Potrace")
                bitmap = cv2.bitwise_not(bitmap)
            else:
                print("Detected: White background, black tool - using as-is")

        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as temp_bmp:
            temp_bmp_path = temp_bmp.name

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as temp_svg:
            temp_svg_path = temp_svg.name

        try:
            # Save bitmap
            success = cv2.imwrite(temp_bmp_path, bitmap)
            if not success:
                print("Failed to save temporary bitmap")
                return None

            print(f"Saved temp bitmap to: {temp_bmp_path}")

            # Call Potrace
            cmd = [
                'potrace',
                '--svg',
                '--resolution', f'{int(dpi)}',
                '--output', temp_svg_path,
                '--unit', '1',
                '--blacklevel', '0.5',
                '--fillcolor', '#000000',
                temp_bmp_path
            ]

            print(f"Running Potrace command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Potrace failed with return code {result.returncode}")
                print(f"Stderr: {result.stderr}")
                print(f"Stdout: {result.stdout}")
                return None

            # Read SVG
            try:
                with open(temp_svg_path, 'r') as f:
                    svg_content = f.read()
                print(f"SVG file size: {len(svg_content)} characters")

                # Basic validation
                if '<svg' not in svg_content:
                    print("Warning: SVG content doesn't look valid")
                    print(f"First 200 chars: {svg_content[:200]}")

            except Exception as e:
                print(f"Failed to read SVG file: {e}")
                return None

            # Save to output path
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(svg_content)
                print(f"SVG saved to: {output_path}")

                # Also save the bitmap we used for debugging
                debug_bitmap_path = output_path.replace(
                    '.svg', '_debug_bitmap.png')
                cv2.imwrite(debug_bitmap_path, bitmap)
                print(f"Debug bitmap saved to: {debug_bitmap_path}")

            return svg_content

        except Exception as e:
            print(f"Error during conversion: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Cleanup
            try:
                os.unlink(temp_bmp_path)
                os.unlink(temp_svg_path)
            except:
                pass

    def _ensure_filled_paths(self, svg_content):
        """Ensure all paths in the SVG are filled (not just stroked)"""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(svg_content)

            # Find all path elements and ensure they're filled
            for path in root.iter():
                if path.tag.endswith('path'):
                    path.set('fill', 'black')
                    path.set('stroke', 'none')

            return ET.tostring(root, encoding='unicode')
        except Exception as e:
            print(f"Warning: Could not process SVG paths: {e}")
            return svg_content
        
    # Add this method to your BitmapToSVG class in bitmap_to_svg.py

    def _save_svg_file(self, svg_content, file_path):
        """
        Save SVG content to file with validation
        
        Args:
            svg_content: The SVG content to save
            file_path: Path where to save the file
            
        Returns:
            str: Path where file was saved, or None if failed
        """
        try:
            from pathlib import Path
            
            file_path = Path(file_path)
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Validate SVG content
            if not svg_content or len(svg_content.strip()) < 50:
                print("Error: SVG content is empty or too short")
                return None
                
            if '<svg' not in svg_content.lower():
                print("Error: SVG content doesn't appear to be valid")
                return None

            # Write the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)

            # Verify file was created and has content
            if file_path.exists() and file_path.stat().st_size > 0:
                print(f"✅ SVG saved successfully: {file_path}")
                print(f"   File size: {file_path.stat().st_size:,} bytes")
                return str(file_path)
            else:
                print(f"❌ Failed to create file: {file_path}")
                return None

        except PermissionError:
            print(f"❌ Permission denied: Cannot save to {file_path}")
            return None
            
        except OSError as e:
            print(f"❌ File system error: {e}")
            return None
            
        except Exception as e:
            print(f"❌ Unexpected error saving SVG: {e}")
            return None
