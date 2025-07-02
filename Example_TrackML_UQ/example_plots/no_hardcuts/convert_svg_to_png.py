#!/usr/bin/env python3
"""
Script to convert all SVG images in subfolders to PNG format and delete the original SVG files.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    # Try different ways to run Inkscape
    inkscape_commands = ['inkscape', 'flatpak run org.inkscape.Inkscape']
    
    for cmd in inkscape_commands:
        try:
            cmd_parts = cmd.split() + ['--version']
            result = subprocess.run(cmd_parts, 
                                  capture_output=True, text=True, check=True, timeout=10)
            print(f"Found Inkscape using '{cmd}': {result.stdout.strip().split()[0]}")
            return cmd.split()[0] if len(cmd.split()) == 1 else cmd
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    # Try alternative: cairosvg (Python library)
    try:
        import cairosvg
        print("Found cairosvg (Python library) - will use as fallback")
        return 'cairosvg'
    except ImportError:
        pass
    
    # Try alternative: rsvg-convert
    try:
        result = subprocess.run(['rsvg-convert', '--version'], 
                              capture_output=True, text=True, check=True, timeout=10)
        print(f"Found rsvg-convert: {result.stdout.strip()}")
        return 'rsvg-convert'
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("Error: No SVG converter found.")
    print("Please install one of the following:")
    print("  1. Inkscape: sudo apt-get install inkscape")
    print("  2. CairoSVG: pip install cairosvg")
    print("  3. rsvg-convert: sudo apt-get install librsvg2-bin")
    print("  4. Try Flatpak Inkscape: flatpak install org.inkscape.Inkscape")
    return None

def convert_svg_to_png(svg_path, png_path, converter):
    """Convert a single SVG file to PNG using the specified converter."""
    try:
        if converter == 'cairosvg':
            # Use cairosvg Python library
            import cairosvg
            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), dpi=300)
            return True
        elif converter == 'rsvg-convert':
            # Use rsvg-convert
            cmd = [
                'rsvg-convert',
                '--format=png',
                '--dpi-x=300',
                '--dpi-y=300',
                '--output', str(png_path),
                str(svg_path)
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        elif converter.startswith('flatpak'):
            # Use Flatpak Inkscape
            cmd = [
                'flatpak', 'run', 'org.inkscape.Inkscape',
                '--export-type=png',
                '--export-dpi=300',
                f'--export-filename={png_path}',
                str(svg_path)
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        else:
            # Use regular Inkscape
            cmd = [
                converter,
                '--export-type=png',
                '--export-dpi=300',
                f'--export-filename={png_path}',
                str(svg_path)
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
    except Exception as e:
        print(f"Error converting {svg_path}: {e}")
        return False

def find_and_convert_svg_files(root_dir, converter):
    """Find all SVG files in subdirectories and convert them to PNG."""
    root_path = Path(root_dir)
    svg_files = list(root_path.rglob('*.svg'))
    
    if not svg_files:
        print("No SVG files found in subdirectories.")
        return
    
    print(f"Found {len(svg_files)} SVG files to convert:")
    for svg_file in svg_files:
        print(f"  {svg_file.relative_to(root_path)}")
    
    converted_count = 0
    failed_count = 0
    
    for svg_file in svg_files:
        # Create PNG filename (same name, different extension)
        png_file = svg_file.with_suffix('.png')
        
        print(f"\nConverting: {svg_file.relative_to(root_path)}")
        
        if convert_svg_to_png(svg_file, png_file, converter):
            print(f"  ✓ Successfully converted to: {png_file.relative_to(root_path)}")
            
            # Delete the original SVG file
            try:
                svg_file.unlink()
                print(f"  ✓ Deleted original SVG file")
                converted_count += 1
            except OSError as e:
                print(f"  ✗ Error deleting SVG file: {e}")
                failed_count += 1
        else:
            print(f"  ✗ Failed to convert")
            failed_count += 1
    
    print(f"\n=== Conversion Summary ===")
    print(f"Successfully converted: {converted_count}")
    print(f"Failed conversions: {failed_count}")
    print(f"Total files processed: {len(svg_files)}")

def main():
    """Main function."""
    print("SVG to PNG Converter")
    print("=" * 50)
    
    # Check dependencies
    converter = check_dependencies()
    if not converter:
        sys.exit(1)
    
    # Get the current directory (where the script is located)
    current_dir = Path(__file__).parent
    print(f"Working directory: {current_dir}")
    print(f"Using converter: {converter}")
    
    # Ask for confirmation
    response = input("\nThis will convert all SVG files in subdirectories to PNG and DELETE the original SVG files. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        sys.exit(0)
    
    # Find and convert SVG files
    find_and_convert_svg_files(current_dir, converter)
    
    print("\nDone!")

if __name__ == "__main__":
    main()