#!/usr/bin/env python3
"""
Test script to verify image support functionality.
"""
import os
import tempfile
import asyncio
from pathlib import Path
from PIL import Image
import numpy as np

# Add the app directory to the path
import sys
sys.path.append('/home/sujal/Work/TDS-P2v3')

from app.utils import validate_file_type
from app.llm import generate_analysis_code, detect_image_files

def create_test_image_with_data():
    """Create a test image that contains some visual data."""
    # Create a simple chart-like image
    img = Image.new('RGB', (400, 300), color='white')
    
    # We'll create a simple bar chart pattern
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    # Title
    try:
        # Try to use a default font, fall back to basic if not available
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((150, 20), "Video Game Sales", fill='black', font=font)
    
    # Draw some bars (simulating a bar chart)
    bars = [
        ("Mario", 150),
        ("Tetris", 120),
        ("Call of Duty", 100),
        ("Pokemon", 110),
        ("GTA", 95)
    ]
    
    y_start = 60
    bar_height = 30
    bar_spacing = 10
    
    for i, (name, value) in enumerate(bars):
        y = y_start + i * (bar_height + bar_spacing)
        # Draw bar
        draw.rectangle([50, y, 50 + value, y + bar_height], fill='blue')
        # Draw label
        draw.text((55, y + 5), f"{name}: {value}", fill='white', font=font)
    
    return img

async def test_image_file_validation():
    """Test that image file validation works."""
    print("Testing image file validation...")
    
    # Test various image extensions
    test_files = [
        "test.png",
        "test.jpg", 
        "test.jpeg",
        "test.PNG",
        "test.JPG",
        "test.JPEG",
        "test.gif",  # Should fail
        "test.pdf"   # Should fail
    ]
    
    expected_results = [True, True, True, True, True, True, False, False]
    
    for filename, expected in zip(test_files, expected_results):
        result = validate_file_type(filename)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {filename}: {result} (expected {expected})")
    
    print()

def test_image_detection():
    """Test that image detection works."""
    print("Testing image file detection...")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        (temp_path / "data.csv").touch()
        (temp_path / "image1.png").touch()
        (temp_path / "document.txt").touch()
        (temp_path / "chart.jpg").touch()
        (temp_path / "photo.jpeg").touch()
        
        all_files = list(temp_path.glob("*"))
        image_files = detect_image_files(all_files)
        
        print(f"  Total files: {len(all_files)}")
        print(f"  Image files detected: {len(image_files)}")
        print(f"  Image files: {[f.name for f in image_files]}")
        
        expected_image_count = 3
        status = "✅" if len(image_files) == expected_image_count else "❌"
        print(f"  {status} Expected {expected_image_count} images, found {len(image_files)}")
    
    print()

async def test_vision_code_generation():
    """Test that vision model generates appropriate code for image analysis."""
    print("Testing vision model code generation...")
    
    # Check if API key is available
    if not os.getenv('GEMINI_API_KEY'):
        print("  ⚠️  GEMINI_API_KEY not found - skipping vision test")
        return
    
    # Create a temporary directory with a test image
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create and save test image
        test_img = create_test_image_with_data()
        image_path = temp_path / "chart.png"
        test_img.save(image_path)
        
        # Create question file
        question_path = temp_path / "question.txt"
        with open(question_path, 'w') as f:
            f.write("""
            Analyze the chart in the provided image and answer:
            1. How many items are shown in the chart?
            2. Which item has the highest value?
            3. What is the approximate value of the highest item?
            4. Create a horizontal bar chart showing the same data.
            
            Return as JSON array: [count, top_item, top_value, base64_chart]
            """)
        
        file_paths = [question_path, image_path]
        
        print("  Generating code with vision model...")
        generated_code = await generate_analysis_code(
            question="Analyze the chart image and extract the data",
            file_paths=file_paths,
            analysis_type="image",
            sandbox_path=temp_path,
            request_id="vision_test"
        )
        
        if generated_code:
            print("  ✅ Successfully generated vision analysis code")
            
            # Check for image-related imports and functions
            required_patterns = [
                "from PIL import Image",
                "Image.open(",
                "json.dump",
                "base64"
            ]
            
            missing_patterns = []
            for pattern in required_patterns:
                if pattern not in generated_code:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                print(f"  ⚠️  Missing patterns: {missing_patterns}")
            else:
                print("  ✅ All required patterns found in generated code")
            
            # Check for vision-specific elements
            if "PIL" in generated_code or "Image.open" in generated_code:
                print("  ✅ Contains image processing code")
            else:
                print("  ❌ Missing image processing code")
                
        else:
            print("  ❌ Failed to generate vision analysis code")
    
    print()

async def main():
    """Run all image support tests."""
    print("=" * 60)
    print("TESTING IMAGE SUPPORT FUNCTIONALITY")
    print("=" * 60)
    
    # Test file validation
    await test_image_file_validation()
    
    # Test image detection
    test_image_detection()
    
    # Test vision code generation
    await test_vision_code_generation()
    
    print("=" * 60)
    print("IMAGE SUPPORT TESTS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
