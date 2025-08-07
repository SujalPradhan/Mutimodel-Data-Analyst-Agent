#!/usr/bin/env python3
"""
Quick test to generate and display vision analysis code.
"""

import os
import sys
import tempfile
import asyncio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add the app directory to the Python path
sys.path.insert(0, '/home/sujal/Work/TDS-P2v3')

from app.llm import generate_analysis_code

def create_sample_chart(path: Path):
    """Create a sample chart image."""
    image = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw title
    draw.text((150, 20), "Sales Data Q1 2024", fill='black')
    
    # Draw bars
    bars = [80, 120, 100, 140]  # Heights
    months = ['Jan', 'Feb', 'Mar', 'Apr']
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, (height, month, color) in enumerate(zip(bars, months, colors)):
        x = 50 + i * 80
        draw.rectangle([x, 250 - height, x + 60, 250], fill=color)
        draw.text((x + 20, 255), month, fill='black')
        draw.text((x + 20, 240 - height), str(height), fill='black')
    
    image.save(path)
    return path

async def test_vision_generation():
    """Test vision model code generation."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test image
        image_path = create_sample_chart(temp_path / "chart.png")
        
        # Generate analysis code
        print("🔮 Generating vision analysis code...")
        
        code = await generate_analysis_code(
            question="Analyze this sales chart. Extract the values for each month and calculate the total sales.",
            file_paths=[image_path],
            analysis_type="image_analysis",
            sandbox_path=temp_path,
            request_id="demo_test"
        )
        
        if code:
            print("✅ Code generation successful!")
            print("\n📄 Generated Code:")
            print("=" * 80)
            print(code)
            print("=" * 80)
        else:
            print("❌ Code generation failed")

if __name__ == "__main__":
    asyncio.run(test_vision_generation())
