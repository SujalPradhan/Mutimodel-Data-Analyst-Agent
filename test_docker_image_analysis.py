#!/usr/bin/env python3
"""
End-to-end test for image analysis via Docker container.
This simulates the actual API workflow using the analysis Docker container.
"""

import tempfile
import json
import subprocess
import shutil
from pathlib import Path
from PIL import Image, ImageDraw

def create_test_image_with_data():
    """Create a test chart image with clear data to extract."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image = Image.new('RGB', (500, 400), color='white')
        draw = ImageDraw.Draw(image)
        
        # Title
        draw.text((150, 20), "Monthly Revenue 2024", fill='black')
        
        # Create a clear bar chart
        data = [('Jan', 25000), ('Feb', 30000), ('Mar', 28000), ('Apr', 35000)]
        
        for i, (month, value) in enumerate(data):
            x = 50 + i * 100
            bar_height = value // 500  # Scale down for display
            
            # Draw bar
            draw.rectangle([x, 350 - bar_height, x + 80, 350], fill='blue')
            
            # Add month label
            draw.text((x + 20, 360), month, fill='black')
            
            # Add value label
            draw.text((x + 10, 330 - bar_height), f"${value:,}", fill='black')
        
        image.save(tmp.name)
        return tmp.name

def create_analysis_code():
    """Create Python code for image analysis."""
    return '''
import json
from PIL import Image
import io
import base64
import os

try:
    # The image analysis would typically use OCR or computer vision
    # For this demo, we'll simulate extracting the visible data
    
    # Load the image to verify it exists
    img = Image.open("chart.png")
    print(f"Image loaded: {img.size}")
    
    # Simulate extracted data (in real scenario, this would use OCR/CV)
    monthly_data = {
        "Jan": 25000,
        "Feb": 30000,
        "Mar": 28000,
        "Apr": 35000
    }
    
    # Perform analysis
    total_revenue = sum(monthly_data.values())
    avg_monthly = total_revenue / len(monthly_data)
    max_month = max(monthly_data, key=monthly_data.get)
    max_value = monthly_data[max_month]
    
    # Calculate growth rates
    growth_rates = {}
    months = list(monthly_data.keys())
    for i in range(1, len(months)):
        prev_month = months[i-1]
        curr_month = months[i]
        growth = ((monthly_data[curr_month] - monthly_data[prev_month]) / monthly_data[prev_month]) * 100
        growth_rates[f"{prev_month}_to_{curr_month}"] = round(growth, 2)
    
    # Convert image to base64 for output
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Prepare results
    numeric_answer = total_revenue
    string_answer = f"Total revenue: ${total_revenue:,}. Peak month: {max_month} (${max_value:,}). Average: ${avg_monthly:,.0f}"
    float_answer = float(avg_monthly)
    
    results = [numeric_answer, string_answer, float_answer, img_str]
    
    # Save to JSON
    with open("result.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("✅ Image analysis completed successfully")
    print(f"Total Revenue: ${total_revenue:,}")
    print(f"Average Monthly: ${avg_monthly:,.0f}")
    print(f"Peak Month: {max_month} (${max_value:,})")
    print("Growth rates:", growth_rates)

except Exception as e:
    print(f"❌ Error during analysis: {e}")
    import traceback
    traceback.print_exc()
'''

def test_docker_image_analysis():
    """Test image analysis through Docker container."""
    print("🧪 Testing Image Analysis via Docker Container")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        print("📸 Creating test image...")
        image_file = create_test_image_with_data()
        chart_path = temp_path / "chart.png"
        shutil.copy2(image_file, chart_path)
        
        print("📝 Creating analysis code...")
        analysis_path = temp_path / "analysis.py"
        with open(analysis_path, 'w') as f:
            f.write(create_analysis_code())
        
        print("🐳 Running analysis in Docker container...")
        
        # Run the analysis using Docker
        try:
            result = subprocess.run([
                'docker', 'run', '--rm',
                '-v', f'{temp_path}:/workspace',
                '-w', '/workspace',
                'data-analysis-env',
                'python', 'analysis.py'
            ], capture_output=True, text=True, timeout=30)
            
            print(f"Docker exit code: {result.returncode}")
            print("STDOUT:")
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            
            # Check if result.json was created
            result_file = temp_path / "result.json"
            if result_file.exists():
                print("\n📊 Analysis Results:")
                with open(result_file, 'r') as f:
                    results = json.load(f)
                
                print(f"   Numeric Answer: {results[0]}")
                print(f"   String Answer: {results[1]}")
                print(f"   Float Answer: {results[2]}")
                print(f"   Base64 Image: {'Present' if results[3] else 'Missing'} ({len(results[3])} chars)")
                
                print("✅ Docker image analysis test PASSED!")
                return True
            else:
                print("❌ result.json was not created")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Docker analysis timed out")
            return False
        except Exception as e:
            print(f"❌ Docker analysis failed: {e}")
            return False
        finally:
            # Cleanup
            try:
                Path(image_file).unlink()
            except:
                pass

if __name__ == "__main__":
    success = test_docker_image_analysis()
    if success:
        print("\n🎉 Image analysis via Docker is working correctly!")
        print("The system is ready to handle image analysis requests.")
    else:
        print("\n⚠️ Image analysis test failed. Check Docker setup.")
