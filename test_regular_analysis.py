#!/usr/bin/env python3
"""
Test script to verify that regular (non-image) analysis still works after adding vision support.
"""
import asyncio
import os
import tempfile
from pathlib import Path
import sys

# Add the app directory to the path
sys.path.append('/home/sujal/Work/TDS-P2v3')

from app.llm import generate_analysis_code

async def test_regular_analysis():
    """Test that regular analysis (without images) still works."""
    
    # Check if API key is available
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  GEMINI_API_KEY not found - skipping test")
        return
    
    # Create a temporary directory with CSV data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a test CSV file
        csv_content = """name,age,salary
Alice,30,50000
Bob,25,45000
Charlie,35,60000
Diana,28,52000"""
        
        csv_path = temp_path / "employees.csv"
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        test_question = """
        Analyze the employee data and answer:
        1. How many employees are there?
        2. What is the average age?
        3. Who has the highest salary?
        4. Create a bar chart showing salaries by name.
        
        Return as JSON array: [count, avg_age, top_earner, base64_chart]
        """
        
        file_paths = [csv_path]
        
        print("Testing regular (non-image) analysis...")
        generated_code = await generate_analysis_code(
            question=test_question,
            file_paths=file_paths,
            analysis_type="general",
            sandbox_path=temp_path,
            request_id="regular_test"
        )
        
        if generated_code:
            print("✅ Successfully generated regular analysis code")
            
            # Check for expected patterns
            required_patterns = [
                "import pandas",
                "read_csv",
                "json.dump",
                "matplotlib"
            ]
            
            missing_patterns = []
            for pattern in required_patterns:
                if pattern not in generated_code:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                print(f"⚠️  Missing patterns: {missing_patterns}")
            else:
                print("✅ All required patterns found")
            
            # Ensure it's NOT using vision-specific code
            vision_patterns = ["from PIL import Image", "Image.open"]
            found_vision = [p for p in vision_patterns if p in generated_code]
            
            if found_vision:
                print(f"⚠️  Unexpected vision patterns found: {found_vision}")
            else:
                print("✅ No vision-specific code (as expected)")
            
        else:
            print("❌ Failed to generate regular analysis code")

async def main():
    """Main test function."""
    print("=" * 60)
    print("TESTING REGULAR ANALYSIS (POST VISION INTEGRATION)")
    print("=" * 60)
    
    await test_regular_analysis()
    
    print("=" * 60)
    print("REGULAR ANALYSIS TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
