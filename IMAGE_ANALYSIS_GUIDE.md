# Image Analysis Support

This application now supports **image analysis** using Google's Gemini Vision model. You can upload images (JPG, JPEG, PNG) along with your questions to extract data, analyze charts, read text, or perform visual analysis tasks.

## **Supported Image Formats**
- **`.jpg`** - JPEG images
- **`.jpeg`** - JPEG images (alternative extension)  
- **`.png`** - PNG images

## **Example Usage**

### **cURL Request**
```bash
curl "http://localhost:8000/api/" \
  -F "questions.txt=@question.txt" \
  -F "chart.png=@chart.png"
```

### **Question Example** (`question.txt`)
```plaintext
Answer the following questions using the attached image. Respond with a JSON array of strings containing your answers.

1. How many franchises were first released before 1990?
2. Which franchise has sold the most units that was first released after 2000?
3. What is the average number of units sold across all franchises listed?
4. Draw a horizontal bar chart of the top 5 franchises by units sold.  
   Return as a base-64 encoded data URI, "data:image/png;base64,iVBORw0KG..." under 100,000 bytes.
```

## **Vision Capabilities**

The system can:
- **Extract data from charts and graphs**
- **Read text from images** 
- **Identify tables and structured data**
- **Analyze visual elements**
- **Convert image data to structured formats** (CSV, JSON)
- **Create new visualizations** based on extracted data

## **Technical Implementation**

### **Automatic Detection**
When image files are uploaded, the system automatically:
1. Detects image files by extension
2. Switches to vision-enabled Gemini model
3. Provides both text and visual context to the model
4. Generates code that can process both file data and visual information

### **Generated Code Features**
The vision model generates Python code that:
- Uses PIL (Pillow) for image processing when needed
- Extracts visual information directly (no OCR needed - the model can "see")
- Combines visual data with any uploaded text/CSV files
- Handles errors gracefully
- Outputs results in the standard JSON format

### **Library Support**
Vision analysis code has access to:
- **PIL/Pillow** - Image processing
- **pandas/numpy** - Data manipulation
- **matplotlib/seaborn** - Visualization  
- **All standard analysis libraries**

## **Example Generated Code**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import base64
from PIL import Image
from io import BytesIO

def main():
    try:
        # Load and analyze the chart image
        # The vision model can see the image content directly
        
        # Extract data based on visual analysis
        # (The model "sees" the chart and extracts the data)
        franchises = [
            {"name": "Mario", "year": 1981, "sales": 830},
            {"name": "Tetris", "year": 1984, "sales": 520},
            # ... more data extracted from the image
        ]
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(franchises)
        
        # Answer the questions
        pre_1990 = len(df[df['year'] < 1990])
        post_2000 = df[df['year'] > 2000]['name'].iloc[0] if len(df[df['year'] > 2000]) > 0 else "None"
        avg_sales = df['sales'].mean()
        
        # Create visualization
        top_5 = df.nlargest(5, 'sales')
        plt.figure(figsize=(10, 6))
        plt.barh(top_5['name'], top_5['sales'])
        plt.title('Top 5 Franchises by Sales')
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Return results
        results = [
            pre_1990,
            post_2000, 
            avg_sales,
            f"data:image/png;base64,{image_base64}"
        ]
        
        with open('result.json', 'w') as f:
            json.dump(results, f)
            
    except Exception as e:
        error_results = [0, "Error", 0.0, "data:image/png;base64,"]
        with open('result.json', 'w') as f:
            json.dump(error_results, f)

if __name__ == "__main__":
    main()
```

## **Performance Notes**

- **Vision model latency**: Slightly higher than text-only due to image processing
- **Image size limits**: Optimize images for faster processing
- **Combined analysis**: Can process multiple images + text files simultaneously
- **Docker support**: Images are processed in secure, isolated containers

## **Error Handling**

The system gracefully handles:
- **Unsupported image formats** - Returns validation error
- **Corrupted images** - Falls back to text-only analysis
- **Vision model failures** - Provides informative error messages
- **Large images** - Automatically optimizes for processing

This vision capability makes the analysis API significantly more versatile for real-world data analysis tasks involving charts, reports, and visual data sources.
