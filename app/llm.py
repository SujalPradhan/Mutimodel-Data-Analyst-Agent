"""
LLM integration module for generating analysis code using Google's Gemini API.
"""
import os
import json
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .logger import setup_logger, log_llm_interaction, logger
from .utils import analyze_file_structure, get_file_sample_content
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
# Initialize logger (use global logger from logger module)
# logger = setup_logger()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Model configuration
MODEL_NAME = "gemini-1.5-flash"  # Using the free tier model
GENERATION_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

# Safety settings - allow code generation
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

async def generate_analysis_code(
    question: str,
    file_paths: List[Path],
    analysis_type: str,
    sandbox_path: Path,
    request_id: str = "unknown"
) -> Optional[str]:
    """
    Generate Python analysis code using Gemini API.
    
    Args:
        question: Natural language analysis question
        file_paths: List of file paths to analyze
        analysis_type: Type of analysis requested
        sandbox_path: Path to sandbox directory
        request_id: Request ID for logging
        
    Returns:
        Generated Python code or None if generation fails
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None
    
    try:
        # Analyze file structure
        file_analysis = analyze_file_structure(file_paths)
        
        # Create prompt
        prompt = create_analysis_prompt(
            question=question,
            file_analysis=file_analysis,
            file_paths=file_paths,
            analysis_type=analysis_type,
            sandbox_path=sandbox_path
        )
        
        # Initialize model
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS
        )
        
        # Generate response
        response = await asyncio.to_thread(model.generate_content, prompt)
        
        if not response or not response.text:
            logger.error(f"Empty response from Gemini API for request {request_id}")
            return None
        
        # Log interaction
        log_llm_interaction(
            logger=logger,
            request_id=request_id,
            prompt_length=len(prompt),
            response_length=len(response.text),
            model_used=MODEL_NAME
        )
        
        # Extract code from response
        generated_code = extract_code_from_response(response.text)
        
        if not generated_code:
            logger.error(f"Could not extract code from Gemini response for request {request_id}")
            return None
        
        # Validate generated code
        from .utils import validate_generated_code
        is_valid, error_msg = validate_generated_code(generated_code)
        
        if not is_valid:
            logger.error(f"Generated code validation failed for request {request_id}: {error_msg}")
            return None
        
        logger.info(f"Successfully generated analysis code for request {request_id}")
        return generated_code
        
    except Exception as e:
        logger.error(f"Error generating analysis code for request {request_id}: {str(e)}")
        return None

def create_analysis_prompt(
    question: str,
    file_analysis: Dict[str, Any],
    file_paths: List[Path],
    analysis_type: str,
    sandbox_path: Path
) -> str:
    """
    Create a detailed prompt for code generation.
    
    Args:
        question: User's analysis question
        file_analysis: Analysis of uploaded files
        file_paths: List of file paths
        analysis_type: Type of analysis
        sandbox_path: Path to sandbox directory
        
    Returns:
        Formatted prompt string
    """
    # Import here to avoid circular imports
    from .utils import get_all_available_files
    
    # Get all available files including any scraped data
    all_files = get_all_available_files(sandbox_path, file_paths)
    
    # Get sample content from files
    file_samples = {}
    for file_path in all_files[:5]:  # Limit to first 5 files for context
        sample = get_file_sample_content(file_path, max_lines=5)
        file_samples[file_path.name] = sample
    
    # Check if scraped data files exist
    scraped_files = [f for f in all_files if 'scraped_data' in f.name]
    scraped_context = ""
    if scraped_files:
        scraped_context = f"""

**PREVIOUSLY SCRAPED DATA AVAILABLE:**
The following scraped data files are available from previous analysis steps:
{json.dumps([f.name for f in scraped_files], indent=2)}
You can use these files directly in your analysis instead of scraping again.
"""
    
    prompt = f"""
You are a data analysis expert. Generate Python code to answer the user's question about the uploaded data files.

**USER QUESTION:** {question}

**ANALYSIS TYPE:** {analysis_type}

**AVAILABLE FILES:**
{json.dumps(file_analysis, indent=2)}

**FILE SAMPLES:**
{json.dumps(file_samples, indent=2)}{scraped_context}

**REQUIREMENTS:**
1. Write complete, executable Python code
2. Use only these allowed libraries: pandas, numpy, matplotlib, seaborn, plotly, networkx, scipy, json, csv, base64, pathlib, requests, beautifulsoup4
3. If files named 'scraped_data.csv', 'scraped_data.html' or 'scraped_data.json' exist in the working directory, load data directly from these files instead of performing web scraping. Only perform web scraping using requests and BeautifulSoup when no pre-scraped data file is available.
4. Read files from the current directory (files are already in the working directory)
5. Save all results to specific output files:
   - JSON results: save to 'result.json' as a JSON array with exactly 4 elements: [numeric_answer, string_answer, float_answer, base64_image_string]
   - **CRITICAL FORMAT REQUIREMENT:** The result.json must contain RAW VALUES ONLY, no descriptive text
   - Example CORRECT format: [1, "Titanic", 0.49, "data:image/png;base64,iVBORw0KGgo..."]
   - Example WRONG format: ["Number of movies: 1", "Film name is Titanic", "Correlation is 0.49", ...]
   - Elements must be: [int/float (just number), str (just text), float (just number), str (base64 image)]
   - Plots: save as PNG files with base64 encoding under 100KB each
   - Text output: use print() statements
6. **IMPORTANT for web scraping tasks when pre-scraped data is not available:** After scraping data from a website, save the scraped data as a file:
   - Save as CSV if tabular data: 'scraped_data.csv'
   - Save as JSON if structured data: 'scraped_data.json'
   - This allows the data to be reused for further analysis steps
7. Handle errors gracefully with try-except blocks
8. Include comments explaining the analysis approach
9. For web scraping tasks, implement proper error handling and retry logic
10. **HTML ANALYSIS REQUIREMENT:** When performing web scraping, first analyze the entire HTML document structure to understand the page layout. Use soup.prettify()[:2000] or similar to examine the HTML structure, identify all available tables, sections, and data containers before selecting the target elements. This ensures robust element selection and fallback strategies.

**OUTPUT FORMAT:**
Your response should contain only Python code between ```python and ``` markers.

**CODE TEMPLATE:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import json
import base64
from pathlib import Path
from io import BytesIO
import time

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    try:
        # Your analysis code here
        # For web scraping tasks, use requests and BeautifulSoup:
        # response = requests.get(url, headers={{'User-Agent': 'Mozilla/5.0...'}})
        # soup = BeautifulSoup(response.content, 'html.parser')
        
        # STEP 1: Analyze HTML structure when scraping
        # print("HTML Structure Analysis:")
        # print(soup.prettify()[:2000])  # Print first 2000 chars of HTML
        # 
        # # Find all tables and examine their structure
        # all_tables = soup.find_all('table')
        # print(f"Found {{len(all_tables)}} tables on the page")
        # for i, table in enumerate(all_tables):
        #     table_classes = table.get('class', [])
        #     print(f"Table {{i}}: classes={{table_classes}}")
        #     # Check for captions, headers, etc.
        #     caption = table.find('caption')
        #     if caption:
        #         print(f"  Caption: {{caption.get_text(strip=True)[:100]}}...")
        
        # IMPORTANT: If you scrape data, save it to a file for future use:
        # For tabular data: df.to_csv('scraped_data.csv', index=False)
        # For structured data: 
        # with open('scraped_data.json', 'w') as f:
        #     json.dump(scraped_data, f, cls=NumpyEncoder)
        
        # **CRITICAL: Create result array with exactly 4 elements in correct format**
        # This must be a JSON array with: [numeric_answer, string_answer, float_answer, base64_image_string]
        # Use ONLY raw values - NO descriptive text or labels
        # 
        # Create a simple plot to generate the required base64 image
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.title('Sample Plot')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        
        # Convert plot to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Create result array - REPLACE WITH YOUR ACTUAL VALUES
        results = [
            0,  # Replace with your numeric answer (JUST THE NUMBER)
            "Template Answer",  # Replace with your string answer (JUST THE TEXT, NO LABELS)
            0.0,  # Replace with your float answer (JUST THE NUMBER)
            f"data:image/png;base64,{{image_base64}}"  # Base64 image (keep this format)
        ]
        
        # Save results using custom encoder
        with open('result.json', 'w') as f:
            json.dump(results, f, cls=NumpyEncoder)
            
    except Exception as e:
        # Error case - still return 4-element array format
        error_results = [0, "Error", 0.0, "data:image/png;base64,"]
        with open('result.json', 'w') as f:
            json.dump(error_results, f)
        print(f"Error: {{e}}")

if __name__ == "__main__":
    main()
```

**SPECIFIC GUIDANCE BY ANALYSIS TYPE:**

{get_analysis_type_guidance(analysis_type)}

Generate the Python code now:
"""
    
    return prompt

def get_analysis_type_guidance(analysis_type: str) -> str:
    """Get specific guidance based on analysis type."""
    guidance = {
        "statistical": """
- Calculate descriptive statistics (mean, median, std, etc.)
- Perform correlation analysis if multiple numeric columns
- Create histograms and box plots
- Include statistical tests if appropriate
        """,
        
        "network": """
- Use networkx for graph analysis
- Calculate network metrics (degree, centrality, clustering)
- Create network visualizations
- Identify communities or important nodes
        """,
        
        "timeseries": """
- Parse datetime columns properly
- Create time series plots
- Calculate trends and seasonality
- Use rolling averages and statistics
        """,
        
        "ml": """
- Perform data preprocessing (scaling, encoding)
- Apply appropriate ML techniques (clustering, classification, etc.)
- Create feature importance plots
- Calculate performance metrics
        """,
        
        "general": """
- Explore data structure and quality
- Create appropriate visualizations
- Identify patterns and insights
- Provide summary statistics
- **For web scraping tasks:** First analyze the complete HTML structure using soup.prettify() to understand page layout, then identify all available data containers (tables, divs, etc.) before selecting target elements. Implement robust fallback strategies in case primary selectors fail.
        """
    }
    
    return guidance.get(analysis_type, guidance["general"])

def extract_code_from_response(response_text: str) -> Optional[str]:
    """
    Extract Python code from the LLM response.
    
    Args:
        response_text: Raw response from the LLM
        
    Returns:
        Extracted Python code or None if not found
    """
    # Look for code blocks
    import re
    
    # Try to find Python code blocks
    python_pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(python_pattern, response_text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Try generic code blocks
    code_pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(code_pattern, response_text, re.DOTALL)
    
    if matches:
        # Check if it looks like Python code
        code = matches[0].strip()
        if any(keyword in code for keyword in ['import ', 'def ', 'if __name__']):
            return code
    
    # If no code blocks found, try to extract code-like content
    lines = response_text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if any(keyword in line for keyword in ['import ', 'def ', 'try:', 'if ', 'for ', 'while ']):
            in_code = True
        
        if in_code:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return None





def get_model_info() -> Dict[str, Any]:
    """
    Get information about the configured model.
    
    Returns:
        Dictionary with model information
    """
    return {
        "model_name": MODEL_NAME,
        "api_key_configured": bool(GEMINI_API_KEY),
        "generation_config": GENERATION_CONFIG
    }

async def generate_code_from_prompt(question_text: str, file_list: list[str]) -> str:
    """
    Generate Python code from a natural language prompt using Gemini Pro API.
    
    Args:
        question_text: Natural language task description
        file_list: List of uploaded file names
        
    Returns:
        Generated Python code as a string
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set")
        return ""
    
    try:
        # Construct the prompt
        prompt = _construct_code_generation_prompt(question_text, file_list)
        
        # Initialize model
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS
        )
        
        # Generate response
        response = await asyncio.to_thread(model.generate_content, prompt)
        
        if not response or not response.text:
            logger.error("Empty response from Gemini API")
            return ""
        
        # Extract Python code from response
        generated_code = extract_code_from_response(response.text)
        
        if not generated_code:
            logger.error("Could not extract Python code from response")
            return ""
        
        logger.info("Successfully generated Python code from prompt")
        return generated_code
        
    except Exception as e:
        logger.error(f"Error generating code from prompt: {str(e)}")
        return ""

def _construct_code_generation_prompt(question_text: str, file_list: list[str]) -> str:
    """
    Construct a prompt for code generation.
    
    Args:
        question_text: Natural language task description
        file_list: List of uploaded file names
        
    Returns:
        Formatted prompt string
    """
    file_list_str = "\n".join([f"- {filename}" for filename in file_list])
    
    prompt = f"""
You are a Python code generation expert. Generate Python code to solve the following task.

**TASK:** {question_text}

**AVAILABLE FILES:**
{file_list_str}

**INSTRUCTIONS:**
1. Write complete, executable Python code
2. Use only standard libraries and common data science libraries (pandas, numpy, matplotlib, seaborn, plotly, scipy, networkx)
3. Read files from the current directory using the filenames provided above
4. Include proper error handling with try-except blocks
5. Save results to appropriate output files (JSON, CSV, or images)
6. Add comments to explain the approach

**IMPORTANT:** Return ONLY Python code. Do not include explanations, markdown formatting, or any other text. The response should be valid Python code that can be executed directly.

Generate the Python code now:
"""
    
    return prompt