"""
LLM integration module for generating analysis code using Google's Gemini API.
"""
import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from logger import setup_logger, log_llm_interaction
from utils import analyze_file_structure, get_file_sample_content

# Initialize logger
logger = setup_logger()

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
        from utils import validate_generated_code
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
    # Get sample content from files
    file_samples = {}
    for file_path in file_paths[:3]:  # Limit to first 3 files for context
        sample = get_file_sample_content(file_path, max_lines=5)
        file_samples[file_path.name] = sample
    
    prompt = f"""
You are a data analysis expert. Generate Python code to answer the user's question about the uploaded data files.

**USER QUESTION:** {question}

**ANALYSIS TYPE:** {analysis_type}

**AVAILABLE FILES:**
{json.dumps(file_analysis, indent=2)}

**FILE SAMPLES:**
{json.dumps(file_samples, indent=2)}

**REQUIREMENTS:**
1. Write complete, executable Python code
2. Use only these allowed libraries: pandas, numpy, matplotlib, seaborn, plotly, networkx, scipy, json, csv, base64, pathlib
3. Read files from the current directory (files are already in the working directory)
4. Save all results to specific output files:
   - JSON results: save to 'result.json'
   - Plots: save as PNG files with base64 encoding under 100KB each
   - Text output: use print() statements
5. Handle errors gracefully with try-except blocks
6. Include comments explaining the analysis approach

**OUTPUT FORMAT:**
Your response should contain only Python code between ```python and ``` markers.

**CODE TEMPLATE:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import base64
from pathlib import Path
from io import BytesIO

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

def main():
    try:
        # Your analysis code here
        results = {{"status": "success", "message": "Analysis completed"}}
        
        # Save results
        with open('result.json', 'w') as f:
            json.dump(results, f)
            
    except Exception as e:
        error_results = {{"status": "error", "error": str(e)}}
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

def create_fallback_code(question: str, file_paths: List[Path]) -> str:
    """
    Create fallback analysis code when LLM generation fails.
    
    Args:
        question: User's question
        file_paths: List of file paths
        
    Returns:
        Basic fallback Python code
    """
    return f'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

plt.switch_backend('Agg')

def main():
    try:
        results = {{"status": "fallback", "message": "Using basic analysis"}}
        
        # Find CSV files
        csv_files = [f for f in {[str(p) for p in file_paths]} if f.endswith('.csv')]
        
        if csv_files:
            # Analyze first CSV file
            df = pd.read_csv(csv_files[0])
            
            results["file_info"] = {{
                "filename": csv_files[0],
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict()
            }}
            
            # Basic statistics
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                results["statistics"] = df.describe().to_dict()
            
            # Create a simple plot
            plt.figure(figsize=(8, 6))
            if len(df.columns) >= 2:
                df.plot(kind='scatter', x=df.columns[0], y=df.columns[1])
            else:
                df[df.columns[0]].hist()
            plt.title("Basic Data Visualization")
            plt.savefig("basic_plot.png", dpi=100, bbox_inches='tight')
            plt.close()
            
        results["question"] = "{question}"
        
        with open('result.json', 'w') as f:
            json.dump(results, f)
            
    except Exception as e:
        error_results = {{"status": "error", "error": str(e)}}
        with open('result.json', 'w') as f:
            json.dump(error_results, f)
        print(f"Error: {{e}}")

if __name__ == "__main__":
    main()
'''

async def test_gemini_connection() -> bool:
    """
    Test connection to Gemini API.
    
    Returns:
        True if connection successful, False otherwise
    """
    if not GEMINI_API_KEY:
        return False
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = await asyncio.to_thread(
            model.generate_content,
            "Generate a simple Python print statement that says 'Hello, World!'"
        )
        return bool(response and response.text)
    except Exception:
        return False

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