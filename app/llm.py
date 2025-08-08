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
You are an expert data analyst. Generate Python code to answer the user's question using the provided data files.

**USER QUESTION:** {question}

**ANALYSIS TYPE:** {analysis_type}

**AVAILABLE FILES:**
{json.dumps(file_analysis, indent=2)}

**FILE SAMPLES:**
{json.dumps(file_samples, indent=2)}{scraped_context}

**CORE REQUIREMENTS:**

**1. Data Source Strategy:**
   - **SQL/DuckDB:** For questions with SQL queries, s3://, database URLs, .parquet/.csv remote files
   - **Web Scraping:** Only for explicit web content requests (Wikipedia pages, HTML tables)
   - **Local Files:** Default for uploaded files in current directory

**2. DuckDB Database Operations:**
   ```python
   import duckdb
   conn = duckdb.connect()
   # Install required extensions
   conn.execute("INSTALL httpfs; LOAD httpfs;")  # For remote URLs
   conn.execute("INSTALL parquet; LOAD parquet;")  # For parquet files
   
   # Query patterns:
   result = conn.execute("SELECT * FROM read_parquet('s3://bucket/file.parquet')").fetchdf()
   result = conn.execute("SELECT * FROM read_csv('https://example.com/data.csv')").fetchdf()
   conn.close()
   ```
   
   **DuckDB Function Reference:**
   - ✅ Date differences: `DATE_DIFF('day', date1, date2)` or `DATEDIFF('day', date1, date2)`
   - ✅ Date arithmetic: `date1 - date2` (for DATE columns)
   - ✅ Date casting: `column::DATE` or `CAST(column AS DATE)`
   - ✅ String aggregation: `STRING_AGG(column, ',')`
   - ❌ Never use SQLite functions: `JULIANDAY()`, `GROUP_CONCAT()`
   - ❌ Never use MySQL/PostgreSQL specific syntax

**3. Output Format (MANDATORY):**
   Save exactly 4 elements to 'result.json':
   ```json
   [numeric_value, "string_value", float_value, "data:image/png;base64,encoded_image"]
   ```
   
   **Examples:**
   - ✅ CORRECT: `[42, "Product A", 0.85, "data:image/png;base64,iVBOR..."]`
   - ❌ WRONG: `["Count: 42", "Top product: Product A", "Correlation: 0.85"]`
   
   **Element Guidelines:**
   - [0]: Raw numeric answer (count, sum, ID, etc.)
   - [1]: String answer (name, category, description)
   - [2]: Calculated metric (average, correlation, percentage as decimal)
   - [3]: Visualization as base64 PNG with full data URI prefix

**4. Analysis Methodology:**
   ```python
   # 1. Data Discovery
   print("Data shape and types:")
   print(df.info())
   print("\nFirst few rows:")
   print(df.head())
   
   # 2. Data Quality Check
   print("\nMissing values:")
   print(df.isnull().sum())
   print("\nDuplicates:", df.duplicated().sum())
   
   # 3. Analysis Execution
   # Apply appropriate methods based on question
   
   # 4. Visualization Creation
   # Create relevant plots for the analysis
   
   # 5. Results Compilation
   # Format as 4-element array
   ```

**5. Error Handling & Robustness:**
   - Wrap file operations in try-except blocks
   - Validate data types and handle conversion errors
   - Check for empty datasets before analysis
   - Provide fallback values if calculations fail
   - Use `df.fillna()` or `df.dropna()` for missing data
   - Handle edge cases (single row, all same values, etc.)

**OUTPUT FORMAT:**
Your response should contain only Python code between ```python and ``` markers.

**STRUCTURED CODE TEMPLATE:**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import base64
from pathlib import Path
from io import BytesIO
import duckdb
import requests
from bs4 import BeautifulSoup

# Configure matplotlib
plt.switch_backend('Agg')
plt.style.use('default')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def create_visualization(data, title="Analysis Results"):
    # Create appropriate visualization based on data type and analysis
    plt.figure(figsize=(10, 6))
    
    try:
        if hasattr(data, 'columns') and len(data.columns) >= 2:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                # Multiple numeric columns - scatter or correlation
                sns.scatterplot(data=data, x=numeric_cols[0], y=numeric_cols[1])
                plt.title(f"{{title}}: {{numeric_cols[0]}} vs {{numeric_cols[1]}}")
            elif len(numeric_cols) == 1:
                # Single numeric column - histogram
                data[numeric_cols[0]].hist(bins=20, alpha=0.7)
                plt.title(f"{{title}}: Distribution of {{numeric_cols[0]}}")
            else:
                # Categorical data - bar chart
                value_counts = data.iloc[:, 0].value_counts().head(10)
                value_counts.plot(kind='bar')
                plt.title(f"{{title}}: Count by Category")
        else:
            # Simple data - basic plot
            if hasattr(data, 'plot'):
                data.plot(kind='bar' if len(data) <= 20 else 'line')
            else:
                plt.bar(range(len(data)), data)
            plt.title(title)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
    except Exception:
        # Fallback simple plot
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.title(title)
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{{image_base64}}"

def safe_convert(value, target_type=float, default=0):
    # Safely convert values with error handling
    try:
        if pd.isna(value):
            return default
        return target_type(value)
    except (ValueError, TypeError):
        return default

def main():
    try:
        question = "{question}"
        question_lower = question.lower()
        
        # STEP 1: DETERMINE DATA SOURCE STRATEGY
        sql_keywords = ['select', 'from', 'where', 's3://', 'read_parquet', 'read_csv', 
                       'duckdb', 'install httpfs', '.parquet', 'group by', 'order by']
        web_keywords = ['wikipedia', 'scrape', 'website', 'html', 'web page', 'url']
        
        is_sql_query = any(keyword in question_lower for keyword in sql_keywords)
        is_web_scraping = any(keyword in question_lower for keyword in web_keywords) and not is_sql_query
        
        print(f"Analysis strategy: SQL={{is_sql_query}}, Web={{is_web_scraping}}")
        
        # STEP 2: DATA ACQUISITION
        if is_sql_query:
            # DuckDB approach for SQL queries and remote data
            print("Using DuckDB for data access...")
            conn = duckdb.connect()
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            conn.execute("INSTALL parquet; LOAD parquet;")
            
            # Execute SQL query extracted from question
            # Replace this with actual SQL from the question:
            # df = conn.execute("YOUR_SQL_QUERY_HERE").fetchdf()
            
            # For template demonstration:
            df = pd.DataFrame({{'example': [1, 2, 3], 'data': ['A', 'B', 'C']}})
            conn.close()
            
        elif is_web_scraping:
            # Web scraping approach
            print("Using web scraping...")
            headers = {{'User-Agent': 'Mozilla/5.0 (compatible; DataAnalyzer/1.0)'}}
            
            # Extract URL from question and scrape
            # url = "https://example.com"  # Extract from question
            # response = requests.get(url, headers=headers)
            # soup = BeautifulSoup(response.content, 'html.parser')
            
            # For template:
            df = pd.DataFrame({{'scraped': [1, 2, 3], 'data': ['X', 'Y', 'Z']}})
            
        else:
            # Local file approach
            print("Loading local data files...")
            data_files = list(Path('.').glob('*.csv')) + list(Path('.').glob('*.json'))
            
            if data_files:
                primary_file = data_files[0]
                if primary_file.suffix == '.csv':
                    df = pd.read_csv(primary_file)
                elif primary_file.suffix == '.json':
                    df = pd.read_json(primary_file)
                else:
                    df = pd.read_csv(primary_file)
            else:
                # Fallback data
                df = pd.DataFrame({{'sample': [1, 2, 3, 4, 5], 'values': [10, 20, 15, 25, 30]}})
        
        # STEP 3: DATA EXPLORATION
        print(f"Data shape: {{df.shape}}")
        print(f"Columns: {{list(df.columns)}}")
        print("\\nFirst few rows:")
        print(df.head())
        
        # Data quality check
        missing_count = df.isnull().sum().sum()
        duplicate_count = df.duplicated().sum()
        print(f"\\nData quality: {{missing_count}} missing, {{duplicate_count}} duplicates")
        
        # STEP 4: ANALYSIS EXECUTION (Customize based on question)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        text_cols = df.select_dtypes(include=['object']).columns
        
        # Default analysis patterns:
        if 'count' in question_lower or 'how many' in question_lower:
            numeric_result = len(df)
            string_result = "Total records" if len(df.columns) == 0 else str(df.columns[0])
            float_result = float(numeric_result)
            
        elif ('top' in question_lower or 'maximum' in question_lower or 'highest' in question_lower) and len(numeric_cols) > 0:
            max_col = numeric_cols[0]
            max_idx = df[max_col].idxmax()
            max_row = df.loc[max_idx]
            
            numeric_result = safe_convert(max_row[max_col], int, 0)
            string_result = str(max_row.iloc[0]) if len(text_cols) > 0 else max_col
            float_result = safe_convert(max_row[max_col], float, 0.0)
            
        elif 'correlation' in question_lower and len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            corr_value = corr_matrix.iloc[0, 1]
            
            numeric_result = len(numeric_cols)
            string_result = f"{{numeric_cols[0]}} vs {{numeric_cols[1]}}"
            float_result = safe_convert(corr_value, float, 0.0)
            
        elif ('average' in question_lower or 'mean' in question_lower) and len(numeric_cols) > 0:
            avg_value = df[numeric_cols[0]].mean()
            
            numeric_result = len(df)
            string_result = numeric_cols[0]
            float_result = safe_convert(avg_value, float, 0.0)
            
        else:
            # General descriptive analysis
            numeric_result = len(df)
            string_result = str(df.columns[0]) if len(df.columns) > 0 else "No data"
            float_result = safe_convert(df[numeric_cols[0]].mean() if len(numeric_cols) > 0 else 0, float, 0.0)
        
        # STEP 5: VISUALIZATION
        print("\\nCreating visualization...")
        image_base64 = create_visualization(df, "Analysis Results")
        
        # STEP 6: RESULTS COMPILATION
        final_results = [
            safe_convert(numeric_result, int, 0),
            str(string_result),
            safe_convert(float_result, float, 0.0),
            image_base64
        ]
        
        print(f"Results: [{{final_results[0]}}, '{{final_results[1]}}', {{final_results[2]}}, '<image>']")
        
        # Save results
        with open('result.json', 'w') as f:
            json.dump(final_results, f, cls=NumpyEncoder)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {{e}}")
        import traceback
        traceback.print_exc()
        
        # Error fallback
        error_results = [0, f"Error: {{str(e)}}", 0.0, "data:image/png;base64,"]
        with open('result.json', 'w') as f:
            json.dump(error_results, f)

if __name__ == "__main__":
    main()
```

**SPECIFIC GUIDANCE BY ANALYSIS TYPE:**

{get_analysis_type_guidance(analysis_type)}

Generate the Python code now:
"""

    # Enhance the prompt with context-specific examples and guidance
    enhanced_prompt = enhance_prompt_with_context(prompt, question, analysis_type)
    
    return enhanced_prompt

def get_analysis_type_guidance(analysis_type: str) -> str:
    """Get specific guidance based on analysis type."""
    guidance = {
        "statistical": """
**STATISTICAL ANALYSIS FOCUS:**
- Calculate comprehensive descriptive statistics (mean, median, std, quartiles, skewness, kurtosis)
- Perform correlation analysis for numeric variables (use df.corr())
- Create distribution plots (histograms, box plots, violin plots)
- Apply statistical tests when appropriate (t-tests, chi-square, normality tests)
- Check for outliers using IQR method or z-scores
- Generate summary statistics tables and interpretation

**Key outputs:** Count of variables, strongest correlation coefficient, primary distribution characteristic
        """,
        
        "network": """
**NETWORK ANALYSIS FOCUS:**
- Use networkx library for graph construction and analysis
- Calculate key network metrics: degree centrality, betweenness centrality, clustering coefficient
- Identify communities using community detection algorithms
- Create network visualizations with node sizing based on importance
- Analyze network properties: diameter, density, average path length
- Find influential nodes and network components

**Key outputs:** Number of nodes/edges, most central node, average clustering coefficient
        """,
        
        "timeseries": """
**TIME SERIES ANALYSIS FOCUS:**
- Parse datetime columns using pd.to_datetime() with appropriate format
- Set datetime as index for time series operations
- Calculate trends using rolling averages and linear regression
- Identify seasonality patterns using decomposition
- Create time series plots with trend lines and seasonal components
- Calculate year-over-year or period-over-period growth rates
- Handle missing time periods and irregular intervals

**Key outputs:** Total time periods, strongest trend direction, peak/trough values
        """,
        
        "ml": """
**MACHINE LEARNING ANALYSIS FOCUS:**
- Perform comprehensive data preprocessing (scaling, encoding, feature selection)
- Apply appropriate ML techniques: clustering (KMeans), classification, regression
- Create feature importance plots and correlation matrices
- Calculate and report performance metrics (accuracy, precision, recall, R²)
- Use cross-validation for robust model evaluation
- Visualize decision boundaries or cluster separations
- Handle categorical variables with proper encoding

**Key outputs:** Number of features, best model performance score, most important feature
        """,
        
        "database": """
**DATABASE ANALYSIS FOCUS:**
- Use DuckDB for SQL query execution on local and remote data sources
- Install and configure necessary extensions (httpfs for S3/HTTP, parquet for .parquet files)
- Execute complex SQL queries with proper JOIN, GROUP BY, and aggregation functions
- Handle large datasets efficiently using DuckDB's columnar processing
- Convert query results to pandas DataFrames for further analysis
- Implement proper error handling for database connections and queries
- Use DuckDB-specific functions and avoid SQLite/MySQL syntax differences

**Key outputs:** Query result count, aggregated metric, database connection status
        """,
        
        "general": """
**GENERAL DATA ANALYSIS FOCUS:**
- **Data Source Strategy:**
  - **SQL/DuckDB:** Use for database queries, S3/HTTP URLs, .parquet/.csv remote files
  - **Web Scraping:** Only for explicit HTML content extraction (Wikipedia, web tables)
  - **Local Files:** Default approach for uploaded CSV/JSON/Parquet files
  
- **Analysis Pipeline:**
  1. Data discovery and quality assessment
  2. Exploratory data analysis with appropriate visualizations
  3. Pattern identification and statistical insights
  4. Meaningful summary and interpretation
  
- **Robust Implementation:**
  - Handle missing data with appropriate strategies (imputation, removal)
  - Validate data types and perform necessary conversions
  - Create informative visualizations that match the data characteristics
  - Provide clear, actionable insights based on the analysis
  - Ensure output format compliance (4-element JSON array)

**Key outputs:** Dataset size, primary insight, calculated metric
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

def get_duckdb_examples_for_question(question: str) -> str:
    """
    Generate relevant DuckDB examples based on question content.
    
    Args:
        question: User's question text
        
    Returns:
        String with relevant DuckDB code examples
    """
    question_lower = question.lower()
    examples = []
    
    if 's3://' in question_lower:
        examples.append("""
# S3 data access example:
conn.execute("INSTALL httpfs; LOAD httpfs;")
df = conn.execute("SELECT * FROM read_parquet('s3://bucket/path/file.parquet?s3_region=us-east-1')").fetchdf()
""")
    
    if 'https://' in question_lower and ('.csv' in question_lower or '.parquet' in question_lower):
        examples.append("""
# HTTPS data access example:
conn.execute("INSTALL httpfs; LOAD httpfs;")
df = conn.execute("SELECT * FROM read_csv('https://example.com/data.csv')").fetchdf()
""")
    
    if 'date' in question_lower or 'time' in question_lower:
        examples.append("""
# Date handling examples:
# Use DATE_DIFF for date differences:
SELECT DATE_DIFF('day', start_date, end_date) as days_between FROM table
# Cast strings to dates:
SELECT column::DATE as date_col FROM table WHERE date_col > '2023-01-01'::DATE
""")
    
    if 'count' in question_lower or 'group' in question_lower:
        examples.append("""
# Aggregation examples:
SELECT category, COUNT(*) as count, AVG(value) as avg_value 
FROM table GROUP BY category ORDER BY count DESC
""")
    
    return '\n'.join(examples) if examples else ""

def validate_output_format_requirements(question: str, analysis_type: str) -> str:
    """
    Generate specific output format guidance based on question and analysis type.
    
    Args:
        question: User's question text
        analysis_type: Type of analysis being performed
        
    Returns:
        Specific guidance for output format
    """
    question_lower = question.lower()
    
    numeric_guidance = "Element 1 (numeric): "
    string_guidance = "Element 2 (string): "
    float_guidance = "Element 3 (float): "
    
    # Customize based on question type
    if 'count' in question_lower or 'how many' in question_lower:
        numeric_guidance += "Total count (integer)"
        string_guidance += "Name of what was counted"
        float_guidance += "Count as decimal (same as element 1)"
    elif 'top' in question_lower or 'best' in question_lower or 'maximum' in question_lower:
        numeric_guidance += "Rank or position (1, 2, 3...)"
        string_guidance += "Name/identifier of top item"
        float_guidance += "Value/score of top item"
    elif 'average' in question_lower or 'mean' in question_lower:
        numeric_guidance += "Sample size or data points"
        string_guidance += "Variable name being averaged"
        float_guidance += "Calculated average value"
    elif 'correlation' in question_lower or 'relationship' in question_lower:
        numeric_guidance += "Number of variables analyzed"
        string_guidance += "Names of correlated variables (e.g., 'A vs B')"
        float_guidance += "Correlation coefficient (-1.0 to 1.0)"
    else:
        # General guidance
        numeric_guidance += "Primary count/ID/rank (integer)"
        string_guidance += "Main categorical result/name"
        float_guidance += "Key calculated metric (ratio/average/score)"
    
    return f"""
**OUTPUT FORMAT SPECIFICATION:**
{numeric_guidance}
{string_guidance}
{float_guidance}
Element 4 (image): Base64 PNG with prefix "data:image/png;base64,"

**EXAMPLES BY QUESTION TYPE:**
- Count question: [247, "products", 247.0, "data:image/png;base64,..."]
- Top item question: [1, "iPhone 14", 999.99, "data:image/png;base64,..."]
- Average question: [1000, "price", 45.67, "data:image/png;base64,..."]
- Correlation question: [2, "price vs sales", 0.85, "data:image/png;base64,..."]
"""

def enhance_prompt_with_context(base_prompt: str, question: str, analysis_type: str) -> str:
    """
    Enhance the base prompt with context-specific examples and guidance.
    
    Args:
        base_prompt: Base prompt template
        question: User's question
        analysis_type: Type of analysis
        
    Returns:
        Enhanced prompt with specific examples
    """
    # Add DuckDB examples if relevant
    duckdb_examples = get_duckdb_examples_for_question(question)
    if duckdb_examples:
        duckdb_section = f"\n**RELEVANT DUCKDB EXAMPLES:**{duckdb_examples}\n"
        base_prompt = base_prompt.replace("**5. Error Handling", duckdb_section + "**5. Error Handling")
    
    # Add specific output format guidance
    output_guidance = validate_output_format_requirements(question, analysis_type)
    base_prompt = base_prompt.replace("**STRUCTURED CODE TEMPLATE:**", 
                                      output_guidance + "\n**STRUCTURED CODE TEMPLATE:**")
    
    # Add question-specific hints in the code template
    question_hint = f"# QUESTION ANALYSIS: {question}\n# Focus on: "
    
    if 'count' in question.lower():
        question_hint += "counting and aggregation"
    elif 'top' in question.lower() or 'best' in question.lower():
        question_hint += "finding maximum/top values"
    elif 'correlation' in question.lower():
        question_hint += "calculating relationships between variables"
    elif 'trend' in question.lower() or 'change' in question.lower():
        question_hint += "analyzing trends over time"
    else:
        question_hint += "comprehensive data analysis"
    
    base_prompt = base_prompt.replace("def main():", f"def main():\n    {question_hint}")
    
    return base_prompt