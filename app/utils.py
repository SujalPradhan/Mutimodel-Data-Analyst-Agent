"""
Utility functions for file processing and data handling.
"""
import os
import json
import csv
import zipfile
import tempfile
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from fastapi import UploadFile
import aiofiles
import shutil

# Supported file types and their extensions
SUPPORTED_EXTENSIONS = {
    '.csv', '.json', '.txt', '.html', '.htm', '.xml', 
    '.zip', '.xlsx', '.xls', '.tsv', '.md', '.log'
}

SUPPORTED_MIME_TYPES = {
    'text/csv', 'application/json', 'text/plain', 'text/html', 
    'application/xml', 'text/xml', 'application/zip',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel', 'text/tab-separated-values',
    'text/markdown', 'application/octet-stream'
}

def validate_file_type(filename: str) -> bool:
    """
    Validate if the uploaded file type is supported.
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        True if file type is supported, False otherwise
    """
    if not filename:
        return False
    
    # Check extension
    extension = Path(filename).suffix.lower()
    if extension in SUPPORTED_EXTENSIONS:
        return True
    
    # Check MIME type
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type in SUPPORTED_MIME_TYPES:
        return True
    
    return False

def create_sandbox_directory(request_id: str) -> Path:
    """
    Create a sandbox directory for a specific request.
    
    Args:
        request_id: Unique request identifier
        
    Returns:
        Path to the created sandbox directory
    """
    sandbox_path = Path("sandbox") / request_id
    sandbox_path.mkdir(parents=True, exist_ok=True)
    return sandbox_path

async def save_uploaded_files(files: List[UploadFile], sandbox_path: Path) -> List[Path]:
    """
    Save uploaded files to the sandbox directory.
    
    Args:
        files: List of uploaded files
        sandbox_path: Path to sandbox directory
        
    Returns:
        List of paths to saved files
    """
    saved_files = []
    
    for i, file in enumerate(files):
        # Generate safe filename
        safe_filename = sanitize_filename(file.filename or f"file_{i}")
        file_path = sandbox_path / safe_filename
        
        # Save file asynchronously
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        saved_files.append(file_path)
        
        # Handle ZIP files by extracting them
        if file_path.suffix.lower() == '.zip':
            extracted_files = extract_zip_file(file_path, sandbox_path)
            saved_files.extend(extracted_files)
    
    return saved_files

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove directory separators and invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename

def extract_zip_file(zip_path: Path, extract_to: Path) -> List[Path]:
    """
    Extract ZIP file contents to the specified directory.
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract files to
        
    Returns:
        List of extracted file paths
    """
    extracted_files = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                # Skip directories and hidden files
                if member.endswith('/') or member.startswith('.'):
                    continue
                
                # Sanitize member name to prevent path traversal
                safe_name = sanitize_filename(os.path.basename(member))
                if not safe_name:
                    continue
                
                # Extract file
                extract_path = extract_to / f"extracted_{safe_name}"
                with zip_ref.open(member) as source, open(extract_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
                
                extracted_files.append(extract_path)
                
    except zipfile.BadZipFile:
        pass  # Ignore bad zip files
    except Exception:
        pass  # Ignore other extraction errors
    
    return extracted_files

def analyze_file_structure(file_paths: List[Path]) -> Dict[str, Any]:
    """
    Analyze the structure and content of uploaded files.
    
    Args:
        file_paths: List of file paths to analyze
        
    Returns:
        Dictionary containing file analysis results
    """
    analysis = {
        'total_files': len(file_paths),
        'file_types': {},
        'files': []
    }
    
    for file_path in file_paths:
        file_info = analyze_single_file(file_path)
        analysis['files'].append(file_info)
        
        # Count file types
        file_type = file_info['type']
        analysis['file_types'][file_type] = analysis['file_types'].get(file_type, 0) + 1
    
    return analysis

def analyze_single_file(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a single file and extract metadata.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file information
    """
    file_info = {
        'name': file_path.name,
        'path': str(file_path),
        'size': file_path.stat().st_size if file_path.exists() else 0,
        'extension': file_path.suffix.lower(),
        'type': 'unknown',
        'encoding': 'unknown',
        'structure': {}
    }
    
    try:
        # Determine file type and analyze structure
        if file_path.suffix.lower() == '.csv':
            file_info.update(analyze_csv_file(file_path))
        elif file_path.suffix.lower() == '.json':
            file_info.update(analyze_json_file(file_path))
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            file_info.update(analyze_excel_file(file_path))
        elif file_path.suffix.lower() in ['.html', '.htm']:
            file_info.update(analyze_html_file(file_path))
        elif file_path.suffix.lower() == '.txt':
            file_info.update(analyze_text_file(file_path))
        else:
            file_info['type'] = 'text'
            
    except Exception as e:
        file_info['error'] = str(e)
    
    return file_info

def analyze_csv_file(file_path: Path) -> Dict[str, Any]:
    """Analyze CSV file structure."""
    try:
        # Try to read with pandas to get basic info
        df = pd.read_csv(file_path, nrows=5)  # Read only first 5 rows for analysis
        
        return {
            'type': 'csv',
            'structure': {
                'columns': list(df.columns),
                'num_columns': len(df.columns),
                'estimated_rows': len(df),  # This is just the sample
                'dtypes': df.dtypes.astype(str).to_dict(),
                'sample_data': df.head(2).to_dict('records')
            }
        }
    except Exception:
        return {'type': 'csv', 'structure': {'error': 'Could not parse CSV'}}

def analyze_json_file(file_path: Path) -> Dict[str, Any]:
    """Analyze JSON file structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        structure = {
            'data_type': type(data).__name__,
        }
        
        if isinstance(data, list):
            structure['length'] = len(data)
            if data:
                structure['item_type'] = type(data[0]).__name__
                if isinstance(data[0], dict):
                    structure['keys'] = list(data[0].keys())
        elif isinstance(data, dict):
            structure['keys'] = list(data.keys())
        
        return {
            'type': 'json',
            'structure': structure
        }
    except Exception:
        return {'type': 'json', 'structure': {'error': 'Could not parse JSON'}}

def analyze_excel_file(file_path: Path) -> Dict[str, Any]:
    """Analyze Excel file structure."""
    try:
        # Get sheet names
        excel_file = pd.ExcelFile(file_path)
        sheets = excel_file.sheet_names
        
        structure = {
            'sheets': sheets,
            'num_sheets': len(sheets)
        }
        
        # Analyze first sheet
        if sheets:
            df = pd.read_excel(file_path, sheet_name=sheets[0], nrows=5)
            structure['first_sheet'] = {
                'columns': list(df.columns),
                'num_columns': len(df.columns)
            }
        
        return {
            'type': 'excel',
            'structure': structure
        }
    except Exception:
        return {'type': 'excel', 'structure': {'error': 'Could not parse Excel'}}

def analyze_html_file(file_path: Path) -> Dict[str, Any]:
    """Analyze HTML file structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic HTML analysis (can be enhanced with BeautifulSoup)
        structure = {
            'size_chars': len(content),
            'has_tables': '<table' in content.lower(),
            'has_forms': '<form' in content.lower(),
            'has_scripts': '<script' in content.lower()
        }
        
        return {
            'type': 'html',
            'structure': structure
        }
    except Exception:
        return {'type': 'html', 'structure': {'error': 'Could not parse HTML'}}

def analyze_text_file(file_path: Path) -> Dict[str, Any]:
    """Analyze text file structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        structure = {
            'size_chars': len(content),
            'num_lines': len(lines),
            'encoding': 'utf-8'
        }
        
        return {
            'type': 'text',
            'structure': structure
        }
    except Exception:
        return {'type': 'text', 'structure': {'error': 'Could not parse text file'}}

def create_code_file(code: str, sandbox_path: Path) -> Path:
    """
    Create a Python file containing the generated code.
    
    Args:
        code: Python code to save
        sandbox_path: Path to sandbox directory
        
    Returns:
        Path to the created Python file
    """
    code_file = sandbox_path / "analysis.py"
    
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write(code)
    
    return code_file

def read_execution_results(sandbox_path: Path) -> Dict[str, Any]:
    """
    Read execution results from the sandbox directory.
    
    Args:
        sandbox_path: Path to sandbox directory
        
    Returns:
        Dictionary containing execution results
    """
    results = {}
    
    # Read JSON results
    result_file = sandbox_path / "result.json"
    if result_file.exists():
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                results['json'] = json.load(f)
        except Exception as e:
            results['json_error'] = str(e)
    
    # Read image results (base64 encoded PNGs)
    image_files = list(sandbox_path.glob("*.png"))
    if image_files:
        results['images'] = []
        for img_file in image_files:
            try:
                import base64
                with open(img_file, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                    results['images'].append({
                        'filename': img_file.name,
                        'data': img_data
                    })
            except Exception as e:
                results[f'image_error_{img_file.name}'] = str(e)
    
    # Read text outputs
    for filename in ['stdout.txt', 'stderr.txt', 'output.txt']:
        file_path = sandbox_path / filename
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results[filename.replace('.txt', '')] = f.read()
            except Exception as e:
                results[f'{filename}_error'] = str(e)
    
    return results

def get_file_sample_content(file_path: Path, max_lines: int = 10) -> str:
    """
    Get sample content from a file for LLM context.
    
    Args:
        file_path: Path to the file
        max_lines: Maximum number of lines to return
        
    Returns:
        Sample content as string
    """
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, nrows=max_lines)
            return df.to_string()
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)[:1000]  # Limit to 1000 chars
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip())
                return '\n'.join(lines)
    except Exception:
        return f"Could not read sample from {file_path.name}"

def estimate_execution_time(analysis_type: str, file_sizes: List[int]) -> int:
    """
    Estimate execution time based on analysis type and file sizes.
    
    Args:
        analysis_type: Type of analysis
        file_sizes: List of file sizes in bytes
        
    Returns:
        Estimated timeout in seconds
    """
    base_timeout = 30
    total_size_mb = sum(file_sizes) / (1024 * 1024)
    
    # Adjust timeout based on analysis type
    multipliers = {
        'network': 2.0,
        'statistical': 1.5,
        'timeseries': 1.8,
        'ml': 3.0,
        'general': 1.0
    }
    
    multiplier = multipliers.get(analysis_type, 1.0)
    
    # Add time based on file size
    size_timeout = min(total_size_mb * 5, 120)  # Max 2 minutes for size
    
    return int(base_timeout * multiplier + size_timeout)

def validate_generated_code(code: str) -> Tuple[bool, str]:
    """
    Perform basic validation on generated code.
    
    Args:
        code: Python code to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for dangerous operations
    dangerous_patterns = [
        'import os',
        'import subprocess',
        'import sys',
        'eval(',
        'exec(',
        '__import__',
        'open(',
        'file(',
        'input(',
        'raw_input(',
        'execfile(',
        'reload(',
        'exit(',
        'quit(',
    ]
    
    # Allow specific safe imports
    safe_imports = [
        'import pandas',
        'import numpy',
        'import matplotlib',
        'import networkx',
        'import json',
        'import csv',
        'import base64',
        'from pandas',
        'from numpy',
        'from matplotlib',
        'from networkx',
        'import seaborn',
        'from seaborn',
        'import plotly',
        'from plotly',
        'import scipy',
        'from scipy'
    ]
    
    lines = code.split('\n')
    for i, line in enumerate(lines, 1):
        line_lower = line.lower().strip()
        
        # Skip empty lines and comments
        if not line_lower or line_lower.startswith('#'):
            continue
        
        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if pattern in line_lower:
                # Check if it's a safe import
                is_safe = any(safe_pattern in line_lower for safe_pattern in safe_imports)
                if not is_safe:
                    return False, f"Potentially dangerous operation on line {i}: {pattern}"
    
    # Basic syntax check
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    
    return True, "Code validation passed"

def cleanup_sandbox_directory(sandbox_path: Path) -> None:
    """
    Clean up sandbox directory after execution.
    
    Args:
        sandbox_path: Path to sandbox directory to clean up
    """
    try:
        if sandbox_path.exists() and sandbox_path.is_dir():
            shutil.rmtree(sandbox_path)
    except Exception:
        pass  # Ignore cleanup errors

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"