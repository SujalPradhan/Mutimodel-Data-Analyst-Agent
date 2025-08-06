"""
Data Analysis API Package

A FastAPI-based system for file-driven data analysis with AI code generation.
"""

__version__ = "1.0.0"
__author__ = "Data Analysis API Team"
__description__ = "AI-powered file analysis system"

# Package exports
from .main import app
from .logger import setup_logger
from .utils import validate_file_type, save_uploaded_files
from .llm import generate_analysis_code
from .docker_runner import execute_code_in_docker

__all__ = [
    "app",
    "setup_logger",
    "validate_file_type",
    "save_uploaded_files",
    "generate_analysis_code",
    "execute_code_in_docker",
]