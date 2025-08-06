"""
FastAPI backend for file-driven data analysis API system.
Accepts file uploads and natural language analysis requests.
"""
import os
import uuid
import json
import time
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .logger import setup_logger, log_api_request
from .utils import (
    validate_file_type, save_uploaded_files, 
    analyze_file_structure, read_execution_results
)
from .llm import generate_analysis_code
from .docker_runner import execute_code_in_docker

# Initialize FastAPI app
app = FastAPI(
    title="Data Analysis API",
    description="Upload files and get AI-powered data analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger
logger = setup_logger()

# Create required directories
SANDBOX_DIR = Path("sandbox")
LOGS_DIR = Path("logs")
EXAMPLES_DIR = Path("examples")

for directory in [SANDBOX_DIR, LOGS_DIR, EXAMPLES_DIR]:
    directory.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Data Analysis API server")
    logger.info("Required directories created/verified")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Data Analysis API server")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Data Analysis API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "directories": {
            "sandbox": SANDBOX_DIR.exists(),
            "logs": LOGS_DIR.exists(),
            "examples": EXAMPLES_DIR.exists()
        }
    }

@app.post("/api/")
async def process_files(
    question: UploadFile = File(...),
    files: List[UploadFile] = File(...)
):
    """
    Phase 2 endpoint for processing files with natural language questions.
    
    Args:
        question: Required .txt file containing the natural language question
        files: One or more additional files (CSV, JSON, HTML, etc.)
    
    Returns:
        JSON response with status and UUID
    """
    request_uuid = str(uuid.uuid4())
    
    try:
        # Log the incoming request with UUID
        logger.info(f"Request {request_uuid}: Starting file processing")
        logger.info(f"Request {request_uuid}: Received question file: {question.filename}")
        logger.info(f"Request {request_uuid}: Received {len(files)} additional files: {[f.filename for f in files]}")
        
        # Validate question file is a .txt file
        if not question.filename.endswith('.txt'):
            logger.error(f"Request {request_uuid}: Question file must be .txt, got: {question.filename}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question file must be a .txt file"
            )
        
        # Create sandbox directory for this request
        sandbox_path = SANDBOX_DIR / request_uuid
        sandbox_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Request {request_uuid}: Created sandbox directory: {sandbox_path}")
        
        # Save question file
        question_path = sandbox_path / "question.txt"
        with open(question_path, "wb") as buffer:
            content = await question.read()
            buffer.write(content)
        logger.info(f"Request {request_uuid}: Saved question file to {question_path}")
        
        # Save additional files
        saved_files = []
        for file in files:
            file_path = sandbox_path / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            saved_files.append(file.filename)
            logger.info(f"Request {request_uuid}: Saved file: {file.filename}")
        
        # Read and print preview of question.txt file
        with open(question_path, "r", encoding="utf-8") as f:
            question_content = f.read()
        
        logger.info(f"Request {request_uuid}: Question content preview: {question_content[:200]}...")
        print(f"Question preview for {request_uuid}: {question_content[:200]}...")
        
        logger.info(f"Request {request_uuid}: Successfully processed all files")
        
        # Return placeholder JSON response
        return JSONResponse(content={
            "status": "ok",
            "uuid": request_uuid
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request {request_uuid}: Unexpected error - {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/status/{request_id}")
async def get_request_status(request_id: str):
    """Get the status of a specific request."""
    return {"request_id": request_id, "message": "Status tracking not implemented yet"}

@app.post("/api/analyze")
async def analyze_files(
    files: List[UploadFile] = File(...),
    question: str = Form(...),
    analysis_type: str = Form(default="general"),
    timeout: int = Form(default=60)
):
    """
    Main analysis endpoint that processes files with natural language questions.
    
    Args:
        files: List of files to upload and analyze
        question: Natural language analysis question
        analysis_type: Type of analysis (statistical, network, timeseries, ml, general)
        timeout: Maximum execution time in seconds
    
    Returns:
        JSON response with analysis results including parsed JSON, base64 images, and execution details
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Log the incoming request
        log_api_request(
            logger=logger,
            request_id=request_id,
            endpoint="/api/analyze",
            method="POST",
            files_count=len(files),
            question_preview=question[:100] + "..." if len(question) > 100 else question
        )
        
        # Validate inputs
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one file must be uploaded"
            )
        
        if not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        # Validate file types
        for file in files:
            if not validate_file_type(file.filename):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file.filename}"
                )
        
        # Create sandbox directory for this request
        sandbox_path = SANDBOX_DIR / request_id
        sandbox_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Request {request_id}: Created sandbox directory: {sandbox_path}")
        
        # Save uploaded files
        file_paths = await save_uploaded_files(files, sandbox_path)
        logger.info(f"Request {request_id}: Saved {len(file_paths)} files to sandbox")
        
        # Analyze file structure for LLM context
        file_analysis = analyze_file_structure(file_paths)
        logger.info(f"Request {request_id}: Analyzed file structure")
        
        # Generate analysis code using LLM
        logger.info(f"Request {request_id}: Generating analysis code with LLM")
        generated_code = await generate_analysis_code(
            question=question,
            file_paths=file_paths,
            analysis_type=analysis_type,
            sandbox_path=sandbox_path,
            request_id=request_id
        )
        
        if not generated_code:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate analysis code"
            )
        
        logger.info(f"Request {request_id}: Successfully generated analysis code")
        
        # Execute the generated code in Docker
        logger.info(f"Request {request_id}: Executing code in Docker container")
        execution_result = await execute_code_in_docker(
            code=generated_code,
            sandbox_path=sandbox_path,
            request_id=request_id,
            timeout=timeout
        )
        
        # Process outputs from sandbox
        response_data = await process_execution_outputs(
            sandbox_path=sandbox_path,
            execution_result=execution_result,
            request_id=request_id
        )
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        # Build final response
        final_response = {
            "request_id": request_id,
            "status": "success" if execution_result.get("success", False) else "error",
            "question": question,
            "analysis_type": analysis_type,
            "files_processed": len(file_paths),
            "execution_time": total_time,
            "docker_execution_time": execution_result.get("execution_time", 0),
            "results": response_data
        }
        
        # Add error information if execution failed
        if not execution_result.get("success", False):
            final_response["error"] = execution_result.get("error", "Unknown execution error")
            final_response["stderr"] = execution_result.get("stderr", "")
        
        # Validate response structure before returning
        validated_response = validate_response_structure(final_response)
        
        logger.info(f"Request {request_id}: Analysis completed successfully in {total_time:.2f}s")
        
        return JSONResponse(content=validated_response)
        
    except HTTPException:
        raise
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Request {request_id}: Unexpected error after {error_time:.2f}s - {str(e)}")
        
        # Return structured error response
        error_response = {
            "request_id": request_id,
            "status": "error",
            "question": question if 'question' in locals() else "Unknown",
            "analysis_type": analysis_type if 'analysis_type' in locals() else "general",
            "files_processed": 0,
            "execution_time": error_time,
            "error": f"Server error: {str(e)}",
            "results": {}
        }
        
        return JSONResponse(
            content=error_response,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


async def process_execution_outputs(
    sandbox_path: Path,
    execution_result: Dict[str, Any],
    request_id: str
) -> Dict[str, Any]:
    """
    Process outputs from Docker execution in the sandbox directory.
    
    Args:
        sandbox_path: Path to sandbox directory
        execution_result: Result from Docker execution
        request_id: Request ID for logging
        
    Returns:
        Processed output data
    """
    try:
        logger.info(f"Request {request_id}: Processing execution outputs")
        
        # Read execution results from sandbox (JSON, images, text outputs)
        results = read_execution_results(sandbox_path)
        
        # Initialize response structure
        response_data = {
            "json": None,
            "images": [],
            "stdout": execution_result.get("stdout", ""),
            "stderr": execution_result.get("stderr", ""),
            "return_code": execution_result.get("return_code", -1)
        }
        
        # Process JSON results
        if "json" in results:
            try:
                response_data["json"] = results["json"]
                logger.info(f"Request {request_id}: Successfully parsed result.json")
            except Exception as e:
                logger.warning(f"Request {request_id}: Error parsing JSON results: {str(e)}")
                response_data["json_parse_error"] = str(e)
        elif "json_error" in results:
            logger.warning(f"Request {request_id}: JSON file error: {results['json_error']}")
            response_data["json_error"] = results["json_error"]
        
        # Process image results with size validation
        if "images" in results:
            processed_images = []
            for img in results["images"]:
                try:
                    # Validate image size (under 100KB as per requirements)
                    img_data = img.get("data", "")
                    original_size = img.get("original_size_bytes", 0)
                    filename = img.get("filename", "unknown.png")
                    
                    if img_data and original_size <= 100 * 1024:  # 100KB limit
                        processed_images.append({
                            "filename": filename,
                            "data": img_data,
                            "size_bytes": original_size
                        })
                        logger.info(f"Request {request_id}: Included image {filename} ({original_size} bytes)")
                    elif original_size > 100 * 1024:
                        logger.warning(f"Request {request_id}: Image {filename} too large ({original_size} bytes), skipping")
                        response_data.setdefault("warnings", []).append(
                            f"Image {filename} excluded: exceeds 100KB limit ({original_size} bytes)"
                        )
                    
                except Exception as e:
                    logger.warning(f"Request {request_id}: Error processing image {img.get('filename', 'unknown')}: {str(e)}")
            
            response_data["images"] = processed_images
        
        # Add any additional text outputs
        for key in ["output", "stdout_file", "stderr_file"]:
            if key in results:
                response_data[f"additional_{key}"] = results[key]
        
        # Add warnings for any file errors
        for key, value in results.items():
            if key.endswith("_error") and not key.startswith("json"):
                response_data.setdefault("warnings", []).append(f"{key}: {value}")
        
        logger.info(f"Request {request_id}: Processed outputs - JSON: {'yes' if response_data['json'] else 'no'}, Images: {len(response_data['images'])}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Request {request_id}: Error processing execution outputs: {str(e)}")
        return {
            "json": None,
            "images": [],
            "stdout": execution_result.get("stdout", ""),
            "stderr": execution_result.get("stderr", ""),
            "return_code": execution_result.get("return_code", -1),
            "processing_error": str(e)
        }


def validate_response_structure(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize the response structure before returning.
    
    Args:
        response: Response dictionary to validate
        
    Returns:
        Validated and sanitized response
    """
    # Ensure required fields exist
    required_fields = [
        "request_id", "status", "question", "analysis_type", 
        "files_processed", "execution_time", "results"
    ]
    
    for field in required_fields:
        if field not in response:
            response[field] = None
    
    # Ensure results structure
    if not isinstance(response.get("results"), dict):
        response["results"] = {}
    
    results = response["results"]
    
    # Ensure results has required subfields
    if "json" not in results:
        results["json"] = None
    if "images" not in results:
        results["images"] = []
    if "stdout" not in results:
        results["stdout"] = ""
    if "stderr" not in results:
        results["stderr"] = ""
    
    # Validate image data structure
    if isinstance(results["images"], list):
        validated_images = []
        for img in results["images"]:
            if isinstance(img, dict) and "filename" in img and "data" in img:
                validated_images.append({
                    "filename": str(img["filename"]),
                    "data": str(img["data"]),
                    "size_bytes": img.get("size_bytes", 0)
                })
        results["images"] = validated_images
    
    # Sanitize string fields to prevent potential issues
    string_fields = ["question", "analysis_type", "error"]
    for field in string_fields:
        if field in response and response[field] is not None:
            response[field] = str(response[field])[:10000]  # Limit to 10K chars
    
    # Ensure numeric fields are properly typed
    numeric_fields = ["files_processed", "execution_time", "docker_execution_time"]
    for field in numeric_fields:
        if field in response:
            try:
                response[field] = float(response[field]) if response[field] is not None else 0
            except (ValueError, TypeError):
                response[field] = 0
    
    return response