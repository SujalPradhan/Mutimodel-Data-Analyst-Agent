"""
FastAPI backend for file-driven data analysis API system.
Accepts file uploads and natural language analysis requests.
"""
import os
import uuid
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile

from logger import setup_logger, log_request, log_execution
from utils import validate_file_type, save_uploaded_files, create_sandbox_directory
from llm import generate_analysis_code
from docker_runner import execute_code_in_docker

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

@app.post("/api/analyze")
async def analyze_data(
    files: List[UploadFile] = File(...),
    question: str = Form(...),
    analysis_type: Optional[str] = Form(default="general"),
    timeout: Optional[int] = Form(default=60)
):
    """
    Main endpoint for data analysis.
    
    Args:
        files: List of uploaded files (CSV, JSON, HTML, ZIP, TXT, etc.)
        question: Natural language question or analysis request
        analysis_type: Type of analysis (general, statistical, network, timeseries)
        timeout: Maximum execution time in seconds
    
    Returns:
        JSON response with analysis results
    """
    request_id = str(uuid.uuid4())
    
    try:
        # Log the incoming request
        log_request(logger, request_id, files, question, analysis_type)
        
        # Validate file types
        for file in files:
            if not validate_file_type(file.filename):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file.filename}"
                )
        
        # Create sandbox directory for this request
        sandbox_path = create_sandbox_directory(request_id)
        
        # Save uploaded files
        file_paths = await save_uploaded_files(files, sandbox_path)
        logger.info(f"Request {request_id}: Saved {len(file_paths)} files")
        
        # Generate analysis code using LLM
        generated_code = await generate_analysis_code(
            question=question,
            file_paths=file_paths,
            analysis_type=analysis_type,
            sandbox_path=sandbox_path
        )
        
        if not generated_code:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate analysis code"
            )
        
        logger.info(f"Request {request_id}: Generated analysis code")
        
        # Execute code in Docker container
        execution_result = await execute_code_in_docker(
            code=generated_code,
            sandbox_path=sandbox_path,
            request_id=request_id,
            timeout=timeout
        )
        
        log_execution(logger, request_id, execution_result)
        
        # Prepare response
        response_data = {
            "request_id": request_id,
            "status": "success" if execution_result["success"] else "error",
            "question": question,
            "analysis_type": analysis_type,
            "files_processed": len(file_paths),
            "execution_time": execution_result.get("execution_time", 0),
            "results": execution_result.get("results", {}),
            "stdout": execution_result.get("stdout", ""),
            "stderr": execution_result.get("stderr", "")
        }
        
        if not execution_result["success"]:
            response_data["error"] = execution_result.get("error", "Unknown error occurred")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request {request_id}: Unexpected error - {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Cleanup sandbox directory after processing
        try:
            if 'sandbox_path' in locals() and sandbox_path.exists():
                shutil.rmtree(sandbox_path)
                logger.info(f"Request {request_id}: Cleaned up sandbox directory")
        except Exception as e:
            logger.error(f"Request {request_id}: Failed to cleanup sandbox - {str(e)}")

@app.get("/api/status/{request_id}")
async def get_request_status(request_id: str):
    """Get the status of a specific request (for future async processing)."""
    # This endpoint can be extended for async processing
    return {"request_id": request_id, "message": "Status tracking not implemented yet"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )