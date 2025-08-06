"""
Docker container runner for executing LLM-generated Python code safely.
"""
import os
import time
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import shutil

from .logger import setup_logger, log_docker_operation, logger
from .utils import create_code_file, read_execution_results

# Initialize logger (use global logger from logger module)
# logger = setup_logger()

# Docker configuration
DOCKER_IMAGE = "python:3.11-slim"
DOCKER_TIMEOUT = 300  # 5 minutes max execution time

async def execute_code_in_docker(
    code: str,
    sandbox_path: Path,
    request_id: str,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Execute Python code safely in a Docker container.
    
    Args:
        code: Python code to execute
        sandbox_path: Path to sandbox directory with files
        request_id: Request ID for logging
        timeout: Maximum execution time in seconds
        
    Returns:
        Dictionary with execution results
    """
    start_time = time.time()
    container_id = None
    
    try:
        # Create code file
        code_file = create_code_file(code, sandbox_path)
        logger.info(f"Created code file for request {request_id}: {code_file}")
        
        # Prepare Docker command
        docker_cmd = [
            "docker", "run",
            "--rm",  # Remove container after execution
            "--network", "none",  # No network access
            "--memory", "512m",  # Memory limit
            "--cpus", "1.0",  # CPU limit
            "-v", f"{sandbox_path.absolute()}:/workspace",  # Mount sandbox as volume
            "-w", "/workspace",  # Set working directory
            "--user", "1000:1000",  # Run as non-root user
            DOCKER_IMAGE,
            "python", "analysis.py"
        ]
        
        log_docker_operation(logger, request_id, "start")
        
        # Execute Docker container
        process = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=sandbox_path
        )
        
        # Wait for completion with timeout
        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            # Kill the process if it times out
            try:
                process.kill()
                await process.wait()
            except:
                pass
            
            execution_time = time.time() - start_time
            log_docker_operation(logger, request_id, "timeout", duration=execution_time)
            
            return {
                "success": False,
                "error": f"Execution timed out after {timeout} seconds",
                "execution_time": execution_time,
                "stdout": "",
                "stderr": f"Process timed out after {timeout} seconds",
                "results": {}
            }
        
        execution_time = time.time() - start_time
        
        # Decode output
        stdout_str = stdout_data.decode('utf-8', errors='replace') if stdout_data else ""
        stderr_str = stderr_data.decode('utf-8', errors='replace') if stderr_data else ""
        
        # Check if execution was successful
        success = process.returncode == 0
        
        # Read results from sandbox directory
        results = read_execution_results(sandbox_path)
        
        # Log completion
        log_docker_operation(
            logger, request_id, "complete",
            duration=execution_time
        )
        
        return {
            "success": success,
            "execution_time": execution_time,
            "return_code": process.returncode,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "results": results,
            "error": None if success else f"Process exited with code {process.returncode}"
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Docker execution error for request {request_id}: {str(e)}")
        
        return {
            "success": False,
            "error": f"Docker execution failed: {str(e)}",
            "execution_time": execution_time,
            "stdout": "",
            "stderr": str(e),
            "results": {}
        }

async def check_docker_availability() -> bool:
    """
    Check if Docker is available and running.
    
    Returns:
        True if Docker is available, False otherwise
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "docker", "version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await asyncio.wait_for(process.communicate(), timeout=10)
        return process.returncode == 0
        
    except Exception:
        return False

async def pull_docker_image() -> bool:
    """
    Pull the required Docker image if not available.
    
    Returns:
        True if image is available/pulled successfully, False otherwise
    """
    try:
        # Check if image exists locally
        check_cmd = ["docker", "images", "-q", DOCKER_IMAGE]
        process = await asyncio.create_subprocess_exec(
            *check_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout_data, _ = await asyncio.wait_for(process.communicate(), timeout=10)
        
        if stdout_data.decode().strip():
            logger.info(f"Docker image {DOCKER_IMAGE} already available locally")
            return True
        
        # Pull image
        logger.info(f"Pulling Docker image {DOCKER_IMAGE}")
        pull_cmd = ["docker", "pull", DOCKER_IMAGE]
        process = await asyncio.create_subprocess_exec(
            *pull_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await asyncio.wait_for(process.communicate(), timeout=300)  # 5 minutes timeout
        
        success = process.returncode == 0
        if success:
            logger.info(f"Successfully pulled Docker image {DOCKER_IMAGE}")
        else:
            logger.error(f"Failed to pull Docker image {DOCKER_IMAGE}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error checking/pulling Docker image: {str(e)}")
        return False

def create_dockerfile_content() -> str:
    """
    Create Dockerfile content for the analysis environment.
    
    Returns:
        Dockerfile content as string
    """
    return """
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \\
    pandas==2.1.4 \\
    numpy==1.24.4 \\
    matplotlib==3.7.4 \\
    seaborn==0.13.0 \\
    networkx==3.2.1 \\
    scipy==1.11.4 \\
    plotly==5.17.0 \\
    beautifulsoup4==4.12.2 \\
    requests==2.31.0 \\
    duckdb==0.9.2 \\
    openpyxl==3.1.2

# Create non-root user
RUN useradd -m -u 1000 analyst
USER analyst

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python"]
"""

def build_custom_docker_image() -> bool:
    """
    Build custom Docker image with required packages.
    
    Returns:
        True if build successful, False otherwise
    """
    try:
        # Create temporary directory for build context
        with tempfile.TemporaryDirectory() as temp_dir:
            dockerfile_path = Path(temp_dir) / "Dockerfile"
            
            # Write Dockerfile
            with open(dockerfile_path, 'w') as f:
                f.write(create_dockerfile_content())
            
            # Build image
            build_cmd = [
                "docker", "build",
                "-t", "data-analysis-env:latest",
                temp_dir
            ]
            
            result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("Successfully built custom Docker image")
                return True
            else:
                logger.error(f"Failed to build Docker image: {result.stderr}")
                return False
                
    except Exception as e:
        logger.error(f"Error building Docker image: {str(e)}")
        return False

async def execute_with_custom_image(
    code: str,
    sandbox_path: Path,
    request_id: str,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Execute code using custom Docker image with pre-installed packages.
    
    Args:
        code: Python code to execute
        sandbox_path: Path to sandbox directory
        request_id: Request ID for logging
        timeout: Maximum execution time in seconds
        
    Returns:
        Dictionary with execution results
    """
    # Use custom image if available, otherwise fallback to standard
    image_name = "data-analysis-env:latest"
    
    # Check if custom image exists
    try:
        check_cmd = ["docker", "images", "-q", image_name]
        process = await asyncio.create_subprocess_exec(
            *check_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout_data, _ = await process.communicate()
        
        if not stdout_data.decode().strip():
            # Custom image doesn't exist, use standard image
            image_name = DOCKER_IMAGE
            
    except Exception:
        image_name = DOCKER_IMAGE
    
    # Execute with the selected image
    return await execute_code_with_image(
        code=code,
        sandbox_path=sandbox_path,
        request_id=request_id,
        timeout=timeout,
        image_name=image_name
    )

async def execute_code_with_image(
    code: str,
    sandbox_path: Path,
    request_id: str,
    timeout: int,
    image_name: str
) -> Dict[str, Any]:
    """
    Execute code with a specific Docker image.
    
    Args:
        code: Python code to execute
        sandbox_path: Path to sandbox directory
        request_id: Request ID for logging
        timeout: Maximum execution time in seconds
        image_name: Docker image to use
        
    Returns:
        Dictionary with execution results
    """
    start_time = time.time()
    
    try:
        # Create code file
        code_file = create_code_file(code, sandbox_path)
        
        # Install packages script for standard Python image
        install_script = """
pip install --no-cache-dir pandas numpy matplotlib seaborn networkx scipy plotly beautifulsoup4 requests duckdb openpyxl 2>/dev/null || echo "Package installation failed" && python analysis.py
"""
        
        # Create install script file
        install_file = sandbox_path / "install_and_run.sh"
        with open(install_file, 'w') as f:
            f.write(install_script)
        
        # Make script executable
        os.chmod(install_file, 0o755)
        
        # Prepare Docker command
        if image_name == DOCKER_IMAGE:
            # Use install script for standard Python image
            cmd_args = ["bash", "install_and_run.sh"]
        else:
            # Use direct Python execution for custom image
            cmd_args = ["python", "analysis.py"]
        
        docker_cmd = [
            "docker", "run",
            "--rm",
            "--network", "none",
            "--memory", "512m",
            "--cpus", "1.0",
            "-v", f"{sandbox_path.absolute()}:/workspace",
            "-w", "/workspace",
            "--user", "1000:1000",
            image_name,
            *cmd_args
        ]
        
        # Execute Docker container
        process = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait()
            except:
                pass
            
            return {
                "success": False,
                "error": f"Execution timed out after {timeout} seconds",
                "execution_time": time.time() - start_time,
                "stdout": "",
                "stderr": f"Process timed out after {timeout} seconds",
                "results": {}
            }
        
        execution_time = time.time() - start_time
        stdout_str = stdout_data.decode('utf-8', errors='replace') if stdout_data else ""
        stderr_str = stderr_data.decode('utf-8', errors='replace') if stderr_data else ""
        
        success = process.returncode == 0
        results = read_execution_results(sandbox_path)
        
        return {
            "success": success,
            "execution_time": execution_time,
            "return_code": process.returncode,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "results": results,
            "error": None if success else f"Process exited with code {process.returncode}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Docker execution failed: {str(e)}",
            "execution_time": time.time() - start_time,
            "stdout": "",
            "stderr": str(e),
            "results": {}
        }

def cleanup_docker_resources():
    """Clean up any dangling Docker resources."""
    try:
        # Remove dangling images
        subprocess.run(
            ["docker", "image", "prune", "-f"],
            capture_output=True,
            timeout=30
        )
        
        # Remove dangling containers
        subprocess.run(
            ["docker", "container", "prune", "-f"],
            capture_output=True,
            timeout=30
        )
        
        logger.info("Docker cleanup completed")
        
    except Exception as e:
        logger.warning(f"Docker cleanup failed: {str(e)}")

async def get_docker_stats() -> Dict[str, Any]:
    """
    Get Docker system statistics.
    
    Returns:
        Dictionary with Docker stats
    """
    try:
        # Get Docker info
        process = await asyncio.create_subprocess_exec(
            "docker", "info", "--format", "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout_data, _ = await process.communicate()
        
        if process.returncode == 0:
            info = json.loads(stdout_data.decode())
            return {
                "available": True,
                "containers": info.get("Containers", 0),
                "images": info.get("Images", 0),
                "server_version": info.get("ServerVersion", "unknown"),
                "memory_limit": info.get("MemTotal", 0)
            }
        else:
            return {"available": False, "error": "Docker not accessible"}
            
    except Exception as e:
        return {"available": False, "error": str(e)}

class DockerExecutor:
    """Context manager for Docker execution with cleanup."""
    
    def __init__(self, sandbox_path: Path, request_id: str):
        self.sandbox_path = sandbox_path
        self.request_id = request_id
        self.start_time = None
        
    async def __aenter__(self):
        self.start_time = time.time()
        log_docker_operation(self.logger, self.request_id, "initialize")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0
        
        if exc_type is None:
            log_docker_operation(self.logger, self.request_id, "cleanup", duration=duration)
        else:
            logger.error(f"Docker execution failed for {self.request_id}: {exc_val}")
        
        # Cleanup any resources if needed
        cleanup_docker_resources()
        
    async def execute(self, code: str, timeout: int = 60) -> Dict[str, Any]:
        """Execute code and return results."""
        return await execute_code_in_docker(
            code=code,
            sandbox_path=self.sandbox_path,
            request_id=self.request_id,
            timeout=timeout
        )

def run_code_in_docker(code_str: str, input_dir: str) -> dict:
    """
    Synchronous wrapper to run Python code in a Docker container.
    
    Args:
        code_str: Python code to execute
        input_dir: Directory path containing input files (will be used as sandbox)
        
    Returns:
        Dictionary with execution results containing:
        - success: Boolean indicating if execution was successful
        - stdout: Standard output from execution
        - stderr: Standard error from execution
        - execution_time: Time taken for execution in seconds
        - return_code: Process return code
        - error: Error message if execution failed
    """
    import subprocess
    import time
    
    try:
        # Convert input_dir to Path object
        input_path = Path(input_dir)
        
        # Create input directory if it doesn't exist
        input_path.mkdir(parents=True, exist_ok=True)
        
        # Save code to script.py in the input directory
        script_path = input_path / "script.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(code_str)
        
        # Prepare Docker command
        docker_cmd = [
            "docker", "run",
            "--rm",  # Remove container after execution
            "--memory", "512m",  # Memory limit
            "-v", f"{input_path.absolute()}:/sandbox",  # Mount input_dir as /sandbox
            "python:3.11-slim",  # Base image
            "python", "/sandbox/script.py"  # Command to execute
        ]
        
        # Record start time
        start_time = time.time()
        
        # Execute Docker container using subprocess.run()
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Decode output
        stdout_str = result.stdout if result.stdout else ""
        stderr_str = result.stderr if result.stderr else ""
        
        # Check if execution was successful
        success = result.returncode == 0
        
        return {
            "success": success,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "execution_time": execution_time,
            "return_code": result.returncode,
            "error": None if success else f"Process exited with code {result.returncode}"
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Process timed out after 300 seconds",
            "execution_time": 300,
            "return_code": -1,
            "error": "Execution timed out"
        }
    except Exception as e:
        logger.error(f"Error in run_code_in_docker: {str(e)}")
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "execution_time": 0,
            "return_code": -1,
            "error": f"Failed to execute code: {str(e)}"
        }
