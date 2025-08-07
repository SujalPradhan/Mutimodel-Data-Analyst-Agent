#!/usr/bin/env python3
"""
Server startup script for the Data Analysis API.
"""
import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Ensure required directories exist
    Path("sandbox").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("examples").mkdir(exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
