# Data Analysis API - Complete Tutorial

## Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Quick Start Guide](#quick-start-guide)
4. [How to Use the API](#how-to-use-the-api)
5. [System Architecture Deep Dive](#system-architecture-deep-dive)
6. [Advanced Usage Examples](#advanced-usage-examples)
7. [Troubleshooting](#troubleshooting)
8. [Development Guide](#development-guide)

---

## System Overview

The **Data Analysis API** is a FastAPI-based system that combines AI-powered code generation with secure Docker execution to perform data analysis tasks. Here's what makes it special:

### 🔄 How It Works
1. **Upload Files**: You provide data files (CSV, JSON, Excel, etc.)
2. **Ask Questions**: Write natural language questions about your data
3. **AI Code Generation**: Google's Gemini API generates Python analysis code
4. **Secure Execution**: Code runs in isolated Docker containers
5. **Get Results**: Receive JSON data, visualizations, and insights

### 🎯 Key Features
- **Natural Language Interface**: No coding required - just ask questions
- **Multi-file Support**: Upload multiple data files simultaneously
- **Secure Sandboxing**: All code execution happens in isolated Docker containers
- **Multiple Analysis Types**: Statistical, network, time-series, ML, and general analysis
- **Rich Output**: JSON results, base64-encoded visualizations, and detailed logs

---

## Prerequisites & Installation

### System Requirements
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: 3.11 or higher
- **Docker**: Latest version with Docker Compose
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: 2GB free disk space

### Step 1: Install Docker
```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose
sudo systemctl start docker
sudo usermod -aG docker $USER  # Add yourself to docker group
newgrp docker  # Refresh group membership

# On macOS
brew install docker docker-compose
# Or download Docker Desktop from docker.com

# On Windows
# Download Docker Desktop from docker.com and enable WSL2 integration
```

### Step 2: Get Google Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key (you'll need it in Step 4)

### Step 3: Clone and Setup Project
```bash
# Clone the repository
git clone <your-repository-url>
cd TDS-P2v3

# Verify project structure
ls -la
# You should see: app/, examples/, requirements.txt, docker-compose.yaml, etc.
```

### Step 4: Configure Environment Variables
```bash
# Create environment file
cp .env.example .env

# Edit the .env file with your API key
nano .env  # or use your preferred editor
```

Add this content to `.env`:
```bash
# Google Gemini API Configuration
GEMINI_API_KEY=your_actual_api_key_here

# Application Configuration
LOG_LEVEL=INFO
DOCKER_TIMEOUT=300
MAX_FILE_SIZE=50MB
MAX_FILES_PER_REQUEST=10

# Development Settings (optional)
DEBUG=False
```

### Step 5: Verify Docker Setup
```bash
# Test Docker installation
docker --version
docker-compose --version

# Ensure Docker daemon is running
docker ps

# Test Docker permissions (should not require sudo)
docker run hello-world
```

---

## Quick Start Guide

### Method 1: Docker Compose (Recommended)
This is the easiest way to get started:

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build

# Check if services are running
docker-compose ps
```

Expected output:
```
Name                      Command               State           Ports         
----------------------------------------------------------------------------
tds-p2v3_data-analysis-api_1   uvicorn app.main:app --host ...   Up      0.0.0.0:8000->8000/tcp
tds-p2v3_redis_1               docker-entrypoint.sh redis ...   Up      0.0.0.0:6379->6379/tcp
```

### Method 2: Local Development
For development and debugging:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Verify Installation
1. **Open your browser** and navigate to: `http://localhost:8000`
2. **Check API documentation**: `http://localhost:8000/docs`
3. **Test health endpoint**: `http://localhost:8000/health`

You should see:
```json
{
  "status": "healthy",
  "directories": {
    "sandbox": true,
    "logs": true,
    "examples": true
  }
}
```

---

## How to Use the API

### Understanding the Endpoints

#### 1. Health Check - `GET /health`
Always start here to verify the system is working:

```bash
curl http://localhost:8000/health
```

#### 2. Main Analysis Endpoint - `POST /api/analyze`
This is where the magic happens:

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@your_data.csv" \
  -F "question=What are the main trends in this data?" \
  -F "analysis_type=statistical"
```

### Step-by-Step Usage Example

#### Example 1: Basic Statistical Analysis

1. **Prepare your data file** (let's say `sales_data.csv`):
```csv
date,product,sales,region
2024-01-01,Widget A,1500,North
2024-01-02,Widget B,2300,South
2024-01-03,Widget A,1800,East
```

2. **Make the API call**:
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@sales_data.csv" \
  -F "question=Analyze sales trends by product and region. Show me descriptive statistics and create visualizations." \
  -F "analysis_type=statistical" \
  -F "timeout=120"
```

3. **Understand the response**:
```json
{
  "request_id": "uuid-here",
  "status": "success",
  "question": "Analyze sales trends...",
  "analysis_type": "statistical", 
  "files_processed": 1,
  "execution_time": 8.5,
  "results": {
    "json": {
      "summary_stats": {
        "sales": {
          "mean": 1866.67,
          "std": 416.33,
          "min": 1500,
          "max": 2300
        }
      },
      "insights": [
        "Widget B has the highest average sales",
        "South region shows strongest performance"
      ]
    },
    "images": [
      {
        "filename": "sales_by_product.png",
        "data": "base64-encoded-image-data"
      }
    ],
    "stdout": "Analysis completed successfully",
    "stderr": ""
  }
}
```

#### Example 2: Network Analysis

Using the provided example:

```bash
# Using the example network data
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@example1/edges.csv" \
  -F "question=$(cat example1/question.txt)" \
  -F "analysis_type=network"
```

#### Example 3: Multiple Files Analysis

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@data1.csv" \
  -F "files=@data2.json" \
  -F "files=@data3.xlsx" \
  -F "question=Compare trends across all three datasets and find correlations" \
  -F "analysis_type=general"
```

---

## Comprehensive API Test Cases

This section provides curl commands to test all the API endpoints using the provided example data. Make sure the API is running before executing these commands.

### Prerequisites for Testing

```bash
# Ensure the API is running
curl http://localhost:8000/health

# Navigate to the project directory
cd /home/sujal/Work/TDS-P2v3

# Verify example files exist
ls -la example*/
```

### Test Case 1: Basic Health Check

```bash
# Health check - should always work first
curl -X GET "http://localhost:8000/health" \
  -H "accept: application/json"
```

**Expected Response:**
```json
{
  "status": "healthy",
  "directories": {
    "sandbox": true,
    "logs": true,
    "examples": true
  }
}
```

### Test Case 2: Network Analysis (Example 1)

```bash
# Network analysis using edges.csv
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@example1/edges.csv" \
  -F "question=$(cat example1/question.txt)" \
  -F "analysis_type=network" \
  -F "timeout=120"
```

**What this tests:**
- Network graph analysis
- Node degree calculations
- Shortest path algorithms
- Graph visualization generation
- Base64 image encoding

### Test Case 3: Sales Data Analysis (Example 2)

```bash
# Statistical analysis of sales data
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@example2/sample-sale.csv" \
  -F "question=$(cat example2/question.txt)" \
  -F "analysis_type=statistical" \
  -F "timeout=120"
```

**What this tests:**
- CSV data processing
- Statistical calculations (totals, medians, correlations)
- Time series analysis
- Regional data grouping
- Chart generation (bar charts, line charts)

### Test Case 4: Weather Data Analysis (Example 3)

```bash
# Time series analysis of weather data
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@example3/sample-weather.csv" \
  -F "question=$(cat example3/question.txt)" \
  -F "analysis_type=timeseries" \
  -F "timeout=120"
```

**What this tests:**
- Temperature and precipitation analysis
- Correlation analysis between variables
- Time series plotting
- Histogram generation
- Date parsing and handling

### Test Case 5: Web Scraping Analysis (Example 4)

```bash
# Complex web scraping and analysis
curl -X POST "http://localhost:8000/api/analyze" \
  -F "question=$(cat example4/question.txt)" \
  -F "analysis_type=general" \
  -F "timeout=300"
```

**What this tests:**
- Web scraping capabilities
- HTML parsing
- Data filtering and analysis
- Regression analysis
- Scatter plot generation

### Test Case 6: Multiple File Upload

```bash
# Test multiple files at once
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@example1/edges.csv" \
  -F "files=@example2/sample-sale.csv" \
  -F "files=@example3/sample-weather.csv" \
  -F "question=Analyze all three datasets and find any interesting patterns or correlations between network properties, sales data, and weather patterns" \
  -F "analysis_type=general" \
  -F "timeout=180"
```

### Test Case 7: Error Handling Tests

#### Missing Files
```bash
# Test error handling for missing files
curl -X POST "http://localhost:8000/api/analyze" \
  -F "question=Analyze this data" \
  -F "analysis_type=statistical"
```

#### Empty Question
```bash
# Test error handling for empty question
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@example1/edges.csv" \
  -F "question=" \
  -F "analysis_type=statistical"
```

#### Invalid File Type
```bash
# Create a test file with unsupported extension
echo "test content" > test.invalid
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@test.invalid" \
  -F "question=Analyze this file" \
  -F "analysis_type=general"
rm test.invalid
```

### Test Case 8: Custom Analysis Types

#### Machine Learning Analysis
```bash
# ML analysis on sales data
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@example2/sample-sale.csv" \
  -F "question=Perform clustering analysis on the sales data. Group similar sales patterns and identify customer segments" \
  -F "analysis_type=ml" \
  -F "timeout=150"
```

#### General Exploratory Analysis
```bash
# General analysis
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@example3/sample-weather.csv" \
  -F "question=Perform a comprehensive exploratory data analysis. Check data quality, find outliers, and generate summary statistics" \
  -F "analysis_type=general" \
  -F "timeout=120"
```

### Test Case 9: Timeout Testing

```bash
# Test with very short timeout (should fail)
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@example1/edges.csv" \
  -F "question=$(cat example1/question.txt)" \
  -F "analysis_type=network" \
  -F "timeout=1"
```

### Test Case 10: Large Analysis Request

```bash
# Complex multi-step analysis
curl -X POST "http://localhost:8000/api/analyze" \
  -F "files=@example2/sample-sale.csv" \
  -F "question=Perform a comprehensive business intelligence analysis: 1) Calculate all key sales metrics, 2) Identify top-performing regions and time periods, 3) Create executive dashboard visualizations, 4) Forecast next quarter sales, 5) Identify anomalies and outliers, 6) Generate actionable business insights" \
  -F "analysis_type=statistical" \
  -F "timeout=300"
```

### Testing Response Validation

Here's a simple script to validate API responses:

```bash
#!/bin/bash
# save as test_api.sh

echo "Testing Data Analysis API..."

# Test 1: Health Check
echo "1. Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
echo "Health Response: $HEALTH_RESPONSE"

# Test 2: Basic Analysis
echo "2. Testing basic analysis..."
ANALYSIS_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/analyze" \
  -F "files=@example1/edges.csv" \
  -F "question=How many nodes are in this network?" \
  -F "analysis_type=network" \
  -F "timeout=60")

echo "Analysis Response: $ANALYSIS_RESPONSE"

# Extract request_id for potential status checking
REQUEST_ID=$(echo $ANALYSIS_RESPONSE | grep -o '"request_id":"[^"]*"' | cut -d'"' -f4)
echo "Request ID: $REQUEST_ID"

# Check if response contains expected fields
if echo $ANALYSIS_RESPONSE | grep -q "request_id"; then
    echo "✓ Response contains request_id"
else
    echo "✗ Response missing request_id"
fi

if echo $ANALYSIS_RESPONSE | grep -q "status"; then
    echo "✓ Response contains status"
else
    echo "✗ Response missing status"
fi

if echo $ANALYSIS_RESPONSE | grep -q "results"; then
    echo "✓ Response contains results"
else
    echo "✗ Response missing results"
fi
```

### Performance Testing

```bash
# Test concurrent requests
for i in {1..5}; do
  curl -X POST "http://localhost:8000/api/analyze" \
    -F "files=@example1/edges.csv" \
    -F "question=Quick network analysis $i" \
    -F "analysis_type=network" \
    -F "timeout=60" &
done
wait
echo "All concurrent requests completed"
```

### Monitoring and Debugging

#### Check API Logs
```bash
# View real-time logs
tail -f logs/api_$(date +%Y%m%d).log

# Check for errors
tail -f logs/errors_$(date +%Y%m%d).log
```

#### Check Docker Containers
```bash
# See running containers
docker ps

# Check Docker logs
docker-compose logs -f data-analysis-api
```

#### Sandbox Cleanup
```bash
# Check sandbox directories (should be cleaned automatically)
ls -la sandbox/

# Manual cleanup if needed
rm -rf sandbox/*
```

### Common Response Patterns

#### Successful Response Structure
```json
{
  "request_id": "uuid-string",
  "status": "success",
  "question": "your-question",
  "analysis_type": "statistical",
  "files_processed": 1,
  "execution_time": 5.23,
  "docker_execution_time": 3.45,
  "results": {
    "json": {
      "key": "value"
    },
    "images": [
      {
        "filename": "plot.png",
        "data": "base64-encoded-data"
      }
    ],
    "stdout": "Analysis completed",
    "stderr": ""
  }
}
```

#### Error Response Structure
```json
{
  "request_id": "uuid-string",
  "status": "error",
  "question": "your-question",
  "analysis_type": "statistical",
  "files_processed": 0,
  "execution_time": 1.23,
  "error": "Detailed error message",
  "results": {}
}
```

### Best Practices for API Testing

1. **Always test health endpoint first**
2. **Start with simple requests before complex ones**
3. **Use appropriate timeouts for complex analyses**
4. **Check file sizes and formats**
5. **Monitor logs for debugging**
6. **Test error conditions**
7. **Validate response structure**
8. **Use realistic analysis questions**

### Troubleshooting Test Failures

| Error | Possible Cause | Solution |
|-------|---------------|----------|
| Connection refused | API not running | Start the API: `docker-compose up` |
| File not found | Wrong file path | Check file exists: `ls example1/edges.csv` |
| Timeout | Analysis too complex | Increase timeout parameter |
| Invalid file type | Unsupported format | Check supported file types |
| Empty response | Server error | Check server logs |
| Invalid JSON | Response parsing error | Check response format |
````
