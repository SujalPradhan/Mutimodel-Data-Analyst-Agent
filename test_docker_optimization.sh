#!/bin/bash
# Test script to verify Docker optimization efficiency

echo "=========================================="
echo "Docker Optimization Test"
echo "=========================================="

# Test 1: Check if optimized image exists
echo "Test 1: Checking for optimized analysis image..."
CUSTOM_IMAGE=$(docker images -q data-analysis-env:latest)
if [ ! -z "$CUSTOM_IMAGE" ]; then
    echo "✅ Optimized image found: data-analysis-env:latest"
else
    echo "⚠️  Optimized image not found, will be built on first use"
fi

# Test 2: Simple analysis with timing
echo -e "\nTest 2: Running simple analysis with timing..."
echo "Creating test data..."

# Create test CSV data
cat > /tmp/test_data.csv << EOF
name,value,category
A,10,type1
B,20,type2
C,15,type1
D,25,type2
E,30,type1
EOF

# Create test question
cat > /tmp/test_question.txt << EOF
Analyze this CSV data and create a simple bar chart showing values by category.
EOF

echo "Starting analysis..."
START_TIME=$(date +%s)

# Test the API endpoint
curl -s -X POST "http://localhost:8000/api/analyze" \
  -F "files=@/tmp/test_data.csv" \
  -F "question=Analyze this CSV data and create a simple bar chart showing values by category." \
  -F "analysis_type=general" \
  -F "timeout=120" | jq '.' > /tmp/test_result.json

END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))

echo "Analysis completed in ${EXECUTION_TIME} seconds"

# Check result
if [ -f "/tmp/test_result.json" ]; then
    STATUS=$(jq -r '.status' /tmp/test_result.json)
    DOCKER_TIME=$(jq -r '.docker_execution_time' /tmp/test_result.json)
    
    if [ "$STATUS" = "success" ]; then
        echo "✅ Analysis successful"
        echo "   Docker execution time: ${DOCKER_TIME}s"
        echo "   Total time: ${EXECUTION_TIME}s"
    else
        echo "❌ Analysis failed"
        echo "Error: $(jq -r '.error' /tmp/test_result.json)"
    fi
else
    echo "❌ No result file generated"
fi

# Test 3: Check health endpoint
echo -e "\nTest 3: Checking health endpoint..."
HEALTH_RESPONSE=$(curl -s "http://localhost:8000/health")
DOCKER_AVAILABLE=$(echo "$HEALTH_RESPONSE" | jq -r '.docker.available')
CURRENT_IMAGE=$(echo "$HEALTH_RESPONSE" | jq -r '.docker.current_image')

if [ "$DOCKER_AVAILABLE" = "true" ]; then
    echo "✅ Docker available"
    echo "   Current image: $CURRENT_IMAGE"
else
    echo "❌ Docker not available"
fi

# Cleanup
rm -f /tmp/test_data.csv /tmp/test_question.txt /tmp/test_result.json

echo -e "\nTest completed!"
echo "=========================================="
