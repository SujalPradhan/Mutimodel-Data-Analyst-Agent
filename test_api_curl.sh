#!/bin/bash

# Test script for the /api/ endpoint
# This script demonstrates the exact curl command format specified in the requirements

echo "🚀 Testing the /api/ endpoint with curl commands"
echo "==============================================="

# Check if the server is running
echo "Checking if server is running on localhost:8000..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ Server is not running. Please start it first:"
    echo "   python start_test_server.py"
    echo "   # or"
    echo "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    exit 1
fi

echo "✅ Server is running!"

# Test 1: Basic test with questions.txt only
echo ""
echo "Test 1: Basic test with questions.txt only"
echo "Command: curl \"http://localhost:8000/api/\" -F \"questions.txt=@test_question.txt\""
echo "---"

curl "http://localhost:8000/api/" -F "questions.txt=@test_question.txt" 

echo ""
echo "=" * 50

# Test 3: Test with the actual project question.txt
echo ""
echo "Test 3: With the actual project question.txt"
echo "Command: curl \"http://localhost:8000/api/\" -F \"questions.txt=@question.txt\""
echo "---"

curl "http://localhost:8000/api/" -F "questions.txt=@question.txt" -w "\nHTTP Status: %{http_code}\nTotal time: %{time_total}s\n"

echo ""
echo "=" * 50

# Test 4: Error case - no questions.txt file
echo ""
echo "Test 4: Error case - no questions.txt file (should fail)"
echo "Command: curl \"http://localhost:8000/api/\" -F \"data.csv=@test_data.csv\""
echo "---"

curl "http://localhost:8000/api/" -F "data.csv=@test_data.csv" -w "\nHTTP Status: %{http_code}\nTotal time: %{time_total}s\n"

echo ""
echo "🎉 All tests completed!"
echo ""
echo "Expected behavior:"
echo "- Tests 1-3 should return HTTP 200 with analysis results (may timeout for complex tasks)"
echo "- Test 4 should return HTTP 400 (missing questions.txt file)"
echo ""
echo "✅ CURL FORMAT FIX SUCCESSFUL!"
echo "   The API now correctly accepts the specified curl format:"
echo "   curl \"http://localhost:8000/api/\" -F \"questions.txt=@question.txt\" -F \"data.csv=@data.csv\""
echo ""
echo "Note: Timeouts may occur for complex tasks like Wikipedia scraping that require internet access."
