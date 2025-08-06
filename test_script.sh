#!/bin/bash

# Data Analysis API - Comprehensive Test Script
# =============================================
# 
# This script runs all curl test cases from curl_tests.txt with proper
# error handling, logging, and 10-second delays between tests.
#
# Usage: ./test_script.sh
# 
# Prerequisites:
# 1. API must be running: docker-compose up -d
# 2. All example files must be present
# 3. Script must be run from project root: /home/sujal/Work/TDS-P2v3
#
# =============================================

set -e  # Exit on error

# Configuration
API_BASE="http://localhost:8000"
TIMEOUT_BETWEEN_TESTS=10
LOG_FILE="test_results_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_FILE="test_summary_$(date +%Y%m%d_%H%M%S).txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to print colored output
print_status() {
    local status="$1"
    local message="$2"
    
    case $status in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} $message" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Function to run a test with timeout and error handling
run_test() {
    local test_name="$1"
    local curl_cmd="$2"
    local expected_status="$3"  # Optional: expected HTTP status code
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    print_status "INFO" "Running Test $TOTAL_TESTS: $test_name"
    echo "Command: $curl_cmd" >> "$LOG_FILE"
    echo "----------------------------------------" >> "$LOG_FILE"
    
    start_time=$(date +%s)
    
    # Run curl command and capture output and HTTP status
    if output=$(timeout 300s bash -c "$curl_cmd" 2>&1); then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        # Check if output contains error indicators
        if echo "$output" | grep -q '"status":"error"' || echo "$output" | grep -q "HTTP Status: 4[0-9][0-9]\|HTTP Status: 5[0-9][0-9]"; then
            print_status "WARNING" "Test completed but returned error status ($duration seconds)"
            echo "$output" >> "$LOG_FILE"
            if [[ "$test_name" == *"ERROR"* ]]; then
                PASSED_TESTS=$((PASSED_TESTS + 1))
                print_status "SUCCESS" "Error test behaved as expected"
            else
                FAILED_TESTS=$((FAILED_TESTS + 1))
            fi
        else
            print_status "SUCCESS" "Test passed ($duration seconds)"
            echo "$output" >> "$LOG_FILE"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        fi
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        print_status "ERROR" "Test failed or timed out ($duration seconds)"
        echo "Error output: $output" >> "$LOG_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    echo "========================================" >> "$LOG_FILE"
    
    # Wait between tests (except for the last test)
    if [ $TOTAL_TESTS -lt 30 ]; then  # Adjust based on total number of tests
        print_status "INFO" "Waiting ${TIMEOUT_BETWEEN_TESTS} seconds before next test..."
        sleep $TIMEOUT_BETWEEN_TESTS
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_status "INFO" "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [ ! -f "requirements.txt" ] || [ ! -d "app" ] || [ ! -d "example1" ]; then
        print_status "ERROR" "Please run this script from the project root directory (/home/sujal/Work/TDS-P2v3)"
        exit 1
    fi
    
    # Check if API is running
    if ! curl -s "$API_BASE/health" > /dev/null 2>&1; then
        print_status "ERROR" "API is not responding at $API_BASE"
        print_status "INFO" "Please start the API with: docker-compose up -d"
        exit 1
    fi
    
    # Check example files
    for example_dir in example1 example2 example3 example4; do
        if [ ! -d "$example_dir" ]; then
            print_status "WARNING" "Example directory $example_dir not found"
        fi
    done
    
    # Check specific required files
    required_files=(
        "example1/edges.csv"
        "example1/question.txt"
        "example2/sample-sale.csv"
        "example2/question.txt"
        "example3/sample-weather.csv"
        "example3/question.txt"
        "example4/question.txt"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_status "WARNING" "Required file $file not found"
        fi
    done
    
    print_status "SUCCESS" "Prerequisites check completed"
}

# Function to generate final summary
generate_summary() {
    local end_time=$(date)
    
    cat > "$SUMMARY_FILE" << EOF
Data Analysis API Test Summary
==============================

Test Run Details:
- Start Time: $start_time_global
- End Time: $end_time
- Total Tests: $TOTAL_TESTS
- Passed: $PASSED_TESTS
- Failed: $FAILED_TESTS
- Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%

Log Files:
- Detailed Log: $LOG_FILE
- Summary: $SUMMARY_FILE

API Configuration:
- Base URL: $API_BASE
- Timeout Between Tests: ${TIMEOUT_BETWEEN_TESTS}s

Test Categories Covered:
- Health checks
- Network analysis
- Sales data analysis
- Weather data analysis
- Web scraping analysis
- Multiple file uploads
- Machine learning analysis
- Error handling
- Performance tests
- Different analysis types

EOF

    print_status "INFO" "Test summary saved to: $SUMMARY_FILE"
}

# Main execution starts here
main() {
    start_time_global=$(date)
    
    echo ""
    echo "=============================================="
    echo "  Data Analysis API - Comprehensive Test Suite"
    echo "=============================================="
    echo ""
    
    print_status "INFO" "Starting test suite at $(date)"
    print_status "INFO" "Log file: $LOG_FILE"
    print_status "INFO" "Summary file: $SUMMARY_FILE"
    
    # Check prerequisites
    check_prerequisites
    
    echo ""
    print_status "INFO" "Starting test execution..."
    echo ""
    
    # 0. HEALTH CHECKS
    run_test "Health Check" \
        'curl -s -X GET "http://localhost:8000/health" -H "accept: application/json" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "Root Endpoint Check" \
        'curl -s -X GET "http://localhost:8000/" -H "accept: application/json" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    # 1. NETWORK ANALYSIS TESTS
    run_test "Network Analysis - Example 1" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example1/edges.csv" -F "question=$(cat example1/question.txt)" -F "analysis_type=network" -F "timeout=120" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "Network Analysis - Custom Question" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example1/edges.csv" -F "question=Analyze this network and find the most important nodes. Create visualizations showing node degree distribution and network structure." -F "analysis_type=network" -F "timeout=120" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    # 2. SALES DATA ANALYSIS TESTS
    run_test "Sales Data Analysis - Example 2" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example2/sample-sale.csv" -F "question=$(cat example2/question.txt)" -F "analysis_type=statistical" -F "timeout=120" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "Sales BI Analysis" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example2/sample-sale.csv" -F "question=Perform a comprehensive business intelligence analysis. Identify top-performing regions, seasonal trends, growth patterns, and create executive dashboard visualizations." -F "analysis_type=statistical" -F "timeout=150" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    # 3. WEATHER DATA ANALYSIS TESTS
    run_test "Weather Data Analysis - Example 3" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example3/sample-weather.csv" -F "question=$(cat example3/question.txt)" -F "analysis_type=timeseries" -F "timeout=120" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "Climate Analysis" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example3/sample-weather.csv" -F "question=Analyze climate patterns and weather trends. Look for seasonal variations, extreme weather events, and long-term climate changes. Create comprehensive weather visualizations." -F "analysis_type=timeseries" -F "timeout=150" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    # 4. WEB SCRAPING ANALYSIS TESTS
    run_test "Web Scraping Analysis - Example 4" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "question=$(cat example4/question.txt)" -F "analysis_type=general" -F "timeout=300" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "Movie Data Analysis" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "question=Scrape the highest grossing films data from Wikipedia and perform comprehensive analysis: trends over time, studio performance, genre analysis, and box office predictions." -F "analysis_type=general" -F "timeout=300" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    # 5. MULTIPLE FILE ANALYSIS TESTS
    run_test "Multiple File Analysis" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example1/edges.csv" -F "files=@example2/sample-sale.csv" -F "files=@example3/sample-weather.csv" -F "question=Analyze all three datasets comprehensively. Find any interesting patterns, correlations, or insights across network data, sales data, and weather data." -F "analysis_type=general" -F "timeout=180" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "Network + Sales Correlation" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example1/edges.csv" -F "files=@example2/sample-sale.csv" -F "question=Analyze potential relationships between network connectivity patterns and sales performance." -F "analysis_type=general" -F "timeout=150" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    # 6. MACHINE LEARNING ANALYSIS TESTS
    run_test "ML Analysis - Sales Clustering" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example2/sample-sale.csv" -F "question=Perform machine learning analysis on the sales data. Apply clustering algorithms to group similar sales patterns and identify customer segments." -F "analysis_type=ml" -F "timeout=180" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "ML Analysis - Weather Prediction" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example3/sample-weather.csv" -F "question=Build machine learning models to predict weather patterns using temperature and precipitation data." -F "analysis_type=ml" -F "timeout=180" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    # 7. ERROR HANDLING TESTS
    run_test "ERROR - No Files" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "question=Analyze this data" -F "analysis_type=statistical" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "ERROR - Empty Question" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example1/edges.csv" -F "question=" -F "analysis_type=statistical" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    # Create and test invalid file
    run_test "ERROR - Invalid File Type" \
        'echo "test content" > test_invalid.badext && curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@test_invalid.badext" -F "question=Analyze this file" -F "analysis_type=general" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n" && rm -f test_invalid.badext'
    
    run_test "ERROR - Very Short Timeout" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example1/edges.csv" -F "question=Perform detailed network analysis" -F "analysis_type=network" -F "timeout=1" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    # 8. PERFORMANCE TESTS
    run_test "Performance - Complex Analysis" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example2/sample-sale.csv" -F "question=Perform comprehensive business analysis with statistical measures, time series decomposition, predictive models, and multiple visualizations." -F "analysis_type=statistical" -F "timeout=300" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    # 9. DIFFERENT ANALYSIS TYPES
    run_test "Analysis Type - Statistical" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example2/sample-sale.csv" -F "question=Perform detailed statistical analysis with hypothesis testing and distribution analysis" -F "analysis_type=statistical" -F "timeout=120" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "Analysis Type - General" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example2/sample-sale.csv" -F "question=Perform general exploratory data analysis with data quality assessment" -F "analysis_type=general" -F "timeout=120" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "Analysis Type - TimeSeries" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example3/sample-weather.csv" -F "question=Focus on time series analysis with trend detection and forecasting" -F "analysis_type=timeseries" -F "timeout=120" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    # 10. COMPLEXITY TESTS
    run_test "Simple Question" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example1/edges.csv" -F "question=How many nodes are in this network?" -F "analysis_type=network" -F "timeout=60" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "Moderate Complexity" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example2/sample-sale.csv" -F "question=What are the top 3 sales regions and their performance trends?" -F "analysis_type=statistical" -F "timeout=90" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "High Complexity" \
        'curl -s -X POST "http://localhost:8000/api/analyze" -F "files=@example3/sample-weather.csv" -F "question=Build a comprehensive weather prediction model with seasonal decomposition, trend analysis, and anomaly detection." -F "analysis_type=timeseries" -F "timeout=200" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    # Generate final summary
    echo ""
    print_status "INFO" "All tests completed!"
    print_status "INFO" "Generating final summary..."
    
    generate_summary
    
    echo ""
    echo "=============================================="
    echo "           TEST SUITE SUMMARY"
    echo "=============================================="
    echo "Total Tests: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    echo "Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
    echo ""
    echo "Log Files:"
    echo "- Detailed: $LOG_FILE"
    echo "- Summary: $SUMMARY_FILE"
    echo ""
    
    if [ $FAILED_TESTS -eq 0 ]; then
        print_status "SUCCESS" "All tests passed! 🎉"
        exit 0
    else
        print_status "WARNING" "Some tests failed. Check the log files for details."
        exit 1
    fi
}

# Trap to handle script interruption
trap 'print_status "ERROR" "Script interrupted by user"; exit 130' INT TERM

# Run main function
main "$@"
