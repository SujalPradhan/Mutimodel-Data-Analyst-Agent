#!/usr/bin/env python3
"""
Demonstration of Phase 4 utility functions integration
"""
import sys
import os
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from utils import infer_task_type, preview_file

def demonstrate_task_inference():
    """Demonstrate task type inference with various questions."""
    print("🧠 TASK INFERENCE DEMONSTRATION")
    print("=" * 50)
    
    sample_questions = [
        "What are the main trends and patterns in this dataset?",
        "Find the shortest path between all nodes in the network",
        "Calculate correlation matrix and identify outliers",
        "Analyze seasonal patterns in the time series data",
        "Scrape product information from e-commerce pages",
        "Create an interactive dashboard with filters",
        "Show network centrality measures and community detection",
        "Perform regression analysis and hypothesis testing",
        "Forecast future values using ARIMA model",
        "Extract structured data from HTML tables"
    ]
    
    for question in sample_questions:
        task_type = infer_task_type(question)
        emoji = {
            "graph": "🕸️",
            "statistical": "📊", 
            "timeseries": "📈",
            "scrape": "🕷️",
            "custom": "🎨"
        }.get(task_type, "❓")
        
        print(f"{emoji} {task_type.upper()}: {question}")
    
    print()

def demonstrate_file_preview():
    """Demonstrate file preview functionality."""
    print("📁 FILE PREVIEW DEMONSTRATION")
    print("=" * 50)
    
    # Preview the sample data file
    sample_file = Path("examples/sample_data.csv")
    if sample_file.exists():
        preview = preview_file(sample_file, max_lines=3)
        
        print(f"📄 File: {preview['filename']}")
        print(f"📏 Size: {preview['size_formatted']}")
        print(f"🏷️  Type: {preview.get('file_type', 'unknown')}")
        
        if 'metadata' in preview:
            metadata = preview['metadata']
            if 'columns' in metadata:
                print(f"📋 Columns ({metadata['num_columns']}): {', '.join(metadata['columns'])}")
        
        print("👀 Preview:")
        content = preview.get('content_preview', '')
        for i, line in enumerate(content.split('\n')[:5]):
            print(f"   {i+1:2d}│ {line}")
    else:
        print(f"❌ Sample file not found: {sample_file}")
    
    print()

def demonstrate_integration_workflow():
    """Show how these functions work together in a typical workflow."""
    print("🔄 INTEGRATION WORKFLOW DEMONSTRATION")
    print("=" * 50)
    
    # Simulate incoming request
    question = "What are the main trends and patterns in this dataset? Please provide a statistical summary and identify any outliers or interesting relationships between variables."
    file_path = Path("examples/sample_data.csv")
    
    print("1️⃣ Incoming Analysis Request:")
    print(f"   Question: {question[:80]}...")
    print(f"   File: {file_path.name}")
    print()
    
    # Step 1: Infer task type
    task_type = infer_task_type(question)
    print(f"2️⃣ Task Type Inference: {task_type.upper()}")
    
    task_descriptions = {
        "statistical": "Will focus on descriptive statistics, correlations, and outlier detection",
        "graph": "Will analyze network structure and graph metrics",
        "timeseries": "Will perform temporal analysis and forecasting",
        "scrape": "Will extract data from web sources",
        "custom": "Will perform general data analysis"
    }
    print(f"   → {task_descriptions.get(task_type, 'Unknown task type')}")
    print()
    
    # Step 2: Preview file
    if file_path.exists():
        preview = preview_file(file_path, max_lines=3)
        print("3️⃣ File Analysis:")
        print(f"   → File type: {preview.get('file_type', 'unknown')}")
        print(f"   → Size: {preview.get('size_formatted', 'unknown')}")
        
        if 'metadata' in preview and 'columns' in preview['metadata']:
            columns = preview['metadata']['columns']
            print(f"   → Data columns: {', '.join(columns)}")
            print(f"   → Suitable for {task_type} analysis: ✅")
        print()
    
    # Step 3: Analysis strategy
    print("4️⃣ Analysis Strategy:")
    if task_type == "statistical":
        print("   → Generate descriptive statistics")
        print("   → Calculate correlation matrix") 
        print("   → Identify outliers using IQR method")
        print("   → Create distribution plots")
        print("   → Perform significance tests")
    elif task_type == "graph":
        print("   → Build network from data relationships")
        print("   → Calculate centrality measures")
        print("   → Detect communities")
        print("   → Visualize network structure")
    elif task_type == "timeseries":
        print("   → Parse temporal columns")
        print("   → Decompose time series")
        print("   → Analyze trends and seasonality")
        print("   → Generate forecasts")
    elif task_type == "scrape":
        print("   → Parse HTML structure")
        print("   → Extract relevant data elements")
        print("   → Clean and structure data")
        print("   → Save to structured format")
    else:
        print("   → Perform exploratory data analysis")
        print("   → Generate summary visualizations")
        print("   → Identify interesting patterns")
        print("   → Create custom analysis based on question")
    
    print("\n✅ Ready for code generation and execution!")

if __name__ == "__main__":
    print("🚀 PHASE 4 - TASK INFERENCE & UTILITIES DEMO")
    print("=" * 60)
    print()
    
    demonstrate_task_inference()
    demonstrate_file_preview()
    demonstrate_integration_workflow()
    
    print("🎉 Phase 4 implementation complete!")
    print("New utilities are ready for integration with the main API.")
