#!/usr/bin/env python3
"""
Demo script showing how to use the generate_evaluation_charts.py functionality
"""

import os
import sys

# Add the current directory to Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_evaluation_charts import load_and_evaluate_results, generate_comprehensive_charts

def demo_chart_generation():
    """
    Demo function showing how to generate evaluation charts programmatically
    """
    
    # Example usage - modify these paths as needed
    results_file = "/data1/hang/Stellantis/InternVL/notebooks/OCR/combined_prediction_results_29_06_2025.json"  # Replace with your actual results file
    output_dir = "./demo_charts"
    prefix = "demo_evaluation_report"
    exclude_fields = ['title_name', 'company_name', 'address_additional']
    
    print("🚀 Demo: Generating Evaluation Charts")
    print("="*60)
    
    # Check if results file exists
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("Please update the 'results_file' variable with your actual file path.")
        return
    
    try:
        print(f"📂 Loading results from: {results_file}")
        
        # Step 1: Load and evaluate results
        evaluation_results = load_and_evaluate_results(results_file, exclude_fields)
        
        print(f"✅ Loaded {evaluation_results['total_samples']} samples")
        print(f"📊 Evaluating {len(evaluation_results['evaluated_fields'])} fields")
        print(f"❌ Excluding {len(evaluation_results['excluded_fields'])} fields: {evaluation_results['excluded_fields']}")
        
        # Step 2: Generate comprehensive charts
        print(f"🎨 Generating charts in: {output_dir}")
        
        chart_info = generate_comprehensive_charts(
            evaluation_results,
            output_dir,
            prefix,
            results_file  # Pass the results file to enable Chart 8 (field accuracy distribution)
        )
        
        print(f"🎉 Successfully generated {chart_info['charts_generated']} charts!")
        print(f"📁 Charts saved in: {chart_info['output_directory']}")
        
        # Print summary of generated charts
        print("\n📈 Generated Charts:")
        chart_files = [
            f"{prefix}_field_performance.png",
            f"{prefix}_field_distribution.png", 
            f"{prefix}_overall_metrics.png",
            f"{prefix}_document_accuracy.png",
            f"{prefix}_confusion_matrix.png",
            f"{prefix}_f1_distribution.png",
            f"{prefix}_top_bottom_fields.png",
            f"{prefix}_field_accuracy_distribution.png"  # New chart!
        ]
        
        for i, chart_file in enumerate(chart_files, 1):
            chart_path = os.path.join(output_dir, chart_file)
            if os.path.exists(chart_path):
                print(f"   ✅ {i}. {chart_file}")
            else:
                print(f"   ❌ {i}. {chart_file} (not generated)")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def command_line_usage_examples():
    """
    Show examples of how to use the command line interface
    """
    print("\n🖥️  Command Line Usage Examples:")
    print("="*60)
    
    examples = [
        # Basic usage
        ("Basic usage", 
         "python generate_evaluation_charts.py --input results.json"),
        
        # Custom output directory
        ("Custom output directory", 
         "python generate_evaluation_charts.py --input results.json --output-dir ./my_charts"),
        
        # Custom prefix
        ("Custom filename prefix", 
         "python generate_evaluation_charts.py --input results.json --prefix my_model_eval"),
        
        # Exclude specific fields
        ("Exclude specific fields", 
         "python generate_evaluation_charts.py --input results.json --exclude-fields title_name company_name"),
        
        # Full example with all options
        ("Complete example", 
         "python generate_evaluation_charts.py \\\n    --input results.json \\\n    --output-dir ./evaluation_reports \\\n    --prefix internvl_model_eval \\\n    --exclude-fields title_name company_name address_additional")
    ]
    
    for title, command in examples:
        print(f"\n📌 {title}:")
        print(f"   {command}")


def main():
    """Main demo function"""
    print("🎯 Evaluation Charts Generation Demo")
    print("="*60)
    
    # Show command line usage examples
    command_line_usage_examples()
    
    print("\n" + "="*60)
    print("🔧 Programmatic Usage Demo")
    print("="*60)
    
    # Run the demo (uncomment the next line if you have a valid results file)
    # demo_chart_generation()
    
    print("\n💡 To run the programmatic demo:")
    print("   1. Update the 'results_file' variable in demo_chart_generation() function")
    print("   2. Uncomment the demo_chart_generation() call in main()")
    print("   3. Run this script again")
    
    print("\n🆕 New Feature: Field Accuracy Distribution Chart")
    print("="*60)
    print("📊 Chart 8 shows the percentage of images that have:")
    print("   • ≥10% correct fields")
    print("   • ≥20% correct fields") 
    print("   • ≥30% correct fields")
    print("   • ... up to 100% correct fields")
    print("\n🎨 This creates a bar chart similar to accuracy_analysis.py")
    print("📈 Includes statistics: mean, median, std dev, perfect scores, etc.")


if __name__ == "__main__":
    main() 