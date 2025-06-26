#!/usr/bin/env python3
"""
Script to export wrong fields in each image to a CSV file.
Compares groundtruth and prediction data and exports mismatches.
"""

import json
import pandas as pd
import argparse
from typing import Dict, List, Any, Tuple
import os


def safe_json_parse(json_string: str) -> Dict[str, Any]:
    """Safely parse JSON string, return empty dict if parsing fails"""
    if not json_string or json_string.strip() == "":
        return {}
    
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON: {json_string[:100]}... Error: {e}")
        return {}


def normalize_value(value: Any) -> str:
    """Normalize values for comparison"""
    if value is None or value == "null" or value == "":
        return None
    
    # Convert to string and strip whitespace
    normalized = str(value).strip()
    
    # Handle empty strings
    if normalized == "" or normalized.lower() == "null":
        return None
    
    return normalized


def extract_filename_from_path(image_path: str) -> str:
    """Extract filename from image path"""
    if not image_path:
        return "unknown"
    return os.path.basename(image_path)


def find_wrong_fields(gt_data: Dict[str, Any], pred_data: Dict[str, Any]) -> List[Tuple[str, Any, Any]]:
    """
    Find fields where groundtruth differs from prediction
    
    Args:
        gt_data: Ground truth data dictionary
        pred_data: Prediction data dictionary
    
    Returns:
        List of tuples (field_name, prediction_value, gt_value) for wrong fields
    """
    wrong_fields = []
    
    # Get all unique field names from both dictionaries
    all_fields = set(gt_data.keys()) | set(pred_data.keys())
    
    for field in all_fields:
        gt_value = gt_data.get(field)
        pred_value = pred_data.get(field)
        
        # Normalize values for comparison
        gt_norm = normalize_value(gt_value)
        pred_norm = normalize_value(pred_value)
        
        # Check if values differ
        if gt_norm != pred_norm:
            wrong_fields.append((field, pred_value, gt_value))
    
    return wrong_fields


def export_wrong_fields_to_csv(results_file: str, output_csv: str = None) -> None:
    """
    Export wrong fields from predictions to CSV file
    
    Args:
        results_file: Path to JSON file with results
        output_csv: Output CSV file path (optional)
    """
    print(f"Loading results from {results_file}...")
    
    # Load results
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} results")
    
    # Prepare data for CSV
    csv_data = []
    parsing_errors = 0
    total_wrong_fields = 0
    
    for i, result in enumerate(results):
        # Get image filename
        image_path = result.get('image_path', result.get('file_name', f'image_{i}'))
        file_name = extract_filename_from_path(image_path)
        
        # Parse groundtruth and prediction
        gt_data = safe_json_parse(result.get('groundtruth', '{}'))
        pred_data = safe_json_parse(result.get('prediction', '{}'))
        
        if not gt_data and not pred_data:
            parsing_errors += 1
            continue
        
        # Find wrong fields
        wrong_fields = find_wrong_fields(gt_data, pred_data)
        
        # Add each wrong field to CSV data
        for field_name, pred_value, gt_value in wrong_fields:
            csv_data.append({
                'file_name': file_name,
                'key': field_name,
                'response': pred_value if pred_value is not None else '',
                'gt': gt_value if gt_value is not None else ''
            })
            total_wrong_fields += 1
    
    # Create DataFrame
    df = pd.DataFrame(csv_data)
    
    # Set default output filename if not provided
    if output_csv is None:
        base_name = os.path.splitext(os.path.basename(results_file))[0]
        output_csv = f"{base_name}_wrong_fields.csv"
    
    # Save to CSV
    df.to_csv(output_csv, index=True, encoding='utf-8')
    
    print(f"\nExport completed!")
    print(f"- Total images processed: {len(results)}")
    print(f"- Parsing errors: {parsing_errors}")
    print(f"- Total wrong fields found: {total_wrong_fields}")
    print(f"- Unique images with errors: {df['file_name'].nunique() if not df.empty else 0}")
    print(f"- Output saved to: {output_csv}")
    
    if not df.empty:
        print(f"\nSample of wrong fields:")
        print(df.head(10).to_string(index=False))
    else:
        print("No wrong fields found!")


def main():
    parser = argparse.ArgumentParser(description='Export wrong fields from image predictions to CSV')
    parser.add_argument('--results_file', help='Path to JSON file with prediction results')
    parser.add_argument('--output', type=str, default="wrong_fields.csv", help='Output CSV file path (optional)')
    parser.add_argument('--sample', type=int, help='Process only first N samples (for testing)')
    
    args = parser.parse_args()
    args.output = args.results_file.replace('.json', '_wrong_fields.csv')
    
    # Check if input file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Input file '{args.results_file}' not found!")
        return
    
    # If sample is specified, create a temporary subset
    if args.sample:
        print(f"Processing only first {args.sample} samples...")
        with open(args.results_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        
        sample_results = all_results[:args.sample]
        temp_file = f"temp_sample_{args.sample}.json"
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(sample_results, f, indent=2)
        
        try:
            export_wrong_fields_to_csv(temp_file, args.output)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
    else:
        export_wrong_fields_to_csv(args.results_file, args.output)


if __name__ == "__main__":
    main() 