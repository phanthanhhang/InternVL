#!/usr/bin/env python3
"""
Analyze OCR field accuracy from validation results and generate a vertical bar chart.
Shows the percentage of images that have >10%, >20%, ..., 100% correct fields.
Uses the same field normalization logic as evaluate_extraction_metrics.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any

# Define fields to exclude from accuracy calculation
EXCLUDED_FIELDS = ['title_name', 'company_name', 'address_additional']

# Define all possible fields (same as in evaluate_extraction_metrics.py)
ALL_FIELDS = [
    'first_name', 'family_name', 'title_name', 
    'address_street', 'address_house_no', 'address_additional', 
    'address_zip', 'address_city', 
    'SV_number', 'tax_id', 
    'salary_month', 'gross_payment', 'net_payment', 'real_payment',
    'bank_account', 'bank_name', 'company_name'
]


def normalize_value(value: Any) -> str:
    """
    Normalize values for comparison (same as in evaluate_extraction_metrics.py)
    """
    if value is None or value == "null" or value == "":
        return None
    
    # Convert to string and strip whitespace
    normalized = str(value).strip()
    
    # Handle empty strings
    if normalized == "" or normalized.lower() == "null":
        return None
    
    return normalized


def load_validation_data(file_path):
    """Load the validation JSON data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def safe_json_parse(json_string: str) -> Dict[str, Any]:
    """Safely parse JSON string, return empty dict if parsing fails"""
    if not json_string or json_string.strip() == "":
        return {}
    
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON: {json_string[:100]}... Error: {e}")
        return {}


def calculate_field_accuracy(ground_truth, prediction):
    """
    Calculate the percentage of correctly predicted fields.
    Uses the same normalization logic as evaluate_extraction_metrics.py
    Dynamically excludes fields defined in EXCLUDED_FIELDS variable
    """
    if not ground_truth and not prediction:
        return 0.0
    
    # Create expected fields by excluding the fields in EXCLUDED_FIELDS
    expected_fields = [field for field in ALL_FIELDS if field not in EXCLUDED_FIELDS]
    
    correct_fields = 0
    total_fields = len(expected_fields)
    
    for field in expected_fields:
        gt_value = ground_truth.get(field)
        pred_value = prediction.get(field)
        
        # Normalize values using the same logic as evaluate_extraction_metrics.py
        gt_norm = normalize_value(gt_value)
        pred_norm = normalize_value(pred_value)
        
        # Consider field correct if normalized values match
        if gt_norm == pred_norm:
            correct_fields += 1
    
    return (correct_fields / total_fields) * 100


def analyze_accuracy(data):
    """Analyze accuracy for all images."""
    accuracy_results = []
    parsing_errors = 0
    
    for entry in data:
        gt_str = entry.get('groundtruth', '{}')
        pred_str = entry.get('prediction', '{}')
        
        ground_truth = safe_json_parse(gt_str)
        prediction = safe_json_parse(pred_str)
        
        if not ground_truth and not prediction:
            parsing_errors += 1
            continue
        
        accuracy = calculate_field_accuracy(ground_truth, prediction)
        accuracy_results.append({
            'id': entry.get('id', 'unknown'),
            'image_path': entry.get('image_path', 'unknown'),
            'accuracy': accuracy
        })
    
    if parsing_errors > 0:
        print(f"Warning: {parsing_errors} entries had parsing errors and were skipped.")
    
    return accuracy_results


def create_accuracy_distribution(accuracy_results):
    """Create accuracy distribution for different thresholds."""
    total_images = len(accuracy_results)
    
    if total_images == 0:
        return {}, 0
    
    # Define accuracy thresholds (10%, 20%, ..., 100%)
    thresholds = list(range(10, 101, 10))
    
    distribution = {}
    
    for threshold in thresholds:
        # Count images with accuracy >= threshold
        count = sum(1 for result in accuracy_results if result['accuracy'] >= threshold)
        percentage = (count / total_images) * 100
        distribution[threshold] = {
            'count': count,
            'percentage': percentage
        }
    
    return distribution, total_images


def create_bar_chart(distribution, total_images, accuracy_results, output_file='accuracy_distribution.png'):
    """Create vertical bar chart of the accuracy distribution."""
    thresholds = sorted(distribution.keys())
    percentages = [distribution[t]['percentage'] for t in thresholds]
    counts = [distribution[t]['count'] for t in thresholds]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create bars with gradient colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(thresholds)))
    bars = ax.bar(thresholds, percentages, 
                  color=colors, 
                  edgecolor='black', 
                  linewidth=1.5,
                  alpha=0.8)
    
    # Add value labels on top of each bar
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Customize the plot
    ax.set_xlabel('Minimum Field Accuracy Threshold (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Images (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Field Accuracy Distribution Across {total_images} Images\n' +
                 f'Percentage of images with ≥X% correct fields\n' +
                 f'(Using normalized field comparison)', 
                 fontsize=16, fontweight='bold', pad=25)
    
    # Set x-axis ticks
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f'{t}%' for t in thresholds], fontsize=12)
    
    # Set y-axis limits and grid
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    
    # Add summary statistics in a text box - moved to right side
    accuracy_values = [result['accuracy'] for result in accuracy_results]
    mean_acc = np.mean(accuracy_values)
    median_acc = np.median(accuracy_values)
    std_acc = np.std(accuracy_values)
    min_acc = np.min(accuracy_values)
    max_acc = np.max(accuracy_values)
    
    # Count perfect scores (100%)
    perfect_count = sum(1 for acc in accuracy_values if acc == 100.0)
    perfect_pct = (perfect_count / total_images) * 100
    
    # Move the statistics box to the right side of the plot
    summary_text = (f'Total Images: {total_images}\n'
                   f'Mean Accuracy: {mean_acc:.1f}%\n'
                   f'Median Accuracy: {median_acc:.1f}%\n'
                   f'Std Dev: {std_acc:.1f}%\n'
                   f'Range: {min_acc:.1f}% - {max_acc:.1f}%\n'
                   f'Perfect (100%): {perfect_count} ({perfect_pct:.1f}%)')
    
    ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
            fontsize=11, fontweight='bold')
    
    # Style improvements
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Bar chart saved as: {output_file}")


def print_detailed_statistics(accuracy_results, distribution):
    """Print detailed statistics."""
    print("="*80)
    print("DETAILED FIELD ACCURACY ANALYSIS")
    print("="*80)
    
    # Show field configuration
    expected_fields = [field for field in ALL_FIELDS if field not in EXCLUDED_FIELDS]
    print(f"Total Fields Available: {len(ALL_FIELDS)}")
    print(f"Fields Included in Analysis: {len(expected_fields)}")
    print(f"Fields Excluded: {len(EXCLUDED_FIELDS)}")
    print(f"Excluded Fields: {', '.join(EXCLUDED_FIELDS)}")
    print(f"Included Fields: {', '.join(expected_fields)}")
    print()
    
    total_images = len(accuracy_results)
    accuracy_values = [result['accuracy'] for result in accuracy_results]
    
    print(f"Total Images Analyzed: {total_images}")
    print(f"Mean Field Accuracy: {np.mean(accuracy_values):.2f}%")
    print(f"Median Field Accuracy: {np.median(accuracy_values):.2f}%")
    print(f"Standard Deviation: {np.std(accuracy_values):.2f}%")
    print(f"Min Accuracy: {np.min(accuracy_values):.2f}%")
    print(f"Max Accuracy: {np.max(accuracy_values):.2f}%")
    
    # Count perfect scores
    perfect_count = sum(1 for acc in accuracy_values if acc == 100.0)
    perfect_pct = (perfect_count / total_images) * 100
    print(f"Perfect Scores (100%): {perfect_count} ({perfect_pct:.1f}%)")
    
    # Count zero scores
    zero_count = sum(1 for acc in accuracy_values if acc == 0.0)
    zero_pct = (zero_count / total_images) * 100
    print(f"Zero Scores (0%): {zero_count} ({zero_pct:.1f}%)")
    
    print("\n" + "="*80)
    print("ACCURACY THRESHOLD DISTRIBUTION")
    print("="*80)
    print(f"{'Threshold':<12} {'Count':<8} {'Percentage':<12} {'Description'}")
    print("-" * 80)
    
    for threshold in sorted(distribution.keys()):
        count = distribution[threshold]['count']
        pct = distribution[threshold]['percentage']
        desc = f"≥{threshold}% fields correct"
        print(f"{threshold:>8}% {count:>8} {pct:>10.1f}% {desc}")
    
    # Show accuracy quartiles
    print("\n" + "="*80)
    print("ACCURACY QUARTILES")
    print("="*80)
    q25 = np.percentile(accuracy_values, 25)
    q50 = np.percentile(accuracy_values, 50)
    q75 = np.percentile(accuracy_values, 75)
    
    print(f"25th Percentile (Q1): {q25:.1f}%")
    print(f"50th Percentile (Q2/Median): {q50:.1f}%")
    print(f"75th Percentile (Q3): {q75:.1f}%")
    
    # Show some examples of different accuracy levels
    print("\n" + "="*80)
    print("SAMPLE IMAGES BY ACCURACY LEVEL")
    print("="*80)
    
    # Sort by accuracy for examples
    sorted_results = sorted(accuracy_results, key=lambda x: x['accuracy'])
    
    # Show examples: lowest, median, highest
    examples = [
        ("Lowest Accuracy", sorted_results[0]),
        ("Median Accuracy", sorted_results[len(sorted_results)//2]),
        ("Highest Accuracy", sorted_results[-1])
    ]
    
    for label, result in examples:
        image_name = result['image_path'].split('/')[-1] if result['image_path'] != 'unknown' else 'unknown'
        print(f"{label}: {result['accuracy']:.1f}% - {image_name}")


def main():
    """Main function to run the analysis."""
    # File path
    # json_file = "/data1/hang/Stellantis/InternVL/notebooks/vdn_chat_valid_fieldtext_ocr___vdn_fieldtext_ocr_internvl3_1b_dynamic_res_2nd_finetune_full_6_patch_448_resolution_3beams.json"
    # json_file = "/data1/hang/Stellantis/InternVL/notebooks/validation_results_vdn_fieldtext_internvl3_1b_dynamic_res_2nd_finetune_full_less_patch_more_resolution_3beams.json"
    # json_file = "/data1/hang/Stellantis/InternVL/notebooks/LLM_prediction_results/new_groundtruth_validation_results_vdn_fieldtext_internvl3_1b_dynamic_res_2nd_finetune_full_less_patch_more_resolution_3beams.json"
    # json_file = "/data1/hang/Stellantis/InternVL/notebooks/LLM_prediction_results/vdn_2606_chat_valid_fieldtext_ocr___vdn_2606_fieldtext_ocr_internvl3_1b_dynamic_res_2nd_finetune_full_6_patch_448_resolution_3beams_draft.json"
    json_file = "/data1/hang/Stellantis/InternVL/notebooks/LLM_prediction_results/vdn_ocr_chat_valid_fieldtext___vdn_2807_fieldtext_ocr_internvl3_1b_dynamic_res_2nd_finetune_full_6_patch_448_resolution_3beams_draft.json"
    print("Loading validation data...")
    try:
        data = load_validation_data(json_file)
        print(f"Loaded {len(data)} entries from validation file.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("Analyzing field accuracy...")
    accuracy_results = analyze_accuracy(data)
    
    if not accuracy_results:
        print("No valid accuracy results found.")
        return
    
    print("Creating accuracy distribution...")
    distribution, total_images = create_accuracy_distribution(accuracy_results)
    
    # Print detailed statistics
    print_detailed_statistics(accuracy_results, distribution)
    
    # Create and save the bar chart
    print("\nGenerating bar chart...")
    create_bar_chart(distribution, total_images, accuracy_results)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()