#!/usr/bin/env python3
"""
Standalone script to generate evaluation report charts from JSON results
"""

import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict


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


def compute_field_metrics(groundtruth_values: List[Any], predicted_values: List[Any]) -> Dict[str, float]:
    """Compute precision, recall, F1, and accuracy for a field"""
    if len(groundtruth_values) != len(predicted_values):
        raise ValueError("Groundtruth and predicted values must have same length")
    
    # Normalize values
    gt_norm = [normalize_value(v) for v in groundtruth_values]
    pred_norm = [normalize_value(v) for v in predicted_values]
    
    # Compute confusion matrix components
    tp = sum(1 for gt, pred in zip(gt_norm, pred_norm) 
             if gt is not None and pred is not None and gt == pred)
    
    tn = sum(1 for gt, pred in zip(gt_norm, pred_norm) 
             if gt is None and pred is None)
    
    fp = sum(1 for gt, pred in zip(gt_norm, pred_norm) 
             if gt is None and pred is not None)
    
    fn = sum(1 for gt, pred in zip(gt_norm, pred_norm) 
             if gt is not None and pred is None)
    
    # Compute exact matches (including null matches)
    exact_matches = sum(1 for gt, pred in zip(gt_norm, pred_norm) if gt == pred)
    
    # Compute non-null exact matches
    non_null_gt = [gt for gt in gt_norm if gt is not None]
    non_null_matches = sum(1 for gt, pred in zip(gt_norm, pred_norm) 
                          if gt is not None and gt == pred)
    
    total = len(gt_norm)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = exact_matches / total if total > 0 else 0.0
    
    # Non-null accuracy (only considering non-null ground truth)
    non_null_accuracy = non_null_matches / len(non_null_gt) if len(non_null_gt) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'non_null_accuracy': non_null_accuracy,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total_samples': total,
        'non_null_samples': len(non_null_gt),
        'exact_matches': exact_matches,
        'non_null_matches': non_null_matches
    }


def load_and_evaluate_results(results_file: str, exclude_fields: List[str] = None) -> Dict[str, Any]:
    """Load results and compute evaluation metrics"""
    print(f"Loading results from {results_file}...")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} results")
    
    # Extract all field values
    field_data = defaultdict(lambda: {'groundtruth': [], 'predicted': []})
    
    # Define expected fields
    expected_fields = [
        'first_name', 'family_name', 'title_name', 
        'address_street', 'address_house_no', 'address_additional', 
        'address_zip', 'address_city', 
        'SV_number', 'tax_id', 
        'salary_month', 'gross_payment', 'net_payment', 'real_payment',
        'bank_account', 'bank_name', 'company_name'
    ]
    
    # Filter out excluded fields
    if exclude_fields:
        exclude_fields_set = set(exclude_fields)
        expected_fields = [field for field in expected_fields if field not in exclude_fields_set]
        print(f"Excluding fields from evaluation: {exclude_fields}")
        print(f"Evaluating {len(expected_fields)} fields: {expected_fields}")
    
    # Parse results and track document-level accuracy
    parsing_errors = 0
    perfect_documents = 0
    valid_documents = 0
    
    for i, result in enumerate(results):
        # Parse groundtruth
        gt_data = safe_json_parse(result.get('groundtruth', '{}'))
        pred_data = safe_json_parse(result.get('prediction', '{}'))
        
        if not gt_data and not pred_data:
            parsing_errors += 1
            continue
        
        valid_documents += 1
        
        # Check if all fields are correct for this document
        all_fields_correct = True
        
        # Extract field values
        for field in expected_fields:
            gt_value = gt_data.get(field)
            pred_value = pred_data.get(field)
            
            field_data[field]['groundtruth'].append(gt_value)
            field_data[field]['predicted'].append(pred_value)
            
            # Check if this field is correct (normalize for comparison)
            gt_norm = normalize_value(gt_value)
            pred_norm = normalize_value(pred_value)
            
            if gt_norm != pred_norm:
                all_fields_correct = False
        
        if all_fields_correct:
            perfect_documents += 1
    
    if parsing_errors > 0:
        print(f"Warning: {parsing_errors} results had parsing errors")
    
    # Compute metrics for each field
    field_metrics = {}
    for field in expected_fields:
        if field in field_data:
            metrics = compute_field_metrics(
                field_data[field]['groundtruth'],
                field_data[field]['predicted']
            )
            field_metrics[field] = metrics
    
    # Compute overall metrics
    overall_metrics = compute_overall_metrics(field_metrics)
    
    # Add document-level accuracy
    document_accuracy = perfect_documents / valid_documents if valid_documents > 0 else 0.0
    overall_metrics['document_accuracy'] = document_accuracy
    overall_metrics['perfect_documents'] = perfect_documents
    overall_metrics['valid_documents'] = valid_documents
    
    return {
        'field_metrics': field_metrics,
        'overall_metrics': overall_metrics,
        'total_samples': len(results),
        'parsing_errors': parsing_errors,
        'excluded_fields': exclude_fields or [],
        'evaluated_fields': expected_fields
    }


def compute_overall_metrics(field_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute overall metrics across all fields"""
    
    # Macro-averaged metrics (average across fields)
    macro_precision = np.mean([metrics['precision'] for metrics in field_metrics.values()])
    macro_recall = np.mean([metrics['recall'] for metrics in field_metrics.values()])
    macro_f1 = np.mean([metrics['f1'] for metrics in field_metrics.values()])
    macro_accuracy = np.mean([metrics['accuracy'] for metrics in field_metrics.values()])
    
    # Micro-averaged metrics (aggregate across all field instances)
    total_tp = sum(metrics['tp'] for metrics in field_metrics.values())
    total_tn = sum(metrics['tn'] for metrics in field_metrics.values())
    total_fp = sum(metrics['fp'] for metrics in field_metrics.values())
    total_fn = sum(metrics['fn'] for metrics in field_metrics.values())
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    micro_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0
    
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_accuracy': macro_accuracy,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'micro_accuracy': micro_accuracy
    }


def calculate_document_field_accuracy(groundtruth_data: Dict[str, Any], predicted_data: Dict[str, Any], 
                                    evaluated_fields: List[str]) -> float:
    """
    Calculate the percentage of correctly predicted fields for a single document.
    Uses the same normalization logic as field metrics calculation.
    """
    if not groundtruth_data and not predicted_data:
        return 0.0
    
    correct_fields = 0
    total_fields = len(evaluated_fields)
    
    for field in evaluated_fields:
        gt_value = groundtruth_data.get(field)
        pred_value = predicted_data.get(field)
        
        # Normalize values using the same logic
        gt_norm = normalize_value(gt_value)
        pred_norm = normalize_value(pred_value)
        
        # Consider field correct if normalized values match
        if gt_norm == pred_norm:
            correct_fields += 1
    
    return (correct_fields / total_fields) * 100 if total_fields > 0 else 0.0


def create_accuracy_distribution_data(results_file: str, evaluated_fields: List[str]) -> List[float]:
    """
    Calculate field accuracy for each document in the results file.
    Returns list of accuracy percentages.
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    accuracy_results = []
    parsing_errors = 0
    
    for result in results:
        # Parse groundtruth and prediction
        gt_data = safe_json_parse(result.get('groundtruth', '{}'))
        pred_data = safe_json_parse(result.get('prediction', '{}'))
        
        if not gt_data and not pred_data:
            parsing_errors += 1
            continue
        
        accuracy = calculate_document_field_accuracy(gt_data, pred_data, evaluated_fields)
        accuracy_results.append(accuracy)
    
    if parsing_errors > 0:
        print(f"Warning: {parsing_errors} documents had parsing errors and were skipped from accuracy distribution.")
    
    return accuracy_results


def generate_comprehensive_charts(evaluation_results: Dict[str, Any], output_dir: str = ".", 
                                prefix: str = "evaluation_report", results_file: str = None):
    """Generate comprehensive evaluation report charts"""
    
    field_metrics = evaluation_results['field_metrics']
    overall_metrics = evaluation_results['overall_metrics']
    
    # Setup plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    save_prefix = os.path.join(output_dir, prefix)
    
    # Prepare data for plotting
    fields = list(field_metrics.keys())
    precision_scores = [field_metrics[f]['precision'] for f in fields]
    recall_scores = [field_metrics[f]['recall'] for f in fields]
    f1_scores = [field_metrics[f]['f1'] for f in fields]
    
    # Sort by F1 score
    sorted_indices = np.argsort(f1_scores)[::-1]
    fields_sorted = [fields[i] for i in sorted_indices]
    precision_sorted = [precision_scores[i] for i in sorted_indices]
    recall_sorted = [recall_scores[i] for i in sorted_indices]
    f1_sorted = [f1_scores[i] for i in sorted_indices]
    
    # Chart 1: Field Performance Metrics
    plt.figure(figsize=(16, 10))
    x = np.arange(len(fields_sorted))
    width = 0.25
    
    bars1 = plt.bar(x - width, precision_sorted, width, label='Precision', alpha=0.8)
    bars2 = plt.bar(x, recall_sorted, width, label='Recall', alpha=0.8)
    bars3 = plt.bar(x + width, f1_sorted, width, label='F1 Score', alpha=0.8)
    
    # Add value labels on top of each bar
    for i, (bar1, bar2, bar3, prec, rec, f1) in enumerate(zip(bars1, bars2, bars3, precision_sorted, recall_sorted, f1_sorted)):
        # Precision bar
        plt.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.01,
                f'{prec:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9, rotation=90)
        
        # Recall bar
        plt.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.01,
                f'{rec:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9, rotation=90)
        
        # F1 Score bar
        plt.text(bar3.get_x() + bar3.get_width()/2., bar3.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9, rotation=90)
    
    plt.xlabel('Fields', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Field-wise Performance Metrics', fontsize=16, fontweight='bold')
    plt.xticks(x, fields_sorted, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_field_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 2: Field Data Distribution
    plt.figure(figsize=(16, 8))
    total_samples = [field_metrics[f]['total_samples'] for f in fields_sorted]
    non_null_samples = [field_metrics[f]['non_null_samples'] for f in fields_sorted]
    null_samples = [total - non_null for total, non_null in zip(total_samples, non_null_samples)]
    
    plt.bar(fields_sorted, non_null_samples, label='Non-Null Values', alpha=0.8)
    plt.bar(fields_sorted, null_samples, bottom=non_null_samples, label='Null Values', alpha=0.8)
    
    plt.xlabel('Fields', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Field Data Distribution (Null vs Non-Null)', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_field_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 3: Overall Metrics Comparison
    plt.figure(figsize=(12, 8))
    metric_names = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    macro_values = [
        overall_metrics['macro_precision'],
        overall_metrics['macro_recall'], 
        overall_metrics['macro_f1'],
        overall_metrics['macro_accuracy']
    ]
    micro_values = [
        overall_metrics['micro_precision'],
        overall_metrics['micro_recall'],
        overall_metrics['micro_f1'], 
        overall_metrics['micro_accuracy']
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    plt.bar(x - width/2, macro_values, width, label='Macro-averaged', alpha=0.8)
    plt.bar(x + width/2, micro_values, width, label='Micro-averaged', alpha=0.8)
    
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Overall Performance Metrics Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x, metric_names)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (macro, micro) in enumerate(zip(macro_values, micro_values)):
        plt.text(i - width/2, macro + 0.01, f'{macro:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, micro + 0.01, f'{micro:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_overall_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 4: Document-Level Accuracy Pie Chart
    plt.figure(figsize=(10, 8))
    perfect_docs = overall_metrics['perfect_documents']
    imperfect_docs = overall_metrics['valid_documents'] - perfect_docs
    
    labels = [f'Perfect Documents\n({perfect_docs})', f'Imperfect Documents\n({imperfect_docs})']
    sizes = [perfect_docs, imperfect_docs]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 12})
    plt.title(f'Document-Level Accuracy\n({overall_metrics["document_accuracy"]:.1%} Perfect Documents)', 
              fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.savefig(f'{save_prefix}_document_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 5: Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    total_tp = sum(field_metrics[f]['tp'] for f in field_metrics)
    total_tn = sum(field_metrics[f]['tn'] for f in field_metrics)
    total_fp = sum(field_metrics[f]['fp'] for f in field_metrics)
    total_fn = sum(field_metrics[f]['fn'] for f in field_metrics)
    
    cm = np.array([[total_tp, total_fn], [total_fp, total_tn]])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Aggregated Confusion Matrix (All Fields)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 6: Performance Distribution
    plt.figure(figsize=(12, 8))
    f1_values = [field_metrics[f]['f1'] for f in fields]
    
    plt.hist(f1_values, bins=10, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(f1_values), color='red', linestyle='--', 
                label=f'Mean F1: {np.mean(f1_values):.3f}')
    plt.xlabel('F1 Score', fontsize=12)
    plt.ylabel('Number of Fields', fontsize=12)
    plt.title('Distribution of Field F1 Scores', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_f1_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 7: Top vs Bottom Performers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 5 fields
    top_5_fields = fields_sorted[:5]
    top_5_f1 = [field_metrics[f]['f1'] for f in top_5_fields]
    bars1 = ax1.bar(range(len(top_5_fields)), top_5_f1, color='green', alpha=0.7)
    ax1.set_title('Top 5 Performing Fields (F1 Score)', fontweight='bold')
    ax1.set_xticks(range(len(top_5_fields)))
    ax1.set_xticklabels(top_5_fields, rotation=45, ha='right')
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, f1 in zip(bars1, top_5_f1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{f1:.3f}', ha='center', va='bottom')
    
    # Bottom 5 fields
    bottom_5_fields = fields_sorted[-5:]
    bottom_5_f1 = [field_metrics[f]['f1'] for f in bottom_5_fields]
    bars2 = ax2.bar(range(len(bottom_5_fields)), bottom_5_f1, color='red', alpha=0.7)
    ax2.set_title('Bottom 5 Performing Fields (F1 Score)', fontweight='bold')
    ax2.set_xticks(range(len(bottom_5_fields)))
    ax2.set_xticklabels(bottom_5_fields, rotation=45, ha='right')
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, f1 in zip(bars2, bottom_5_f1):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{f1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_top_bottom_fields.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 8: Field Accuracy Distribution (similar to accuracy_analysis.py)
    if results_file:
        print("Generating field accuracy distribution chart...")
        accuracy_data = create_accuracy_distribution_data(results_file, evaluation_results['evaluated_fields'])
        
        if accuracy_data:
            plt.figure(figsize=(14, 10))
            
            # Define accuracy thresholds (10%, 20%, ..., 100%)
            thresholds = list(range(10, 101, 10))
            total_images = len(accuracy_data)
            
            # Calculate distribution
            distribution = {}
            for threshold in thresholds:
                count = sum(1 for acc in accuracy_data if acc >= threshold)
                percentage = (count / total_images) * 100
                distribution[threshold] = {'count': count, 'percentage': percentage}
            
            # Create the chart
            percentages = [distribution[t]['percentage'] for t in thresholds]
            counts = [distribution[t]['count'] for t in thresholds]
            
            # Create bars with gradient colors
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(thresholds)))
            bars = plt.bar(thresholds, percentages, 
                          color=colors, 
                          edgecolor='black', 
                          linewidth=1.5,
                          alpha=0.8)
            
            # Add value labels on top of each bar
            for bar, count, pct in zip(bars, counts, percentages):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{count}\n({pct:.1f}%)',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Customize the plot
            plt.xlabel('Minimum Field Accuracy Threshold (%)', fontsize=14, fontweight='bold')
            plt.ylabel('Percentage of Images (%)', fontsize=14, fontweight='bold')
            plt.title(f'Field Accuracy Distribution Across {total_images} Images\n' +
                     f'Percentage of images with ≥X% correct fields\n' +
                     f'(Using normalized field comparison)', 
                     fontsize=16, fontweight='bold', pad=25)
            
            # Set x-axis ticks
            plt.xticks(thresholds, [f'{t}%' for t in thresholds], fontsize=12)
            
            # Set y-axis limits and grid
            plt.ylim(0, 105)
            plt.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
            plt.gca().set_axisbelow(True)
            
            # Add summary statistics in a text box
            mean_acc = np.mean(accuracy_data)
            median_acc = np.median(accuracy_data)
            std_acc = np.std(accuracy_data)
            min_acc = np.min(accuracy_data)
            max_acc = np.max(accuracy_data)
            
            # Count perfect scores (100%)
            perfect_count = sum(1 for acc in accuracy_data if acc == 100.0)
            perfect_pct = (perfect_count / total_images) * 100
            
            summary_text = (f'Total Images: {total_images}\n'
                           f'Mean Accuracy: {mean_acc:.1f}%\n'
                           f'Median Accuracy: {median_acc:.1f}%\n'
                           f'Std Dev: {std_acc:.1f}%\n'
                           f'Range: {min_acc:.1f}% - {max_acc:.1f}%\n'
                           f'Perfect (100%): {perfect_count} ({perfect_pct:.1f}%)')
            
            plt.gca().text(0.98, 0.98, summary_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                    fontsize=11, fontweight='bold')
            
            # Style improvements
            plt.gca().tick_params(axis='both', which='major', labelsize=12)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_linewidth(1.5)
            plt.gca().spines['bottom'].set_linewidth(1.5)
            
            plt.tight_layout()
            plt.savefig(f'{save_prefix}_field_accuracy_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Generate summary report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    total_charts = 8 if results_file else 7
    
    print(f"\n{'='*80}")
    print("📊 EVALUATION REPORT CHARTS GENERATED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"⏰ Timestamp: {timestamp}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📈 Generated {total_charts} charts:")
    print(f"   1. 📊 {prefix}_field_performance.png - Field-wise performance metrics")
    print(f"   2. 📊 {prefix}_field_distribution.png - Field data distribution")  
    print(f"   3. 📊 {prefix}_overall_metrics.png - Overall metrics comparison")
    print(f"   4. 🥧 {prefix}_document_accuracy.png - Document-level accuracy")
    print(f"   5. 🔥 {prefix}_confusion_matrix.png - Aggregated confusion matrix")
    print(f"   6. 📈 {prefix}_f1_distribution.png - F1 score distribution")
    print(f"   7. 🏆 {prefix}_top_bottom_fields.png - Best/worst performing fields")
    if results_file:
        print(f"   8. 📊 {prefix}_field_accuracy_distribution.png - Field accuracy distribution")
    print(f"{'='*80}")
    
    return {
        'charts_generated': total_charts,
        'output_directory': output_dir,
        'timestamp': timestamp
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation report charts from JSON results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate charts with default settings
  python generate_evaluation_charts.py --input results.json
  
  # Generate charts with custom output directory
  python generate_evaluation_charts.py --input results.json --output-dir ./charts
  
  # Exclude specific fields from evaluation
  python generate_evaluation_charts.py --input results.json --exclude-fields title_name company_name
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input JSON results file')
    parser.add_argument('--output-dir', '-o', type=str, default='./evaluation_charts',
                        help='Output directory for charts (default: ./evaluation_charts)')
    parser.add_argument('--prefix', '-p', type=str, default='evaluation_report',
                        help='Filename prefix for generated charts (default: evaluation_report)')
    parser.add_argument('--exclude-fields', nargs='+', 
                        default=['title_name', 'company_name', 'address_additional'],
                        help='List of fields to exclude from evaluation')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    try:
        # Load and evaluate results
        print(f"🔄 Loading and evaluating results from {args.input}...")
        evaluation_results = load_and_evaluate_results(args.input, args.exclude_fields)
        
        # Generate charts
        print(f"🎨 Generating charts...")
        chart_info = generate_comprehensive_charts(
            evaluation_results, 
            args.output_dir, 
            args.prefix,
            args.input
        )
        
        print(f"✅ Successfully generated {chart_info['charts_generated']} charts!")
        
    except Exception as e:
        print(f"❌ Error generating charts: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 