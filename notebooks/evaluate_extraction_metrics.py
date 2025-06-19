import json
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


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
    """
    Compute precision, recall, F1, and accuracy for a field
    
    Args:
        groundtruth_values: List of ground truth values
        predicted_values: List of predicted values
    
    Returns:
        Dictionary with metrics
    """
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


def evaluate_extraction_results(results_file: str) -> Dict[str, Any]:
    """
    Evaluate extraction results from JSON file
    
    Args:
        results_file: Path to JSON file with results
    
    Returns:
        Dictionary with evaluation metrics
    """
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
        'parsing_errors': parsing_errors
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


def print_field_counts(field_metrics: Dict[str, Dict[str, float]]):
    """Print field counts in a clear format"""
    print("\n" + "="*60)
    print("FIELD COUNTS SUMMARY")
    print("="*60)
    print(f"{'Field':<25} {'Total':<10} {'Non-Null':<10} {'Null':<10}")
    print("-" * 60)
    
    # Sort fields by total count (descending)
    sorted_fields = sorted(field_metrics.items(), key=lambda x: x[1]['total_samples'], reverse=True)
    
    total_all_fields = 0
    total_non_null_all = 0
    
    for field, metrics in sorted_fields:
        total = metrics['total_samples']
        non_null = metrics['non_null_samples']
        null_count = total - non_null
        
        print(f"{field:<25} {total:<10} {non_null:<10} {null_count:<10}")
        
        total_all_fields += total
        total_non_null_all += non_null
    
    print("-" * 60)
    print(f"{'TOTAL':<25} {total_all_fields:<10} {total_non_null_all:<10} {total_all_fields - total_non_null_all:<10}")
    print("="*60)


def print_evaluation_results(evaluation_results: Dict[str, Any], save_to_file: str = None):
    """Print evaluation results in a formatted way"""
    
    field_metrics = evaluation_results['field_metrics']
    overall_metrics = evaluation_results['overall_metrics']
    
    # Print header
    print("\n" + "="*140)
    print("DOCUMENT EXTRACTION EVALUATION RESULTS")
    print("="*140)
    print(f"Total samples: {evaluation_results['total_samples']}")
    print(f"Parsing errors: {evaluation_results['parsing_errors']}")
    print("="*140)
    
    # Print field counts first
    print_field_counts(field_metrics)
    
    # Print field-wise metrics
    print("\nFIELD-WISE METRICS:")
    print("-" * 140)
    print(f"{'Field':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Accuracy':<10} {'NonNull-Acc':<12} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6} {'Samples':<8} {'NonNull':<8}")
    print("-" * 140)
    
    # Sort fields by F1 score (descending)
    sorted_fields = sorted(field_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for field, metrics in sorted_fields:
        print(f"{field:<20} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
              f"{metrics['f1']:<10.3f} {metrics['accuracy']:<10.3f} {metrics['non_null_accuracy']:<12.3f} "
              f"{metrics['tp']:<6} {metrics['tn']:<6} {metrics['fp']:<6} {metrics['fn']:<6} "
              f"{metrics['total_samples']:<8} {metrics['non_null_samples']:<8}")
    
    # Print overall metrics
    print("\n" + "="*140)
    print("OVERALL METRICS:")
    print("="*140)
    print(f"Macro-averaged Precision: {overall_metrics['macro_precision']:.3f}")
    print(f"Macro-averaged Recall:    {overall_metrics['macro_recall']:.3f}")
    print(f"Macro-averaged F1:        {overall_metrics['macro_f1']:.3f}")
    print(f"Macro-averaged Accuracy:  {overall_metrics['macro_accuracy']:.3f}")
    print()
    print(f"Micro-averaged Precision: {overall_metrics['micro_precision']:.3f}")
    print(f"Micro-averaged Recall:    {overall_metrics['micro_recall']:.3f}")
    print(f"Micro-averaged F1:        {overall_metrics['micro_f1']:.3f}")
    print(f"Micro-averaged Accuracy:  {overall_metrics['micro_accuracy']:.3f}")
    print()
    print("DOCUMENT-LEVEL ACCURACY:")
    print(f"Perfect Documents:        {overall_metrics['perfect_documents']}/{overall_metrics['valid_documents']} ({overall_metrics['document_accuracy']:.1%})")
    print(f"Documents with all fields correct: {overall_metrics['document_accuracy']:.3f} ({overall_metrics['document_accuracy']:.1%})")
    
    # Print best and worst performing fields
    print("\n" + "="*140)
    print("FIELD PERFORMANCE ANALYSIS:")
    print("="*140)
    
    # Best performing fields (by F1)
    best_fields = sorted(field_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)[:5]
    print("Top 5 best performing fields (by F1):")
    for field, metrics in best_fields:
        print(f"  {field}: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
    
    # Worst performing fields (by F1)
    worst_fields = sorted(field_metrics.items(), key=lambda x: x[1]['f1'])[:5]
    print("\nTop 5 worst performing fields (by F1):")
    for field, metrics in worst_fields:
        print(f"  {field}: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
    
    # Save to CSV if requested
    if save_to_file:
        # Save detailed field-by-field metrics
        df_data = []
        for field, metrics in field_metrics.items():
            df_data.append({
                'field': field,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'accuracy': metrics['accuracy'],
                'non_null_accuracy': metrics['non_null_accuracy'],
                'tp': metrics['tp'],
                'tn': metrics['tn'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'total_samples': metrics['total_samples'],
                'non_null_samples': metrics['non_null_samples'],
                'null_samples': metrics['total_samples'] - metrics['non_null_samples'],
                'exact_matches': metrics['exact_matches'],
                'non_null_matches': metrics['non_null_matches']
            })
        
        df = pd.DataFrame(df_data)
        csv_file = save_to_file.replace('.json', '_detailed_metrics.csv')
        df.to_csv(csv_file, index=False)
        print(f"\nDetailed field metrics saved to: {csv_file}")
        
        # Save comprehensive summary with all aggregate statistics
        summary_data = []
        
        # Add all overall metrics
        for metric_name, value in overall_metrics.items():
            summary_data.append({
                'metric_type': 'overall',
                'metric_name': metric_name,
                'value': value
            })
        
        # Add aggregate statistics across all fields
        total_tp = sum(metrics['tp'] for metrics in field_metrics.values())
        total_tn = sum(metrics['tn'] for metrics in field_metrics.values())
        total_fp = sum(metrics['fp'] for metrics in field_metrics.values())
        total_fn = sum(metrics['fn'] for metrics in field_metrics.values())
        total_samples = sum(metrics['total_samples'] for metrics in field_metrics.values())
        total_non_null = sum(metrics['non_null_samples'] for metrics in field_metrics.values())
        total_exact_matches = sum(metrics['exact_matches'] for metrics in field_metrics.values())
        total_non_null_matches = sum(metrics['non_null_matches'] for metrics in field_metrics.values())
        
        aggregate_stats = [
            ('aggregate', 'total_tp', total_tp),
            ('aggregate', 'total_tn', total_tn),
            ('aggregate', 'total_fp', total_fp),
            ('aggregate', 'total_fn', total_fn),
            ('aggregate', 'total_samples', total_samples),
            ('aggregate', 'total_non_null_samples', total_non_null),
            ('aggregate', 'total_null_samples', total_samples - total_non_null),
            ('aggregate', 'total_exact_matches', total_exact_matches),
            ('aggregate', 'total_non_null_matches', total_non_null_matches),
            ('aggregate', 'number_of_fields', len(field_metrics)),
        ]
        
        for metric_type, metric_name, value in aggregate_stats:
            summary_data.append({
                'metric_type': metric_type,
                'metric_name': metric_name,
                'value': value
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = save_to_file.replace('.json', '_comprehensive_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"Comprehensive summary saved to: {summary_csv}")
        
        # Also save a simple totals CSV for quick reference
        totals_data = [{
            'total_fields': len(field_metrics),
            'total_tp': total_tp,
            'total_tn': total_tn,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'total_samples': total_samples,
            'total_non_null_samples': total_non_null,
            'total_null_samples': total_samples - total_non_null,
            'total_exact_matches': total_exact_matches,
            'total_non_null_matches': total_non_null_matches,
            'macro_precision': overall_metrics['macro_precision'],
            'macro_recall': overall_metrics['macro_recall'],
            'macro_f1': overall_metrics['macro_f1'],
            'macro_accuracy': overall_metrics['macro_accuracy'],
            'micro_precision': overall_metrics['micro_precision'],
            'micro_recall': overall_metrics['micro_recall'],
            'micro_f1': overall_metrics['micro_f1'],
            'micro_accuracy': overall_metrics['micro_accuracy'],
            'document_accuracy': overall_metrics['document_accuracy'],
            'perfect_documents': overall_metrics['perfect_documents'],
            'valid_documents': overall_metrics['valid_documents']
        }]
        
        totals_df = pd.DataFrame(totals_data)
        totals_csv = save_to_file.replace('.json', '_totals.csv')
        totals_df.to_csv(totals_csv, index=False)
        print(f"Quick totals summary saved to: {totals_csv}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate document extraction results")
    parser.add_argument('--results-file', type=str, 
                        default='train_results_vdn_fieldtext_internvl3_1b_dynamic_res_2nd_finetune_full_3beams.json',
                        help='Path to results JSON file')
    parser.add_argument('--save-csv', action='store_true', default=True,
                        help='Save detailed metrics to CSV file')
    
    args = parser.parse_args()
    
    # Check if results file exists
    import os
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        return
    
    # Evaluate results
    evaluation_results = evaluate_extraction_results(args.results_file)
    
    # Print results
    save_file = args.results_file if args.save_csv else None
    print_evaluation_results(evaluation_results, save_file)


if __name__ == '__main__':
    main() 