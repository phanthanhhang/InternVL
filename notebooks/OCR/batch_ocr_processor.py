#!/usr/bin/env python3
"""
Batch OCR Processor for VDN documents.
Runs PaddleOCR on all images from annotation file and formats results for downstream processing.
"""

import sys
sys.path.append("/data1/hang/Stellantis/PaddleOCR")

import json
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse
from paddleocr import PaddleOCR


class BatchOCRProcessor:
    def __init__(self, 
                 annotation_file: str,
                 local_image_dir: str,
                 output_dir: str = "./ocr_results"):
        """
        Initialize the batch OCR processor.
        
        Args:
            annotation_file: Path to annotation JSON file (e.g., processed_vdn_data_2807.json)
            local_image_dir: Local directory where images are stored
            output_dir: Directory to save OCR results
        """
        self.annotation_file = annotation_file
        self.local_image_dir = Path(local_image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PaddleOCR with the same configuration as doc_ocr.py
        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_dir="/data1/hang/Stellantis/PaddleOCR/output/export_model_german_custom_PP-OCRv5_server_rec_18072025",
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
        print("PaddleOCR initialized successfully")
        
        # Load annotation data
        print(f"Loading annotations from {annotation_file}")
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        print(f"Loaded annotations for {len(self.annotations)} images")
        
    def correct_image_orientation(self, img: np.ndarray, angle: int) -> np.ndarray:
        """
        Correct image orientation based on detected angle.
        
        Args:
            img: Input image
            angle: Detected rotation angle
            
        Returns:
            Corrected image
        """
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            return img
    
    def convert_paddleocr_to_mock_format(self, ocr_result: Dict) -> List[Dict]:
        """
        Convert PaddleOCR result to mock_ocr_results format.
        
        Args:
            ocr_result: Single OCR result from PaddleOCR
            
        Returns:
            List of formatted OCR results with x, y, width, height, text
        """
        formatted_results = []
        
        if not ocr_result:
            return formatted_results
            
        rec_boxes = ocr_result.get('rec_boxes', [])
        rec_texts = ocr_result.get('rec_texts', [])
        rec_scores = ocr_result.get('rec_scores', [])
        
        for i, box in enumerate(rec_boxes):
            if i < len(rec_texts) and i < len(rec_scores):
                # rec_boxes format: [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                
                formatted_result = {
                    'text': rec_texts[i],
                    'x': int(x_min),
                    'y': int(y_min), 
                    'width': int(width),
                    'height': int(height),
                    'confidence': float(rec_scores[i])
                }
                formatted_results.append(formatted_result)
        
        return formatted_results
    
    def process_single_image(self, image_path: str, save_visualization: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Process a single image with PaddleOCR.
        
        Args:
            image_path: Path to the image file
            save_visualization: Whether to save visualization images
            
        Returns:
            Tuple of (formatted_ocr_results, raw_ocr_result)
        """
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            return [], {}
            
        try:
            # Run OCR
            result = self.ocr.predict(input=image_path, text_rec_score_thresh=0.7)
            
            if not result:
                print(f"Warning: No OCR results for {image_path}")
                return [], {}
                
            # Get the first result (should be only one for single image)
            ocr_result = result[0]
            
            # Convert to mock format
            formatted_results = self.convert_paddleocr_to_mock_format(ocr_result)
            
            # Optionally save visualization
            if save_visualization:
                self.save_visualization(image_path, ocr_result, formatted_results)
                
            return formatted_results, ocr_result
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return [], {}
    
    def save_visualization(self, image_path: str, ocr_result: Dict, formatted_results: List[Dict]) -> None:
        """
        Save visualization of OCR results similar to doc_ocr.py.
        
        Args:
            image_path: Original image path
            ocr_result: Raw OCR result from PaddleOCR
            formatted_results: Formatted OCR results
        """
        try:
            input_img = cv2.imread(image_path)
            if input_img is None:
                return
                
            # Correct orientation if needed
            angle = int(ocr_result.get('doc_preprocessor_res', {}).get('angle', 0))
            corrected_img = self.correct_image_orientation(input_img, angle)
            
            # Draw bounding boxes
            img_with_boxes = corrected_img.copy()
            
            for result in formatted_results:
                x, y, width, height = result['x'], result['y'], result['width'], result['height']
                text = result['text']
                confidence = result['confidence']
                
                # Draw rectangle
                cv2.rectangle(img_with_boxes, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Add text with confidence
                display_text = f"{text} ({confidence:.2f})"
                font_scale = 0.4
                thickness = 1
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Draw background rectangle
                cv2.rectangle(img_with_boxes, 
                             (x, y - text_height - baseline - 5), 
                             (x + text_width, y - 2), 
                             (0, 255, 0), -1)
                
                # Draw text
                cv2.putText(img_with_boxes, display_text, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            
            # Save visualization
            filename = os.path.basename(image_path)
            output_path = self.output_dir / f"{filename}_ocr_visualization.jpg"
            cv2.imwrite(str(output_path), img_with_boxes)
            
        except Exception as e:
            print(f"Error saving visualization for {image_path}: {str(e)}")
    
    def process_all_images(self, save_visualizations: bool = False, max_images: int = None, 
                          save_interval: int = 100) -> Dict:
        """
        Process all images in the annotation file.
        
        Args:
            save_visualizations: Whether to save visualization images
            max_images: Maximum number of images to process (None for all)
            save_interval: Save results every N images (default: 100)
            
        Returns:
            Dictionary mapping image paths to OCR results
        """
        all_ocr_results = {}
        processed_count = 0
        
        # Check if there's an existing results file to resume from
        temp_results_file = self.output_dir / "temp_batch_ocr_results.json"
        if temp_results_file.exists():
            print(f"Found existing results file: {temp_results_file}")
            try:
                with open(temp_results_file, 'r', encoding='utf-8') as f:
                    all_ocr_results = json.load(f)
                print(f"Loaded {len(all_ocr_results)} existing results. Resuming from where we left off...")
            except Exception as e:
                print(f"Error loading existing results: {e}. Starting fresh...")
                all_ocr_results = {}
        
        # Get list of images to process
        image_list = list(self.annotations.keys())
        if max_images:
            image_list = image_list[:max_images]
            
        # Filter out already processed images
        remaining_images = [img for img in image_list if img not in all_ocr_results]
        
        print(f"Total images: {len(image_list)}")
        print(f"Already processed: {len(all_ocr_results)}")
        print(f"Remaining to process: {len(remaining_images)}")
        
        if not remaining_images:
            print("All images already processed!")
            return all_ocr_results
        
        for idx, s3_img_path in enumerate(tqdm(remaining_images, desc="Processing images")):
            # Construct local image path
            local_img_path = self.local_image_dir / os.path.basename(s3_img_path)
            
            if not local_img_path.exists():
                print(f"Warning: Local image not found: {local_img_path}")
                continue
                
            # Process the image
            formatted_results, raw_result = self.process_single_image(
                str(local_img_path), save_visualizations)
            
            if formatted_results:
                all_ocr_results[s3_img_path] = {
                    'local_path': str(local_img_path),
                    'ocr_results': formatted_results,
                    'num_words': len(formatted_results)
                    # Don't include raw_result to avoid numpy serialization issues
                }
                processed_count += 1
            else:
                print(f"No OCR results for: {local_img_path}")
            
            # Save periodically
            if (idx + 1) % save_interval == 0:
                print(f"\nSaving intermediate results after processing {idx + 1} images...")
                self.save_intermediate_results(all_ocr_results, temp_results_file)
                print(f"Intermediate results saved. Total processed so far: {len(all_ocr_results)}")
        
        # Final save
        print(f"\nFinal save...")
        self.save_intermediate_results(all_ocr_results, temp_results_file)
        
        print(f"Successfully processed {processed_count} new images")
        print(f"Total results: {len(all_ocr_results)} images")
        return all_ocr_results
    
    def save_intermediate_results(self, ocr_results: Dict, temp_file_path: Path) -> None:
        """
        Save intermediate OCR results to a temporary JSON file.
        
        Args:
            ocr_results: Dictionary of all OCR results so far
            temp_file_path: Path to temporary results file
        """
        try:
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                json.dump(ocr_results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving intermediate results: {e}")
    
    def save_results(self, ocr_results: Dict, output_filename: str = "batch_ocr_results.json") -> None:
        """
        Save OCR results to JSON file.
        
        Args:
            ocr_results: Dictionary of all OCR results
            output_filename: Name of output file
        """
        output_path = self.output_dir / output_filename
        
        # Create a clean version without raw OCR data (which contains numpy arrays)
        clean_results = {}
        for img_path, data in ocr_results.items():
            clean_results[img_path] = {
                'local_path': data['local_path'],
                'ocr_results': data['ocr_results'],
                'num_words': data['num_words']
                # Skip 'raw_ocr' as it contains numpy arrays that aren't JSON serializable
            }
        
        print(f"Saving final results to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)
            
        print(f"✅ OCR results saved to: {output_path}")
        
        # Save statistics
        stats = {
            'total_images': len(clean_results),
            'total_words': sum(result['num_words'] for result in clean_results.values()),
            'avg_words_per_image': sum(result['num_words'] for result in clean_results.values()) / len(clean_results) if clean_results else 0
        }
        
        stats_path = self.output_dir / f"{output_filename.split('.')[0]}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
            
        print(f"📊 Statistics saved to: {stats_path}")
        print(f"📈 Final Statistics:")
        print(f"   - Total images processed: {stats['total_images']}")
        print(f"   - Total words detected: {stats['total_words']}")
        print(f"   - Average words per image: {stats['avg_words_per_image']:.1f}")
    
    def create_sample_visualization(self, num_samples: int = 3) -> None:
        """
        Process a few sample images for quick testing.
        
        Args:
            num_samples: Number of sample images to process
        """
        print(f"Processing {num_samples} sample images for visualization...")
        
        sample_keys = list(self.annotations.keys())[:num_samples]
        
        for s3_img_path in sample_keys:
            local_img_path = self.local_image_dir / os.path.basename(s3_img_path)
            print(f"Processing sample: {local_img_path}")
            
            if local_img_path.exists():
                formatted_results, raw_result = self.process_single_image(
                    str(local_img_path), save_visualization=True)
                print(f"  Found {len(formatted_results)} text regions")
                
                # Print first few results as example
                for i, result in enumerate(formatted_results[:3]):
                    print(f"    {i+1}. '{result['text']}' at ({result['x']}, {result['y']}) "
                          f"size: {result['width']}x{result['height']} conf: {result['confidence']:.2f}")
            else:
                print(f"  Image not found: {local_img_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch OCR processing for VDN documents')
    parser.add_argument('--annotation-file',
                       default='/data1/hang/Stellantis/processed_vdn_data_2807.json',
                       help='Path to annotation JSON file')
    parser.add_argument('--image-dir',
                       default='/data1/stellantis/images/VDN',
                       help='Directory containing the images')
    parser.add_argument('--output-dir',
                       default='./ocr_results',
                       help='Output directory for OCR results')
    parser.add_argument('--max-images', type=int,
                       help='Maximum number of images to process (default: all)')
    parser.add_argument('--save-visualizations', action='store_true',
                       help='Save visualization images with bounding boxes')
    parser.add_argument('--sample-only', action='store_true',
                       help='Process only a few sample images for testing')
    parser.add_argument('--output-filename',
                       default='batch_ocr_results.json',
                       help='Name of the output JSON file')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save intermediate results every N images (default: 100)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume processing from existing temp file if available')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BatchOCRProcessor(
        annotation_file=args.annotation_file,
        local_image_dir=args.image_dir,
        output_dir=args.output_dir
    )
    
    if args.sample_only:
        # Process only sample images
        processor.create_sample_visualization(num_samples=3)
    else:
        # Process all images
        ocr_results = processor.process_all_images(
            save_visualizations=args.save_visualizations,
            max_images=args.max_images,
            save_interval=args.save_interval
        )
        
        # Save final results
        processor.save_results(ocr_results, args.output_filename)
        
        # Clean up temporary file after successful completion
        temp_file = processor.output_dir / "temp_batch_ocr_results.json"
        if temp_file.exists():
            temp_file.unlink()
            print(f"Cleaned up temporary file: {temp_file}")


if __name__ == "__main__":
    main() 