#!/usr/bin/env python3
"""
Format VDN data using OCR results from batch_ocr_processor.py
Combines OCR text formatting with annotation extraction similar to process_data_getting_ocr.py
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import argparse

# Import the formatting functions from process_data_getting_ocr.py
import sys
sys.path.append('/data1/hang/Stellantis/InternVL/notebooks/OCR')
from process_data_getting_ocr import format_ocr_output, calculate_box_overlap, filter_overlapping_annotations


class VDNOCRFormatter:
    def __init__(self, 
                 annotation_file: str,
                 ocr_results_file: str,
                 local_image_dir: str):
        """
        Initialize the VDN OCR formatter.
        
        Args:
            annotation_file: Path to annotation JSON file (e.g., processed_vdn_data_2807.json)
            ocr_results_file: Path to OCR results from batch_ocr_processor.py
            local_image_dir: Directory containing the images
        """
        self.local_image_dir = Path(local_image_dir)
        
        # VDN fields
        self.fields = [
            'first_name', 'family_name', 'address_street', 'address_house_no', 'address_zip', 'address_city',
            'SV_number', 'tax_id', 'salary_month', 'gross_payment', 'real_payment', 'net_payment',
            'bank_account', 'bank_name', 'title_name', 'company_name', 'address_additional'
        ]
        
        # Load annotation data
        print(f"Loading annotations from {annotation_file}")
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        print(f"Loaded annotations for {len(self.annotations)} images")
        
        # Load OCR results
        print(f"Loading OCR results from {ocr_results_file}")
        with open(ocr_results_file, 'r', encoding='utf-8') as f:
            self.ocr_results = json.load(f)
        print(f"Loaded OCR results for {len(self.ocr_results)} images")
        
    def get_conversation_prompt(self, formatted_text: str) -> str:
        """
        Return the VDN extraction prompt with formatted OCR text.
        
        Args:
            formatted_text: Formatted OCR text from format_ocr_output
            
        Returns:
            Complete conversation prompt
        """
        return (
            "<image>\n"
            "You are a document information extraction assistant.\n"
            f"See the attached document image. And read the formatted text of the document: {formatted_text}.\n"
            "Extract the required information from the document and return it in the following JSON structure.\n"
            "Use the provided field descriptions and keyword hints for accurate matching.\n"
            "Return null if a field is missing. Do not add explanations.\n\n"
            "Output format:\n"
            "{\n"
            "  \"first_name\": \"Given name (e.g., 'Carolin')\",\n"
            "  \"family_name\": \"Family or last name (e.g., 'Balgenort')\",\n"
            "  \"title_name\": \"Title or honorific usually before the name, like Mr., Ms., Dr., Herr, Frau, etc. Use the title that is before the name, not after the name.\",\n"
            "  \"address_street\": \"Street name only, without house number (e.g., 'Hof im Hagen')\",\n"
            "  \"address_house_no\": \"House or building number (e.g., '7')\",\n"
            "  \"address_additional\": \"Optional address info (e.g., apartment, district)\",\n"
            "  \"address_zip\": \"Postal/ZIP code (e.g., '49134')\",\n"
            "  \"address_city\": \"City or town name (e.g., 'Wallenhorst'). If the city is not in the document, return 'null'.\",\n"
            "  \"SV_number\": \"Social security or pension number. Look for labels like 'SV-Nr.', 'RV-Nr.', or 'RV-Nummer'. Must be 12 characters, with 1 letter at position 10 (e.g., '50130984D504')\",\n"
            "  \"tax_id\": \"Tax identification number. Look for 'Steuer-ID', 'Steuer-Ident-Nr.'. It should be exactly 11 digits (e.g., '49285079139')\",\n"
            "  \"salary_month\": \"Salary period. Look for labels like 'Monat', 'für'. (e.g., '2025-04, April 2025')\",\n"
            "  \"gross_payment\": \"Monthly gross amount. Use value labeled like 'Gesamtbrutto' or 'Brutto'. Return number only, e.g., '3764.01'\",\n"
            "  \"net_payment\": \"Statutory net amount. Look for 'Gesetzliches Netto' or 'Netto'. Return number only, e.g., '3100.10'\",\n"
            "  \"real_payment\": \"Actual paid amount. Use value near bank name/number or 'Auszahlungsbetrag'. Return number only, e.g., '3100.10'\",\n"
            "  \"bank_account\": \"Bank account number or IBAN. Usually starts with 'DE' (Germany), near 'überwiesen' or 'Konto', usually 22-27 characters long\",\n"
            "  \"bank_name\": \"Bank name (e.g., 'Frankfurter Sparkasse'). Usually follows 'überwiesen bei' or after IBAN\",\n"
            "  \"company_name\": \"Company name. Look for label 'Firma' or names containing 'GmbH', 'AG', 'UG', 'oHG', 'BGB-Gesellschaft', 'Kommanditgesellschaft', etc.\"\n"
            "}\n"
            "If any field is missing or not visible in the document, set its value to null.\n"
            "Return only the JSON, with no explanation or commentary."
        )
    
    def extract_fields_from_annotations(self, annotations: Dict) -> Dict:
        """
        Extract field values from annotation data.
        
        Args:
            annotations: Annotation data for a single image
            
        Returns:
            Dictionary with extracted field values
        """
        extracted = {field: None for field in self.fields}
        
        if not annotations:
            return extracted
            
        for ann in annotations.values():
            label = ann.get("label")
            if label in self.fields:
                text_value = ann.get("text", "") if ann.get("text") else ""
                extracted[label] = text_value if text_value else None
                
        return extracted
    
    def process_single_image(self, s3_img_path: str, idx: int) -> Dict:
        """
        Process a single image to create formatted entry.
        
        Args:
            s3_img_path: S3 image path key
            idx: Entry index/ID
            
        Returns:
            Formatted entry dictionary or None if processing failed
        """
        local_img_path = self.local_image_dir / os.path.basename(s3_img_path)
        
        # Check if we have annotations for this image
        annotations = self.annotations.get(s3_img_path)
        if not annotations:
            print(f"No annotations found for: {s3_img_path}")
            return None
            
        # Check if we have OCR results for this image
        ocr_data = self.ocr_results.get(s3_img_path)
        if not ocr_data:
            print(f"No OCR results found for: {s3_img_path}")
            return None
            
        # Get image dimensions from first annotation
        first_ann = next(iter(annotations.values()))
        image_width = first_ann["image_width"]
        image_height = first_ann["image_height"]
        
        # Create entry structure
        entry = {
            'id': idx,
            'image': str(local_img_path),
            'width_list': image_width,
            'height_list': image_height
        }
        
        # Extract field values from annotations
        extracted = self.extract_fields_from_annotations(annotations)
        
        # Get OCR results and format them
        ocr_results = ocr_data['ocr_results']
        
        # Format OCR text using the same function as process_data_getting_ocr.py
        formatted_text = format_ocr_output(ocr_results)
        
        # Create conversation
        prompt = self.get_conversation_prompt(formatted_text)
        entry["conversations"] = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": json.dumps(extracted, ensure_ascii=False)}
        ]
        
        return entry
    
    def process_all_images(self) -> List[Dict]:
        """
        Process all images and create formatted dataset.
        
        Returns:
            List of formatted entries
        """
        formatted_data = []
        
        print("Processing all images...")
        
        # Get intersection of images that have both annotations and OCR results
        annotation_keys = set(self.annotations.keys())
        ocr_keys = set(self.ocr_results.keys())
        common_keys = annotation_keys.intersection(ocr_keys)
        
        print(f"Found {len(common_keys)} images with both annotations and OCR results")
        print(f"(Annotations: {len(annotation_keys)}, OCR: {len(ocr_keys)})")
        
        for idx, s3_img_path in enumerate(tqdm(common_keys, desc="Processing images")):
            entry = self.process_single_image(s3_img_path, idx)
            if entry:
                formatted_data.append(entry)
        
        print(f"Successfully processed {len(formatted_data)} images")
        return formatted_data
    
    def split_train_test(self, formatted_data: List[Dict], 
                        train_file_list: str, test_file_list: str) -> tuple:
        """
        Split formatted data into train/test based on file lists.
        
        Args:
            formatted_data: List of formatted entries
            train_file_list: Path to train_VDN.txt
            test_file_list: Path to test_VDN.txt
            
        Returns:
            Tuple of (train_data, test_data)
        """
        # Load train/test file lists
        train_files = set()
        test_files = set()
        
        if os.path.exists(train_file_list):
            with open(train_file_list, 'r', encoding='utf-8') as f:
                train_files = {line.strip() for line in f if line.strip()}
                
        if os.path.exists(test_file_list):
            with open(test_file_list, 'r', encoding='utf-8') as f:
                test_files = {line.strip() for line in f if line.strip()}
        
        print(f"Train file list: {len(train_files)} files")
        print(f"Test file list: {len(test_files)} files")
        
        # Split data
        train_data = []
        test_data = []
        unassigned = []
        
        for entry in formatted_data:
            image_path = entry['image']
            filename = os.path.basename(image_path)
            
            if filename in train_files:
                train_data.append(entry)
            elif filename in test_files:
                test_data.append(entry)
            else:
                unassigned.append(entry)
        
        print(f"Train data: {len(train_data)} entries")
        print(f"Test data: {len(test_data)} entries") 
        print(f"Unassigned: {len(unassigned)} entries")
        
        return train_data, test_data
    
    def save_results(self, train_data: List[Dict], test_data: List[Dict], 
                    output_dir: str) -> None:
        """
        Save train/test data to JSONL files.
        
        Args:
            train_data: Training data entries
            test_data: Test data entries  
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training data
        train_output = output_path / "vdn_ocr_chat_train_fieldtext.jsonl"
        with open(train_output, "w", encoding="utf-8") as f:
            for entry in train_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        # Save test data  
        test_output = output_path / "vdn_ocr_chat_valid_fieldtext.jsonl"
        with open(test_output, "w", encoding="utf-8") as f:
            for entry in test_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"Files saved:")
        print(f"  - {train_output} ({len(train_data)} entries)")
        print(f"  - {test_output} ({len(test_data)} entries)")
        
        # Save combined data as well
        combined_output = output_path / "vdn_ocr_chat_combined_fieldtext.jsonl"
        with open(combined_output, "w", encoding="utf-8") as f:
            for entry in train_data + test_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"  - {combined_output} ({len(train_data) + len(test_data)} entries)")


def main():
    parser = argparse.ArgumentParser(description='Format VDN data using OCR results')
    parser.add_argument('--annotation-file',
                       default='/data1/hang/Stellantis/processed_vdn_data_2807.json',
                       help='Path to annotation JSON file')
    parser.add_argument('--ocr-results-file',
                       default='./ocr_results/batch_ocr_results.json',
                       help='Path to OCR results from batch_ocr_processor.py')
    parser.add_argument('--image-dir',
                       default='/data1/stellantis/images/VDN',
                       help='Directory containing the images')
    parser.add_argument('--train-file',
                       default='/data1/hang/Stellantis/text_recognition/train_test_filenames/train_VDN.txt',
                       help='Path to train file list')
    parser.add_argument('--test-file',
                       default='/data1/hang/Stellantis/text_recognition/train_test_filenames/test_VDN.txt',
                       help='Path to test file list')
    parser.add_argument('--output-dir',
                       default='./formatted_ocr_data',
                       help='Output directory for formatted JSONL files')
    
    args = parser.parse_args()
    
    # Initialize formatter
    formatter = VDNOCRFormatter(
        annotation_file=args.annotation_file,
        ocr_results_file=args.ocr_results_file,
        local_image_dir=args.image_dir
    )
    
    # Process all images
    formatted_data = formatter.process_all_images()
    
    if not formatted_data:
        print("No data was processed successfully!")
        return
    
    # Split into train/test
    train_data, test_data = formatter.split_train_test(
        formatted_data, args.train_file, args.test_file)
    
    # Save results
    formatter.save_results(train_data, test_data, args.output_dir)


if __name__ == "__main__":
    main() 