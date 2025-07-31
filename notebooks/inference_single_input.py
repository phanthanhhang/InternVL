import json
from pprint import pprint
import sys
import torch
import os
from tqdm import tqdm
sys.path.append("/data1/hang/Stellantis/InternVL/internvl_chat")
from internvl.model import InternVLChatConfig, load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
import argparse

# data_path = "/data1/hang/Stellantis/InternVL/notebooks/vdn_chat_valid_only_fieldtext.jsonl"
# data = [json.loads(line) for line in open(data_path).readlines()]

# data = data[0]
# pprint(data)
# checkpoint = "/data1/hang/Stellantis/InternVL/internvl_chat/work_dirs/internvl_chat_v3/vdn_fieldtext_internvl3_1b_dynamic_res_2nd_finetune_lora"
# config = InternVLChatConfig.from_pretrained(checkpoint)
# num_hidden_layers = config.llm_config.num_hidden_layers
# # device_map = split_model(num_hidden_layers)
# print(num_hidden_layers)

def inference_single_image_vqa_style(image_path, question, model, tokenizer, 
                                   dataset_type='default', 
                                   dynamic_image_size=False,
                                   use_thumbnail=False,
                                   max_num=6,
                                   max_new_tokens=10,
                                   num_beams=1,
                                   temperature=0.0):
    """
    Run inference on a single image based on the VQA evaluation style
    
    Args:
        image_path (str): Path to the image file
        question (str): Question to ask about the image
        model: The loaded InternVL model
        tokenizer: The model's tokenizer
        dataset_type (str): Type of dataset to determine prompt style 
                           ('vizwiz', 'ai2d', 'infographicsvqa', or 'default')
        dynamic_image_size (bool): Whether to use dynamic image preprocessing
        use_thumbnail (bool): Whether to use thumbnail in dynamic preprocessing
        max_num (int): Maximum number of image patches for dynamic preprocessing
        max_new_tokens (int): Maximum number of tokens to generate
        num_beams (int): Number of beams for beam search
        temperature (float): Temperature for sampling
    
    Returns:
        str: Model's answer to the question
    """
    
    # Get image size from model config
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    
    # Set prompt based on dataset type (following evaluate_vqa.py logic)
    base_prompt = 'Answer the question using a single word or phrase.'
    vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
    infovqa_prompt = 'Answer the question using a single word or phrase.'
    ai2d_prompt = ''
    
    if dataset_type == 'vizwiz':
        input_prompt = vizwiz_prompt + base_prompt
    elif dataset_type == 'ai2d':
        input_prompt = ai2d_prompt
    elif dataset_type == 'infographicsvqa':
        input_prompt = infovqa_prompt
    else:
        input_prompt = base_prompt
    
    # Add prompt to question if not empty
    if len(input_prompt) != 0:
        question = question + ' ' + input_prompt
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    if dynamic_image_size:
        # Use dynamic preprocessing (similar to evaluate_vqa.py)
        images = dynamic_preprocess(image, image_size=image_size,
                                  use_thumbnail=use_thumbnail,
                                  max_num=max_num)
    else:
        images = [image]
    
    # Apply transforms
    transform = build_transform(is_train=False, input_size=image_size)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    
    # Move to GPU and convert to bfloat16
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    
    # Set up generation config (following evaluate_vqa.py)
    generation_config = dict(
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
    )
    
    # Get model prediction
    answer = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question,
        generation_config=generation_config,
        verbose=True
    )
    
    return answer

def inference_single_image(image_path, question, model, tokenizer, 
                          image_size=224, max_new_tokens=1000, 
                          max_num=24, num_beams=5, temperature=0.0):
    """
    Simple inference function for a single image (original version)
    
    Args:
        image_path (str): Path to the image file
        question (str): Question to ask about the image
        model: The loaded InternVL model
        tokenizer: The model's tokenizer
        image_size (int): Size to resize the image to
        max_new_tokens (int): Maximum number of tokens to generate
        max_num (int): Maximum number of image patches for dynamic preprocessing
        num_beams (int): Number of beams for beam search
        temperature (float): Temperature for sampling
    
    Returns:
        str: Model's answer to the question
    """
    # Prepare the image
    print(f'[info] image_path: {image_path}')
    transform = build_transform(is_train=False, input_size=image_size)
    image = Image.open(image_path).convert('RGB')
    pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Move to GPU and convert to bfloat16
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    
    # Add base prompt
    # base_prompt = 'Answer the question using a single word or phrase.'
    base_prompt = ''
    question = question + ' ' + base_prompt
    
    # Set up generation config
    generation_config = dict(
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
    )
    # print('generation_config', generation_config)
    # Get model prediction
    answer = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question,
        generation_config=generation_config,
        verbose=True
    )
    # print('answer', answer)
    return answer

def process_jsonl_file(jsonl_path, model, tokenizer, args, output_file):
    """
    Process all instances in the JSONL file and run inference on each image
    
    Args:
        jsonl_path (str): Path to the JSONL file
        model: The loaded InternVL model
        tokenizer: The model's tokenizer
        args: Command line arguments
        output_file (str): Path to output file for saving results
    
    Returns:
        list: List of results for each instance
    """
    # Read all data from JSONL file
    data = [json.loads(line) for line in open(jsonl_path).readlines()]
    
    print(f"Loaded {len(data)} instances from {jsonl_path}")
    
    results = []
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    
    # Check if there's existing results file to resume from
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            processed_ids = {result['id'] for result in existing_results}
            results.extend(existing_results)
            print(f"Found existing results file with {len(existing_results)} instances. Resuming...")
        except:
            processed_ids = set()
            print("Could not load existing results, starting fresh...")
    else:
        processed_ids = set()
    
    save_counter = 0
    for i, instance in tqdm(enumerate(data), total=len(data), desc="Processing instances"):
        # Extract information from instance
        instance_id = instance.get('id', i)
        
        # Skip if already processed
        if instance_id in processed_ids:
            continue
            
        image_path = instance['image']
        conversations = instance['conversations']
        
        # Find the human question (first conversation from human)
        human_question = None
        expected_answer = None
        for conv in conversations:
            if conv['from'] == 'human':
                human_question = conv['value']
                break
        
        # Find the expected answer (first conversation from gpt)
        for conv in conversations:
            if conv['from'] == 'gpt':
                expected_answer = conv['value']
                break
        
        if human_question is None:
            print(f"Warning: No human question found for instance {instance_id}")
            continue
            
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        human_question = human_question + "DO NOT HALLUCINATE THE ANSWER"
        try:
            # Run inference
            if args.use_vqa_style:
                predicted_answer = inference_single_image_vqa_style(
                    image_path=image_path,
                    question=human_question,
                    model=model,
                    tokenizer=tokenizer,
                    dataset_type=args.dataset_type,
                    dynamic_image_size=args.dynamic,
                    use_thumbnail=use_thumbnail,
                    max_num=args.max_num,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    temperature=args.temperature
                )
            else:
                predicted_answer = inference_single_image(
                    image_path=image_path,
                    question=human_question,
                    model=model,
                    tokenizer=tokenizer,
                    image_size=image_size,
                    max_new_tokens=args.max_new_tokens,
                    max_num=args.max_num,
                    num_beams=args.num_beams,
                    temperature=args.temperature
                )
            
            # Store result
            result = {
                'id': instance_id,
                'image_path': image_path,
                # 'question': human_question,
                'expected_answer': expected_answer,
                'predicted_answer': predicted_answer,
                'width': instance.get('width_list'),
                'height': instance.get('height_list')
            }
            results.append(result)
            save_counter += 1
            
            # Print progress for first few instances
            if i < 5 or (i + 1) % 10 == 0:
                print(f"\n--- Instance {i+1}/{len(data)} (ID: {instance_id}) ---")
                print(f"Image: {image_path}")
                print(f"Question: {human_question}..." )
                print(f"Predicted: {predicted_answer}")
                if expected_answer:
                    print(f"Expected: {expected_answer}...")
            
            # Save results every 5 instances
            if save_counter % 5 == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\n[SAVED] Progress saved after {save_counter} new instances (Total: {len(results)} instances)")
                
        except Exception as e:
            print(f"Error processing instance {instance_id}: {str(e)}")
            continue
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[FINAL SAVE] All results saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/data1/hang/Stellantis/InternVL/internvl_chat/work_dirs/internvl_chat_v3/vdn_fieldtext_internvl3_1b_dynamic_res_2nd_finetune_lora')
    parser.add_argument('--jsonl-file', type=str, default='/data1/hang/Stellantis/InternVL/notebooks/vdn_chat_valid_only_fieldtext.jsonl', help='Path to JSONL file with instances')
    parser.add_argument('--image', type=str, help='Path to single input image (optional, for single image mode)')
    parser.add_argument('--question', type=str, 
                        default=('<image>\n'
                                'You are a document information extraction '
                                'assistant.\n'
                                'Extract the required information from the '
                                'document and return it in the following JSON '
                                'structure.\n'
                                'Use the provided field descriptions and keyword '
                                'hints for accurate matching.\n'
                                'Return null if a field is missing. Do not add '
                                'explanations.\n'
                                '\n'
                                'Output format:\n'
                                '{\n'
                                '  "first_name": "Given name (e.g., '
                                '\'Carolin\')",\n'
                                '  "family_name": "Family or last name (e.g., '
                                '\'Balgenort\')",\n'
                                '  "title_name": "Title or honorific usually '
                                'before the name, like Mr., Ms., Dr., Herr, Frau, '
                                'etc.\\ Use the title that is before the name, '
                                'not after the name.\n'
                                '  "address_street": "Street name only, without '
                                'house number (e.g., \'Hof im Hagen\')",\n'
                                '  "address_house_no": "House or building number '
                                '(e.g., \'7\')",\n'
                                '  "address_additional": "Optional address info '
                                '(e.g., apartment, district)",\n'
                                '  "address_zip": "Postal/ZIP code (e.g., '
                                '\'49134\')",\n'
                                '  "address_city": "City or townname (e.g., '
                                "'Wallenhorst'). If the city is not in the "
                                "document, return 'null'.\n"
                                '  "SV_number": "Social security or pension '
                                "number. Look for labels like 'SV-Nr.', 'RV-Nr.', "
                                "or 'RV-Nummer'. Must be 12 characters, with 1 "
                                'letter at position 10 (e.g., '
                                '\'50130984D504\')",\n'
                                '  "tax_id": "Tax identification number. Look for '
                                "'Steuer-ID', 'Steuer-Ident-Nr.'. It should be "
                                'exactly 11 digits (e.g., \'49285079139\')",\n'
                                '  "salary_month": "Salary period. Look for '
                                "labels like 'Monat', 'für'. (e.g., '2025-04, "
                                'April 2025\')",\n'
                                '  "gross_payment": "Monthly gross amount. Use '
                                "value labeled like 'Gesamtbrutto' or 'Brutto'. "
                                'Return number only, e.g., \'3764.01\'",\n'
                                '  "net_payment": "Statutory net amount. Look for '
                                "'Gesetzliches Netto' or 'Netto'. Return number "
                                'only, e.g., \'3100.10\'",\n'
                                '  "real_payment": "Actual paid amount. Use value '
                                "near bank name/number or 'Auszahlungsbetrag'. "
                                'Return number only, e.g., \'3100.10\'",\n'
                                '  "bank_account": "Bank account number or IBAN. '
                                "Usually starts with 'DE' (Germany), near "
                                "'überwiesen' or 'Konto', usually 22-27 "
                                'characters long",\n'
                                '  "bank_name": "Bank name (e.g., \'Frankfurter '
                                "Sparkasse'). Usually follows 'überwiesen bei' or "
                                'after IBAN",\n'
                                '  "company_name": "Company name. Look for label '
                                "'Firma' or names containing 'GmbH', 'AG', 'UG', "
                                "'oHG', 'BGB-Gesellschaft', "
                                '\'Kommanditgesellschaft\', etc."\n'
                                '}\n'
                                'If any field is missing or not visible in the '
                                'document, set its value to null.\n'
                                'Return only the JSON, with no explanation or '
                                'commentary.'),
                        help='Question about the image (optional, for single image mode)')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--dataset-type', type=str, default='default', 
                        choices=['default', 'vizwiz', 'ai2d', 'infographicsvqa'],
                        help='Dataset type to determine prompt style')
    parser.add_argument('--max-new-tokens', type=int, default=1000, help='Maximum tokens to generate')
    parser.add_argument('--use-vqa-style', action='store_true', default=False, help='Use VQA-style inference instead of simple inference (default: simple)')
    parser.add_argument('--single-mode', action='store_true', help='Run single image inference instead of processing JSONL file')
    parser.add_argument('--max-instances', type=int, default=None, help='Maximum number of instances to process')
    parser.add_argument('--output-name', type=str, default=None, help='Custom output file name (without extension)')
    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[info] total_params: {total_params}B')
    print(f'[info] image_size: {image_size}')
    print(f'[info] template: {model.config.template}')
    print(f'[info] use_thumbnail: {use_thumbnail}')
    
    # Choose mode: single image or JSONL processing
    if args.single_mode and args.image and args.question:
        print("\n--- Single Image Mode ---")
        # Run single image inference
        if args.use_vqa_style:
            answer = inference_single_image_vqa_style(
                image_path=os.path.join('/data1/hang/Stellantis/VDN_annotated', args.image),
                question=args.question,
                model=model,
                tokenizer=tokenizer,
                dataset_type=args.dataset_type,
                dynamic_image_size=args.dynamic,
                use_thumbnail=use_thumbnail,
                max_num=args.max_num,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                temperature=args.temperature
            )
        else:
            answer = inference_single_image(
                image_path=os.path.join('/data1/hang/Stellantis/VDN_annotated', args.image),
                question=args.question,
                model=model,
                tokenizer=tokenizer,
                image_size=image_size,
                max_new_tokens=args.max_new_tokens,
                max_num=args.max_num,
                num_beams=args.num_beams,
                temperature=args.temperature
            )
        
        print(f"\nQuestion: {args.question}")
        print(f"Answer: {answer}")
        
    else:
        print("\n--- JSONL Processing Mode ---")
        
        # Prepare output file name
        import time
        if args.output_name:
            output_file = os.path.join(args.out_dir, f'{args.output_name}.json')
        else:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(args.out_dir, f'inference_results_{timestamp}.json')
        
        print(f"Results will be saved to: {output_file}")
        
        # Process JSONL file
        results = process_jsonl_file(args.jsonl_file, model, tokenizer, args, output_file)
        
        # Limit results if specified
        if args.max_instances:
            results = results[:args.max_instances]
            # Save limited results
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n--- Results Summary ---")
        print(f"Total instances processed: {len(results)}")
        print(f"Results saved to: {output_file}")
        
        # Show some statistics
        if results:
            print(f"\nFirst few results:")
            for i, result in enumerate(results[:3]):
                print(f"{i+1}. ID: {result['id']}")
                print(f"   Predicted: {result['predicted_answer'][:100]}...")
                if result['expected_answer']:
                    print(f"   Expected: {result['expected_answer'][:100]}...")

if __name__ == '__main__':
    main()





