import os
import sys
sys.path.append("/data1/hang/Stellantis/InternVL/internvl_chat")
from internvl.model import InternVLChatConfig, load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
import argparse
import torch
import json


def get_groundtruth(image_path):
    path1 = '/data1/hang/Stellantis/InternVL/notebooks/vdn_chat_train_only_fieldtext.jsonl'
    path2 = '/data1/hang/Stellantis/InternVL/notebooks/vdn_chat_valid_only_fieldtext.jsonl'
    data1 = [json.loads(line) for line in open(path1).readlines()]
    data2 = [json.loads(line) for line in open(path2).readlines()]
    data = data1 + data2
    for item in data1:
        if item['image'] == image_path:
            print('traindata groundtruth')
            return item['conversations'][-1]['value']
    for item in data2:
        if item['image'] == image_path:
            print('validdata')
            return item['conversations'][-1]['value']

def inference_single_image(image_path, question, model, tokenizer, device='cuda', use_thumbnail=False, image_size=224, max_new_tokens=1000, max_num=6, num_beams=3, temperature=0.0):
    # Load and preprocess image with dynamic sizing (matching training)
    image = Image.open(image_path).convert('RGB')
    
    # Use dynamic preprocessing like in training
    images = dynamic_preprocess(image, image_size=image_size,
                              use_thumbnail=use_thumbnail,
                              max_num=max_num)
    
    # Apply transforms
    transform = build_transform(is_train=False, input_size=image_size)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    
    # Move to specified CUDA device and convert to bfloat16
    pixel_values = pixel_values.to(torch.bfloat16).to(device)
    
    if question is None:
        question = '<image>'
    
    generation_config = dict(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
        min_new_tokens=1,
    )
    
    with torch.no_grad():
        outputs = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
        )
        return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
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
    # parser.add_argument('--question', type=str, default="""You are a document understanding assistant. Given a document image, extract the following structured fields as accurately and completely as possible. Handle complex layouts or faint text gracefully. If a field is missing or unreadable, return null. Respond strictly in JSON format.

    #     Fields to extract:
    #     bank_name: Name of the bank
    #                     bank_account: Bank account number # lấy cao nhất
    #                     SV_number: Social insurance number or employee number
    #                     tax_id: Tax identification number
    #                     first_name: Person's first name
    #                     family_name: Person's family/last name
    #                     title_name: Person's title or honorific (e.g., Mr., Mrs., Dr., Prof., etc.)
    #                     salary_month: The month the salary corresponds to (e.g., "March 2023")
    #                     gross_payment: Gross salary amount before deductions
    #                     net_payment: Net salary received after deductions
    #                     real_payment: Actual amount paid (can be same as net_payment)
    #                     address_street: Street name of the address
    #                     address_house_no: House or building number
    #                     address_city: City name
    #                     address_zip: Postal or ZIP code
    #                     company_name: Name of the company or organization associated with the person or document
    #                     address_additional: Supplementary address details such as apartment number, building name, floor, suite, or additional delivery instructions

                        
    #                 Output format (JSON):
    #                      {
    #                     "bank_name": "...",
    #                     "bank_account": "...",
    #                     "SV_number": "...",
    #                     "tax_id": "...",
    #                     "first_name": "...",
    #                     "family_name": "...",
    #                     "title_name": "...",   
    #                     "salary_month": "...",
    #                     "gross_payment": "...",
    #                     "net_payment": "...",
    #                     "real_payment": "...",
    #                     "address_street": "...",
    #                     "address_house_no": "...",
    #                     "address_city": "...",
    #                     "address_zip": "...",
    #                     "company_name": "..."
    #                     "address_additional": "..."
    #                     }
    #     """, help='Question about the image (optional, for single image mode)')
    parser.add_argument('--checkpoint', type=str, default = "/data1/hang/Stellantis/InternVL/internvl_chat/work_dirs/internvl_chat_v3/vdn_fieldtext_internvl3_1b_dynamic_res_2nd_finetune_full")
    parser.add_argument('--num-beams', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=1000)
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--auto', action='store_true', default=False, help='Use auto device mapping')
    parser.add_argument('--load-in-8bit', action='store_true', default=False, help='Load model in 8-bit mode')
    parser.add_argument('--load-in-4bit', action='store_true', default=False, help='Load model in 4-bit mode')
    parser.add_argument('--device', type=str, default='cuda', help='CUDA device to run inference on (e.g., cuda, cuda:0, cuda:1)')
    
    args = parser.parse_args()
    print('image', args.image)
    
    # Handle image path - try absolute path first, then relative to VDN_annotated
    image_path = args.image
    if not os.path.exists(image_path):
        vdn_path = os.path.join('/data1/hang/Stellantis/VDN_annotated', args.image)
        if os.path.exists(vdn_path):
            image_path = vdn_path
        else:
            print(f"Warning: Image not found at {args.image} or {vdn_path}")
    
    print(f'Using image path: {image_path}')
    print(f'Using device: {args.device}')
    
    # Load model and tokenizer - pass args object, not individual parameters
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Move model to specified CUDA device
    model = model.to(args.device)
    print(f'Model moved to {args.device}')
    
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    max_num = model.config.max_dynamic_patch
    
    total_params = sum(p.numel() for p in model.parameters())/1e9
    print(f'Total parameters: {total_params}B')
    print(f'Image size: {image_size}')
    print(f'Use thumbnail: {use_thumbnail}')
    
    if args.question:
        question = args.question
    else:
        question = None
        
    answer = inference_single_image(
        image_path=image_path,
        question=question,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        use_thumbnail=use_thumbnail,
        image_size=image_size,
        max_new_tokens=args.max_new_tokens,
        max_num=max_num,
        num_beams=args.num_beams,
        temperature=args.temperature,
    )
    print('='*100)
    print(get_groundtruth(image_path))
    print('='*100)
    print(answer)
    
if __name__ == '__main__':
    main()
        
    