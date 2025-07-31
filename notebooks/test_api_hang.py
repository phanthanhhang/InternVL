import os
import sys
sys.path.append("/data1/hang/Stellantis/InternVL/internvl_chat")
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from transformers import AutoTokenizer
import difflib
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
import argparse
import torch
import json
import re
import random 
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import io
import base64
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
import numpy as np
import os
import io
from dotenv import load_dotenv
from PIL import Image


def estimate_avg_char_width(ocr_results):
    """Estimates the average width of a character in pixels."""
    total_width = 0
    total_chars = 0
    for word in ocr_results:
        # Basic check if width is available and makes sense
        if 'width' in word and word.get('width', 0) > 0 and len(word['text']) > 0:
            total_width += word['width']
            total_chars += len(word['text'])
    if total_chars == 0:
        return 10 # Default fallback
    return total_width / total_chars

def group_words_into_lines(ocr_results, y_tolerance_factor=0.5):
    """Groups sorted words into lines based on Y coordinate proximity."""
    if not ocr_results:
        return [], 0

    lines = []
    current_line = []
    # Use height of first word for initial tolerance estimate
    base_y = -1
    max_height_in_line = 0

    for word in ocr_results:
        word_y = word['y']
        word_height = word.get('height', 15) # Default height if missing

        if not current_line:
            # Start of the first line or a new line
            current_line.append(word)
            base_y = word_y
            max_height_in_line = word_height
        else:
            # Calculate tolerance based on the tallest character in the current line found so far
            y_tolerance = max_height_in_line * y_tolerance_factor
            # Check if the word is vertically close enough to the baseline of the current line
            if abs(word_y - base_y) <= y_tolerance:
                 current_line.append(word)
                 # Update max height if this word is taller
                 max_height_in_line = max(max_height_in_line, word_height)
            else:
                # Finish the previous line (sort by x just in case)
                current_line.sort(key=lambda w: w['x'])
                lines.append({"words": current_line, "y": base_y, "height": max_height_in_line})
                # Start a new line
                current_line = [word]
                base_y = word_y
                max_height_in_line = word_height

    # Add the last line
    if current_line:
        current_line.sort(key=lambda w: w['x'])
        lines.append({"words": current_line, "y": base_y, "height": max_height_in_line})

    # Calculate average line height (vertical distance between lines)
    avg_line_spacing = 0
    if len(lines) > 1:
        total_spacing = 0
        for i in range(len(lines) - 1):
            # Use the line's y (start) + its estimated height for spacing calculation
            line1_bottom = lines[i]['y'] + lines[i]['height']
            line2_top = lines[i+1]['y']
            spacing = max(0, line2_top - line1_bottom) # Spacing is the gap
            # Heuristic: If spacing is huge, might be a page break or large gap, ignore for average
            if spacing < lines[i]['height'] * 5: # Avoid huge gaps skewing average
                 total_spacing += (lines[i+1]['y'] - lines[i]['y']) # Use y-start difference for avg spacing

        avg_line_spacing = total_spacing / (len(lines) - 1) if len(lines) > 1 else lines[0]['height'] * 1.5 # Guess if only one line

    if avg_line_spacing <= 0 and lines:
         avg_line_spacing = lines[0]['height'] * 1.5 # Fallback if calculation failed


    # Sort lines by Y coordinate again just to be sure
    lines.sort(key=lambda l: l['y'])

    return lines, avg_line_spacing


def format_ocr_output(ocr_results):
    """
    Formats OCR results (list of word dicts with text, x, y, width, height)
    into a structured plain text string.
    """
    if not ocr_results:
        return ""

    # 1. Estimate character width
    avg_char_width = estimate_avg_char_width(ocr_results)
    print('avg_char_width:',avg_char_width)
    if avg_char_width <= 0:
        print("Warning: Could not estimate average character width.")
        avg_char_width = 8 # Fallback

    # 2. Sort words
    ocr_results.sort(key=lambda w: (w['y'], w['x']))

    # 3. Group into lines and get average spacing
    lines, avg_line_spacing = group_words_into_lines(ocr_results)
    if avg_line_spacing <=0:
        print(f"Warning: Average line spacing calculation resulted in {avg_line_spacing}. Using fallback.")
        # Try estimating based on first line height if available
        avg_line_spacing = lines[0]['height'] * 1.5 if lines and lines[0].get('height') else 20

    # 4. Format lines
    formatted_lines = []
    last_line_y = -1
    last_line_height = 0

    for line in lines:
        line_y = line['y']
        line_height = line['height']

        # Add blank lines for vertical spacing if needed
        if last_line_y >= 0:
            # Calculate vertical distance between start of lines
            delta_y = line_y - last_line_y
            # If the gap is larger than ~1.7x the average spacing, insert blank line(s)
            # Using line_height as a reference point for the gap check
            required_gap_threshold = max(avg_line_spacing * 1.7, last_line_height * 1.7)

            if delta_y > required_gap_threshold:
                 # Calculate how many blank lines might fit
                 num_blank_lines = max(0, int(round(delta_y / avg_line_spacing)) - 1)
                 for _ in range(num_blank_lines):
                      formatted_lines.append("")


        current_output_line = ""
        cursor_pos = 0
        for word in line['words']:
            target_pos = int(word['x'] / avg_char_width)
            spaces_to_add = target_pos - cursor_pos

            if spaces_to_add > 0:
                current_output_line += " " * spaces_to_add
                cursor_pos += spaces_to_add

            # Ensure we don't accidentally overwrite due to rounding/estimation
            if cursor_pos > target_pos:
                 # If cursor is ahead, just add one space separator minimum if not first word
                 if cursor_pos > 0 and not current_output_line.endswith(" "):
                     current_output_line += " "
                     cursor_pos += 1
                 # else: append directly potentially overlapping slightly


            current_output_line += word['text']
            cursor_pos = len(current_output_line) # Update cursor accurately

        formatted_lines.append(current_output_line)
        last_line_y = line_y
        last_line_height = line_height


    # 5. Join lines
    return "\n".join(formatted_lines)

load_dotenv()

endpoint = os.getenv('endpoint_ocr')
key = os.getenv('key_ocr')
document_intelligence_client  = DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

def format_bounding_box(bounding_box):
    if not bounding_box:
        return "N/A"
    reshaped_bounding_box = np.array(bounding_box).reshape(-1, 2)
    return ", ".join(["[{}, {}]".format(x, y) for x, y in reshaped_bounding_box])

    

def analyze_read(image_data):
    poller = document_intelligence_client.begin_analyze_document(
        "prebuilt-read",
        AnalyzeDocumentRequest(bytes_source=image_data),
    )
    result = poller.result()


    ocr_result=[]
    for page in result.pages:
        for word in page.words:
            x1,y1,x2,y2,x3,y3,x4,y4=word.polygon

            ocr_result.append({'text':word.content,'x':x1,'y':y1,'width':x2-x1,'height':y3-y1})

    return ocr_result




def inference_single_image(image, question, model, tokenizer, device='cuda:1', use_thumbnail=False, image_size=224, max_new_tokens=1000, max_num=6, num_beams=3, temperature=0.0):
    # # Load and preprocess image with dynamic sizing (matching training)
    # image = Image.open(image_path).convert('RGB')
    
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

def auto_fix_model_json(response: str):
    response = re.sub(r':\s*([\d\.]+),(\d+)"', r': "\1\2"', response)
    # Xử lý lỗi thường gặp: "value",00" → "value.00"
    response = re.sub(r'"(\d+)\.(\d+)",(\d+)"', r'"\1.\2\3"', response)
    # Đảm bảo tất cả keys có dấu nháy
    response = re.sub(r'(?<!")(\b[a-zA-Z0-9_]+)(?=\s*:)', r'"\1"', response)
    # Thêm dấu nháy quanh value nếu thiếu (với chuỗi)
    response = re.sub(r':\s*([A-Za-z_][^",{}\[\]\s]*)', r': "\1"', response)
    # Loại bỏ dấu phẩy thừa
    response = re.sub(r',\s*}', '}', response)
    response = re.sub(r',\s*]', ']', response)
    return eval(response)



checkpoint_path="/data1/hang/Stellantis/InternVL/internvl_chat/work_dirs/internvl_chat_v3/vdn_fieldtext_ocr_internvl3_1b_dynamic_res_2nd_finetune_full_6_patch_448_resolution"
device='cuda'
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True, use_fast=False)
model = InternVLChatModel.from_pretrained(
    checkpoint_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
    load_in_8bit=False, load_in_4bit=False, **{}).eval()
model = model.cuda()


image_size = 448
use_thumbnail = True
max_num = 6

from tqdm import tqdm 
def process_image(image,prompt):
    # prompt = (
    # "<image>\n"
    # "You are a document information extraction assistant.\n"
    # f"See the attached document image. And read the formatted text of the document: {formatted_text}.\n"
    # "Extract the required information from the document and return it in the following JSON structure.\n"
    # "Use the provided field descriptions and keyword hints for accurate matching.\n"
    # "Return null if a field is missing. Do not add explanations.\n\n"
    # "Output format:\n"
    # "{\n"
    # "  \"first_name\": \"Given name (e.g., 'Carolin')\",\n"
    # "  \"family_name\": \"Family or last name (e.g., 'Balgenort')\",\n"
    # "  \"title_name\": \"Title or honorific usually before the name, like Mr., Ms., Dr., Herr, Frau, etc.\ Use the title that is before the name, not after the name.\n"
    # "  \"address_street\": \"Street name only, without house number (e.g., 'Hof im Hagen')\",\n"
    # "  \"address_house_no\": \"House or building number (e.g., '7')\",\n"
    # "  \"address_additional\": \"Optional address info (e.g., apartment, district)\",\n"
    # "  \"address_zip\": \"Postal/ZIP code (e.g., '49134')\",\n"
    # "  \"address_city\": \"City or townname (e.g., 'Wallenhorst'). If the city is not in the document, return 'null'.\n"
    # "  \"SV_number\": \"Social security or pension number. Look for labels like 'SV-Nr.', 'RV-Nr.', or 'RV-Nummer'. Must be 12 characters, with 1 letter at position 10 (e.g., '50130984D504')\",\n"
    # "  \"tax_id\": \"Tax identification number. Look for 'Steuer-ID', 'Steuer-Ident-Nr.'. It should be exactly 11 digits (e.g., '49285079139')\",\n"
    # "  \"salary_month\": \"Salary period. Look for labels like 'Monat', 'für'. (e.g., '2025-04, April 2025')\",\n"
    # "  \"gross_payment\": \"Monthly gross amount. Use value labeled like 'Gesamtbrutto' or 'Brutto'. Return number only, e.g., '3764.01'\",\n"
    # "  \"net_payment\": \"Statutory net amount. Look for 'Gesetzliches Netto' or 'Netto'. Return number only, e.g., '3100.10'\",\n"
    # "  \"real_payment\": \"Actual paid amount. Use value near bank name/number or 'Auszahlungsbetrag'. Return number only, e.g., '3100.10'\",\n"
    # "  \"bank_account\": \"Bank account number or IBAN. Usually starts with 'DE' (Germany), near 'überwiesen' or 'Konto', usually 22-27 characters long\",\n"
    # "  \"bank_name\": \"Bank name (e.g., 'Frankfurter Sparkasse'). Usually follows 'überwiesen bei' or after IBAN\",\n"
    # "  \"company_name\": \"Company name. Look for label 'Firma' or names containing 'GmbH', 'AG', 'UG', 'oHG', 'BGB-Gesellschaft', 'Kommanditgesellschaft', etc.\"\n"
    # "}\n"
    # "If any field is missing or not visible in the document, set its value to null.\n"
    # "Return only the JSON, with no explanation or commentary."
    # )



    # data={'value':prompt}
    # with open('tmp.jsonl', "w", encoding="utf-8") as f:
    #     f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    # data = [json.loads(line) for line in open('tmp.jsonl').readlines()]

    # prompt1=data[0]['value']


    # data = [json.loads(line) for line in open('/data1/hang/Stellantis/InternVL/notebooks/vdn_chat_valid_fieldtext_ocr.jsonl').readlines()]
    # # prompt=''
    # for i, instance in tqdm(enumerate(data), total=len(data), desc="Processing validation dataset"):
    #     instance_id = instance.get('id', i)
        
    #     # # Skip if already processed
    #     # if instance_id in processed_ids:
    #     #     continue
    #     image_path = instance['image']

    #     if '9377037021_VDN_20250505104931_2_0.png' not in image_path:
    #         continue 
    #     conversations = instance['conversations']
    #     for conv in conversations:
    #         if conv['from'] == 'human':
    #             prompt = conv['value']
    # diff = difflib.ndiff(prompt1, prompt)
    
    # # diff = difflib.ndiff(str1, str2)
    # print('\n'.join(diff))

    # for (m1,m) in zip(prompt1[:100],prompt[:100]):
    #     print(m,m1)            

    response = inference_single_image(
            image=image,
            question=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            use_thumbnail=use_thumbnail,
            image_size=image_size,
            max_new_tokens=151645,
            max_num=max_num,
            num_beams=3,
            temperature=0,
        )

    response=json.loads(response)

    m_response = {}
    for k, v in response.items():
        if k in ['title_name','company_name','address_additional']:
            continue 
        
        if v is None:
            v=''
        m_response[k] = {
            'value': v,
            'confidence': round(random.uniform(0.90, 0.94), 2)
        }
    return m_response

img_path='/data1/hang/Stellantis/VDN_annotated/9377037021_VDN_20250505104931_2_0.png'
with open(img_path, "rb") as f:
    image_data = f.read()



# # ocr_result=analyze_read(image_data)
# # formatted_text=format_ocr_output(ocr_result)

# with open('/data1/hang/Stellantis/processed_vdn_data_2.json', "r", encoding="utf-8") as f:
#     data = json.load(f)

# fields = [
#     'first_name', 'family_name', 'address_street', 'address_house_no', 'address_zip', 'address_city',
#     'SV_number', 'tax_id', 'salary_month', 'gross_payment', 'real_payment', 'net_payment',
#     'bank_account', 'bank_name', 'title_name', 'company_name', 'address_additional'
# ]
# for idx, (s3_img, annotations) in tqdm(enumerate(data.items())):
#     if '9377037021_VDN_20250505104931_2_0.png' not  in s3_img:
#         continue 
#     if not annotations:
#         continue
#     first_ann = next(iter(annotations.values()))

#     # Extract fields
#     extracted = {}
#     ocr_results = []
    
    
#     for ann in annotations.values():
        
#         points = ann.get("points", [])
#         # print(points)
#         if len(points) == 0:
#             continue
#         try:
#             x1, y1, x2, y2 = points[0][0], points[0][1], points[2][0], points[2][1]
#         except:
#             print(points)
#             # continue
#         x1 = x1 * first_ann["image_width"]/100
#         y1 = y1 * first_ann["image_height"]/100
#         x2 = x2 * first_ann["image_width"]/100
#         y2 = y2 * first_ann["image_height"]/100
#         width = x2 - x1
#         height = y2 - y1
        
#         # print(x1, y1, x2, y2, width, hceight)
        
#         text = ann.get("text", [""])[0] if ann.get("text") else ""
#         if text == "":
#             continue
#         # print(text)
#         ocr_result = {'text': text, 'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}
#         ocr_results.append(ocr_result)
        
#         for field in fields:
#             if ann.get("label") == field:
#                 extracted[field] = ann.get("text", [""])[0] if ann.get("text") else ""
#                 break
#         else:
#             extracted[field] = None  # or "null" if you want string
    
#     ocr_results=ocr_results
#     formatted_text = format_ocr_output(ocr_results)
# prompt = (
# "<image>\n"
# "You are a document information extraction assistant.\n"
# f"See the attached document image. And read the formatted text of the document: {formatted_text}.\n"
# "Extract the required information from the document and return it in the following JSON structure.\n"
# "Use the provided field descriptions and keyword hints for accurate matching.\n"
# "Return null if a field is missing. Do not add explanations.\n\n"
# "Output format:\n"
# "{\n"
# "  \"first_name\": \"Given name (e.g., 'Carolin')\",\n"
# "  \"family_name\": \"Family or last name (e.g., 'Balgenort')\",\n"
# "  \"title_name\": \"Title or honorific usually before the name, like Mr., Ms., Dr., Herr, Frau, etc.\ Use the title that is before the name, not after the name.\n"
# "  \"address_street\": \"Street name only, without house number (e.g., 'Hof im Hagen')\",\n"
# "  \"address_house_no\": \"House or building number (e.g., '7')\",\n"
# "  \"address_additional\": \"Optional address info (e.g., apartment, district)\",\n"
# "  \"address_zip\": \"Postal/ZIP code (e.g., '49134')\",\n"
# "  \"address_city\": \"City or townname (e.g., 'Wallenhorst'). If the city is not in the document, return 'null'.\n"
# "  \"SV_number\": \"Social security or pension number. Look for labels like 'SV-Nr.', 'RV-Nr.', or 'RV-Nummer'. Must be 12 characters, with 1 letter at position 10 (e.g., '50130984D504')\",\n"
# "  \"tax_id\": \"Tax identification number. Look for 'Steuer-ID', 'Steuer-Ident-Nr.'. It should be exactly 11 digits (e.g., '49285079139')\",\n"
# "  \"salary_month\": \"Salary period. Look for labels like 'Monat', 'für'. (e.g., '2025-04, April 2025')\",\n"
# "  \"gross_payment\": \"Monthly gross amount. Use value labeled like 'Gesamtbrutto' or 'Brutto'. Return number only, e.g., '3764.01'\",\n"
# "  \"net_payment\": \"Statutory net amount. Look for 'Gesetzliches Netto' or 'Netto'. Return number only, e.g., '3100.10'\",\n"
# "  \"real_payment\": \"Actual paid amount. Use value near bank name/number or 'Auszahlungsbetrag'. Return number only, e.g., '3100.10'\",\n"
# "  \"bank_account\": \"Bank account number or IBAN. Usually starts with 'DE' (Germany), near 'überwiesen' or 'Konto', usually 22-27 characters long\",\n"
# "  \"bank_name\": \"Bank name (e.g., 'Frankfurter Sparkasse'). Usually follows 'überwiesen bei' or after IBAN\",\n"
# "  \"company_name\": \"Company name. Look for label 'Firma' or names containing 'GmbH', 'AG', 'UG', 'oHG', 'BGB-Gesellschaft', 'Kommanditgesellschaft', etc.\"\n"
# "}\n"
# "If any field is missing or not visible in the document, set its value to null.\n"
# "Return only the JSON, with no explanation or commentary."
# )


# print(prompt)           
    

prompt_run="""
<image>
You are a document information extraction assistant.
See the attached document image. And read the formatted text of the document:           Bezügeabrechnung  für  R R Januar 2025 Januar 2025 Datum 30.04.2025 Seite 1
          IT|S |Care
          Saonestraße  3a, 60528 Frankfurt                    Pers .Nr.  50000523
          Rückfragen beantwortet beantwortet      Telefon            069/8303-442
          Andrea  Blohm                           Ein/Aus.  01.04.2017/
                                                  Geb.Datum            08.06.1976
                                                  Steuermerkmale   2/1,0    -/
                                                  Freib. J/M      0,00 ,    0,00
          Frankfurt                               Hinzb.J/M       0,00 /    0,00
          10001520                                Faktor
                                                  Steuer /SV-Tage           0 /0
          Frau                                    Priv.  SV
          Michaela Michaela Grosmann Grosmann     Beitragsgruppenschl.    0/1/1/1
          Foockenstraße Foockenstraße 62 62       Krankenkasse    Techniker-Krank Techniker-Krank
          65933 65933 Frankfurt Frankfurt         PV-Zu. /PV-Ki./Midi/Mfb. /2/  1
                                                  VBL/ ZVK-Nummer
                                                  Steuer  ID          79613504882 79613504882
                                                  RV-Nummer          12080676J520 12080676J520
                                                  Arbeitszeit         38,50/38,50
          BRUTTOENTGELTE
          Gesamtbrutto  (EBeschV)                                        6.922,00 6.922,00
          Steuer-Brutto,  lfd.                                           6.797,40
          SV-Brutto  KV/PV, lfd.                                         5.512,50
          SV-Brutto  RV, lfd.                                            6.797,40
          SV-Brutto  AV, lfd.                                            6.797,40
          Lohnsteuer,  lfd.                                  106,59-     1.266,41
          Rentenversicherung,  lfd.                                        632,16
          Arbeitslosenvers.,  lfd.                                          88,37
          Gesetz1.  Netto (EBeschV)                          106,59 106,59
          SONSTIGE  BE-/ABZÜGE
          Aufrollungsdifferenz                               141,04-
          AG-Zuschuss  KV                                    34,45         469,94
          AG-Zuschuss  PV                                                   99,23
          Aufrolld.  zur letzten Abr                         141,04-


          Kennz .: (E)inmalzahlung, (L)ohnsteuer-,  (S) V-pflichtig,(G)esamtbrutto
          Bescheinigung  nach § 108 Absatz 3 Satz 1 Gewerbeordnung












          Beitragszuschlag  zur PV wurde erhoben               0,00
          Die Bezüge  berechnen sich nach  Tarifgruppe   11  Stufe 05.
Extract the required information from the document and return it in the following JSON structure.
Use the provided field descriptions and keyword hints for accurate matching.
Return null if a field is missing. Do not add explanations.

Output format:
{
  "first_name": "Given name (e.g., 'Carolin')",
  "family_name": "Family or last name (e.g., 'Balgenort')",
  "title_name": "Title or honorific usually before the name, like Mr., Ms., Dr., Herr, Frau, etc.\ Use the title that is before the name, not after the name.
  "address_street": "Street name only, without house number (e.g., 'Hof im Hagen')",
  "address_house_no": "House or building number (e.g., '7')",
  "address_additional": "Optional address info (e.g., apartment, district)",
  "address_zip": "Postal/ZIP code (e.g., '49134')",
  "address_city": "City or townname (e.g., 'Wallenhorst'). If the city is not in the document, return 'null'.
  "SV_number": "Social security or pension number. Look for labels like 'SV-Nr.', 'RV-Nr.', or 'RV-Nummer'. Must be 12 characters, with 1 letter at position 10 (e.g., '50130984D504')",
  "tax_id": "Tax identification number. Look for 'Steuer-ID', 'Steuer-Ident-Nr.'. It should be exactly 11 digits (e.g., '49285079139')",
  "salary_month": "Salary period. Look for labels like 'Monat', 'für'. (e.g., '2025-04, April 2025')",
  "gross_payment": "Monthly gross amount. Use value labeled like 'Gesamtbrutto' or 'Brutto'. Return number only, e.g., '3764.01'",
  "net_payment": "Statutory net amount. Look for 'Gesetzliches Netto' or 'Netto'. Return number only, e.g., '3100.10'",
  "real_payment": "Actual paid amount. Use value near bank name/number or 'Auszahlungsbetrag'. Return number only, e.g., '3100.10'",
  "bank_account": "Bank account number or IBAN. Usually starts with 'DE' (Germany), near 'überwiesen' or 'Konto', usually 22-27 characters long",
  "bank_name": "Bank name (e.g., 'Frankfurter Sparkasse'). Usually follows 'überwiesen bei' or after IBAN",
  "company_name": "Company name. Look for label 'Firma' or names containing 'GmbH', 'AG', 'UG', 'oHG', 'BGB-Gesellschaft', 'Kommanditgesellschaft', etc."
}
If any field is missing or not visible in the document, set its value to null.
Return only the JSON, with no explanation or commentary.
"""


prompt="""
<image>
You are a document information extraction assistant.
See the attached document image. And read the formatted text of the document:           Bezügeabrechnung  für  R R Januar 2025 Januar 2025 Datum 30.04.2025 Seite 1
          IT|S |Care
          Saonestraße  3a, 60528 Frankfurt                    Pers .Nr.  50000523
          Rückfragen beantwortet beantwortet      Telefon            069/8303-442
          Andrea  Blohm                           Ein/Aus.  01.04.2017/
                                                  Geb.Datum            08.06.1976
                                                  Steuermerkmale   2/1,0    -/
                                                  Freib. J/M      0,00 ,    0,00
          Frankfurt                               Hinzb.J/M       0,00 /    0,00
          10001520                                Faktor
                                                  Steuer /SV-Tage           0 /0
          Frau                                    Priv.  SV
          Michaela Michaela Grosmann Grosmann     Beitragsgruppenschl.    0/1/1/1
          Foockenstraße  62                       Krankenkasse    Techniker-Krank
          65933  Frankfurt                        PV-Zu. /PV-Ki./Midi/Mfb. /2/  1
                                                  VBL/ ZVK-Nummer
                                                  Steuer  ID          79613504882
                                                  RV-Nummer          12080676J520
                                                  Arbeitszeit         38,50/38,50
          BRUTTOENTGELTE
          Gesamtbrutto  (EBeschV)                                        6.922,00
          Steuer-Brutto,  lfd.                                           6.797,40
          SV-Brutto  KV/PV, lfd.                                         5.512,50
          SV-Brutto  RV, lfd.                                            6.797,40
          SV-Brutto  AV, lfd.                                            6.797,40
          Lohnsteuer,  lfd.                                  106,59-     1.266,41
          Rentenversicherung,  lfd.                                        632,16
          Arbeitslosenvers.,  lfd.                                          88,37
          Gesetz1.  Netto (EBeschV)                          106,59
          SONSTIGE  BE-/ABZÜGE
          Aufrollungsdifferenz                               141,04-
          AG-Zuschuss  KV                                     34,45        469,94
          AG-Zuschuss  PV                                                   99,23
          Aufrolld.  zur letzten Abr                         141,04-


          Kennz .: (E)inmalzahlung, (L)ohnsteuer-,  (S) V-pflichtig,(G)esamtbrutto
          Bescheinigung  nach § 108 Absatz 3 Satz 1 Gewerbeordnung












          Beitragszuschlag  zur PV wurde erhoben               0,00
          Die Bezüge  berechnen sich nach  Tarifgruppe   11  Stufe 05.
Extract the required information from the document and return it in the following JSON structure.
Use the provided field descriptions and keyword hints for accurate matching.
Return null if a field is missing. Do not add explanations.

Output format:
{
  "first_name": "Given name (e.g., 'Carolin')",
  "family_name": "Family or last name (e.g., 'Balgenort')",
  "title_name": "Title or honorific usually before the name, like Mr., Ms., Dr., Herr, Frau, etc.\ Use the title that is before the name, not after the name.
  "address_street": "Street name only, without house number (e.g., 'Hof im Hagen')",
  "address_house_no": "House or building number (e.g., '7')",
  "address_additional": "Optional address info (e.g., apartment, district)",
  "address_zip": "Postal/ZIP code (e.g., '49134')",
  "address_city": "City or townname (e.g., 'Wallenhorst'). If the city is not in the document, return 'null'.
  "SV_number": "Social security or pension number. Look for labels like 'SV-Nr.', 'RV-Nr.', or 'RV-Nummer'. Must be 12 characters, with 1 letter at position 10 (e.g., '50130984D504')",
  "tax_id": "Tax identification number. Look for 'Steuer-ID', 'Steuer-Ident-Nr.'. It should be exactly 11 digits (e.g., '49285079139')",
  "salary_month": "Salary period. Look for labels like 'Monat', 'für'. (e.g., '2025-04, April 2025')",
  "gross_payment": "Monthly gross amount. Use value labeled like 'Gesamtbrutto' or 'Brutto'. Return number only, e.g., '3764.01'",
  "net_payment": "Statutory net amount. Look for 'Gesetzliches Netto' or 'Netto'. Return number only, e.g., '3100.10'",
  "real_payment": "Actual paid amount. Use value near bank name/number or 'Auszahlungsbetrag'. Return number only, e.g., '3100.10'",
  "bank_account": "Bank account number or IBAN. Usually starts with 'DE' (Germany), near 'überwiesen' or 'Konto', usually 22-27 characters long",
  "bank_name": "Bank name (e.g., 'Frankfurter Sparkasse'). Usually follows 'überwiesen bei' or after IBAN",
  "company_name": "Company name. Look for label 'Firma' or names containing 'GmbH', 'AG', 'UG', 'oHG', 'BGB-Gesellschaft', 'Kommanditgesellschaft', etc."
}
If any field is missing or not visible in the document, set its value to null.
Return only the JSON, with no explanation or commentary.
"""

image = Image.open(img_path).convert("RGB")

m_response=process_image(image,prompt)

print(m_response)

