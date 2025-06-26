import math
import json
import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import numpy as np

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

def visualize_ocr_output(image_path, ocr_results):
    """
    Visualizes OCR results (list of word dicts with text, x, y, width, height)
    using matplotlib.
    """
    if not ocr_results:
        return

    # Load the image
    image = Image.open(image_path)

    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Plot each word as a rectangle
    
    for word in ocr_results:
        x1, y1, x2, y2 = word['x'], word['y'], word['x'] + word['width'], word['y'] + word['height']
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        points = np.array(points, dtype=np.int32)
        cv2.polylines(image_cv, [points], True, (0, 0, 255), 2)
        # cv2.rectangle(image_cv, (word['x'], word['y']), (word['x'] + word['width'], word['y'] + word['height']), (0, 0, 255), 2)
        cv2.putText(image_cv, word['text'], (points[0][0], points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # break
    # Set axis limits and labels
    #save the figure
    cv2.imwrite('ocr_output.png', image_cv)

# --- Example Usage ---
# You would populate this list with your actual OCR data
# including 'x', 'y', 'width', 'height' for each word.
# This is a tiny snippet for demonstration.
# For accurate results, ALL words from the image need coordinates.
local_image_dir = "/data1/hang/Stellantis/VDN_annotated" 
fields = [
    'first_name', 'family_name', 'address_street', 'address_house_no', 'address_zip', 'address_city',
    'SV_number', 'tax_id', 'salary_month', 'gross_payment', 'real_payment', 'net_payment',
    'bank_account', 'bank_name', 'title_name', 'company_name', 'address_additional'
]
with open('/data1/hang/Stellantis/processed_vdn_data_2.json', "r", encoding="utf-8") as f:
    data = json.load(f)
    
formatted_data = []
for idx, (s3_img, annotations) in tqdm(enumerate(data.items())):
    
    entry = {'id': idx}
    local_img_path = os.path.join(local_image_dir, os.path.basename(s3_img))
    
    entry["image"] = local_img_path
    if not annotations:
        continue
    first_ann = next(iter(annotations.values()))
    entry["width_list"] = first_ann["image_width"]
    entry["height_list"] = first_ann["image_height"]
    print('width, height:',first_ann["image_width"], first_ann["image_height"])

    # Extract fields
    extracted = {}
    ocr_results = []
    
    
    for ann in annotations.values():
        
        points = ann.get("points", [])
        # print(points)
        if len(points) == 0:
            continue
        try:
            x1, y1, x2, y2 = points[0][0], points[0][1], points[2][0], points[2][1]
        except:
            print(points)
            # continue
        x1 = x1 * first_ann["image_width"]/100
        y1 = y1 * first_ann["image_height"]/100
        x2 = x2 * first_ann["image_width"]/100
        y2 = y2 * first_ann["image_height"]/100
        width = x2 - x1
        height = y2 - y1
        
        # print(x1, y1, x2, y2, width, hceight)
        
        text = ann.get("text", [""])[0] if ann.get("text") else ""
        if text == "":
            continue
        # print(text)
        ocr_result = {'text': text, 'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}
        ocr_results.append(ocr_result)
        
        for field in fields:
            if ann.get("label") == field:
                extracted[field] = ann.get("text", [""])[0] if ann.get("text") else ""
                break
        else:
            extracted[field] = None  # or "null" if you want string
            
    # visualize_ocr_output(local_img_path, ocr_results)
    print(local_img_path)
    formatted_text = format_ocr_output(ocr_results)
    # print(formatted_text)
    prompt = (
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
    "  \"title_name\": \"Title or honorific usually before the name, like Mr., Ms., Dr., Herr, Frau, etc.\ Use the title that is before the name, not after the name.\n"
    "  \"address_street\": \"Street name only, without house number (e.g., 'Hof im Hagen')\",\n"
    "  \"address_house_no\": \"House or building number (e.g., '7')\",\n"
    "  \"address_additional\": \"Optional address info (e.g., apartment, district)\",\n"
    "  \"address_zip\": \"Postal/ZIP code (e.g., '49134')\",\n"
    "  \"address_city\": \"City or townname (e.g., 'Wallenhorst'). If the city is not in the document, return 'null'.\n"
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

    entry["conversations"] = [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": json.dumps(extracted, ensure_ascii=False)}
    ]
    formatted_data.append(entry)

    # break

    
    
mock_ocr_results = [
    # Header Line 1 (Approximation)
    {'text': 'Manage', 'x': 275, 'y': 30, 'width': 50, 'height': 12},
    {'text': 'your', 'x': 330, 'y': 30, 'width': 30, 'height': 12},
    {'text': 'account', 'x': 365, 'y': 30, 'width': 55, 'height': 12},
    {'text': 'online', 'x': 425, 'y': 30, 'width': 40, 'height': 12},
    {'text': 'at:', 'x': 470, 'y': 30, 'width': 20, 'height': 12},
    {'text': 'Customer', 'x': 545, 'y': 30, 'width': 60, 'height': 12},
    {'text': 'Service:', 'x': 610, 'y': 30, 'width': 55, 'height': 12},
    {'text': 'Mobile:', 'x': 710, 'y': 30, 'width': 45, 'height': 12},
    {'text': 'Download', 'x': 760, 'y': 30, 'width': 65, 'height': 12},
    {'text': 'the', 'x': 830, 'y': 30, 'width': 25, 'height': 12},

    # Header Line 2
    {'text': 'www.chase.com/cardhelp', 'x': 360, 'y': 45, 'width': 150, 'height': 12},
    {'text': '1-800-524-3880', 'x': 570, 'y': 45, 'width': 100, 'height': 12},
    {'text': 'Chase', 'x': 760, 'y': 45, 'width': 40, 'height': 12},
    {'text': 'Mobile', 'x': 805, 'y': 45, 'width': 45, 'height': 12},
    {'text': '®', 'x': 850, 'y': 45, 'width': 10, 'height': 12},
    {'text': 'app', 'x': 865, 'y': 45, 'width': 25, 'height': 12},
    {'text': 'today', 'x': 895, 'y': 45, 'width': 40, 'height': 12},

    # Freedom logo (might be image, text depends on OCR)
    {'text': 'freedom', 'x': 90, 'y': 60, 'width': 60, 'height': 15},

    # Gap
    {'text': 'YOUR', 'x': 20, 'y': 90, 'width': 40, 'height': 12},
    {'text': 'ACCOUNT', 'x': 65, 'y': 90, 'width': 60, 'height': 12},
    {'text': 'MESSAGES', 'x': 130, 'y': 90, 'width': 70, 'height': 12},
    {'text': '(CONTINUED)', 'x': 205, 'y': 90, 'width': 80, 'height': 12},

    # Message line 1
    {'text': 'Your', 'x': 20, 'y': 105, 'width': 30, 'height': 12},
    {'text': 'AutoPay', 'x': 55, 'y': 105, 'width': 55, 'height': 12},
    # ... add ALL other words with their coordinates ...

    # Example Table Row
    {'text': '01/04', 'x': 25, 'y': 210, 'width': 40, 'height': 12},
    {'text': 'LARRY', 'x': 180, 'y': 210, 'width': 45, 'height': 12},
    {'text': 'HOPKINS', 'x': 230, 'y': 210, 'width': 55, 'height': 12},
    {'text': 'HONDA', 'x': 290, 'y': 210, 'width': 45, 'height': 12},
    {'text': '7074304151', 'x': 340, 'y': 210, 'width': 75, 'height': 12},
    {'text': 'CA', 'x': 420, 'y': 210, 'width': 20, 'height': 12},
    {'text': '265.40', 'x': 700, 'y': 210, 'width': 50, 'height': 12}, # Note the large X jump for the amount

     # Footer Example
    {'text': 'LARRY', 'x': 10, 'y': 700, 'width': 45, 'height': 12},
    {'text': 'PAGE', 'x': 60, 'y': 700, 'width': 40, 'height': 12},
    {'text': 'Page', 'x': 440, 'y': 700, 'width': 35, 'height': 12},
    {'text': '2', 'x': 480, 'y': 700, 'width': 10, 'height': 12},
    {'text': 'of', 'x': 495, 'y': 700, 'width': 15, 'height': 12},
    {'text': '3', 'x': 515, 'y': 700, 'width': 10, 'height': 12},
    {'text': 'Statement', 'x': 670, 'y': 700, 'width': 65, 'height': 12},
    {'text': 'Date:', 'x': 740, 'y': 700, 'width': 35, 'height': 12},
    {'text': '02/03/24', 'x': 780, 'y': 700, 'width': 60, 'height': 12},
]

existed_train = "/data1/hang/Stellantis/InternVL/notebooks/vdn_chat_train_only_fieldtext.jsonl"
existed_valid = "/data1/hang/Stellantis/InternVL/notebooks/vdn_chat_valid_only_fieldtext.jsonl"

with open(existed_train, "r", encoding="utf-8") as f:
    existed_train_data = [json.loads(line) for line in f]
with open(existed_valid, "r", encoding="utf-8") as f:
    existed_valid_data = [json.loads(line) for line in f]

existed_train_images = [item['image'] for item in existed_train_data]
existed_valid_images = [item['image'] for item in existed_valid_data]

output_train = "/data1/hang/Stellantis/InternVL/notebooks/vdn_chat_train_fieldtext_ocr.jsonl"
output_valid = "/data1/hang/Stellantis/InternVL/notebooks/vdn_chat_valid_fieldtext_ocr.jsonl"

train_data = []
valid_data = []
for item in formatted_data:
    if item['image'] in existed_train_images and item['image'] not in existed_valid_images:
        train_data.append(item)
    elif item['image'] in existed_valid_images and item['image'] not in existed_train_images:
        valid_data.append(item)
    else:
        print(item['image'])


# print(len(train_data), len(valid_data))
for path, data in [(output_train, train_data), (output_valid, valid_data)]:
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"Train: {len(train_data)}, Valid: {len(valid_data)}") 





# --- Run the formatting ---
# Important: You need to replace mock_ocr_results with your FULL, accurate OCR data
# The quality depends entirely on the accuracy of the OCR coordinates and text.
# formatted_text = fc