import json
import os
import random
from pprint import pprint
# Path to your input and output files
input_path = "/data1/hang/Stellantis/processed_vdn_data_2.json"
local_image_dir = "/data1/hang/Stellantis/VDN_annotated"  # adjust if needed
output_train = "/data1/hang/Stellantis/InternVL/notebooks/vdn_chat_train_only_fieldtext.jsonl"
output_valid = "/data1/hang/Stellantis/InternVL/notebooks/vdn_chat_valid_only_fieldtext.jsonl"

fields = [
    'first_name', 'family_name', 'address_street', 'address_house_no', 'address_zip', 'address_city',
    'SV_number', 'tax_id', 'salary_month', 'gross_payment', 'real_payment', 'net_payment',
    'bank_account', 'bank_name', 'title_name', 'company_name', 'address_additional'
]

with open(input_path, "r", encoding="utf-8") as f:
    vdn_data = json.load(f)

formatted_data = []
for idx, (s3_img, annotations) in enumerate(vdn_data.items()):
    entry = {"id": idx}
    local_img_path = os.path.join(local_image_dir, os.path.basename(s3_img))
    entry["image"] = local_img_path
    # pprint(annotations)
    # Get width/height from any annotation (they are the same for all)
    if not annotations:
        continue
    first_ann = next(iter(annotations.values()))
    entry["width_list"] = first_ann["image_width"]
    entry["height_list"] = first_ann["image_height"]

    # Extract fields
    extracted = {}
    for field in fields:
        for ann in annotations.values():
            if ann.get("label") == field:
                extracted[field] = ann.get("text", [""])[0] if ann.get("text") else ""
                break
        else:
            extracted[field] = None  # or "null" if you want string

    # Conversation
    prompt = (
    "<image>\n"
    "You are a document information extraction assistant.\n"
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

# Shuffle and split
random.seed(42)
random.shuffle(formatted_data)
split_idx = int(len(formatted_data) * 0.85)
train_data = formatted_data[:split_idx]
valid_data = formatted_data[split_idx:]

# Write JSONL
for path, data in [(output_train, train_data), (output_valid, valid_data)]:
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Train: {len(train_data)}, Valid: {len(valid_data)}") 