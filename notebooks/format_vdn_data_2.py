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
    entry["image"] = [local_img_path]
    # pprint(annotations)
    # Get width/height from any annotation (they are the same for all)
    if not annotations:
        continue
    first_ann = next(iter(annotations.values()))
    entry["width_list"] = [first_ann["image_width"]]
    entry["height_list"] = [first_ann["image_height"]]

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
        "<image>\nExtract the following information in this follow json format: "
        "first_name, family_name, address_street,address_house_no, address_zip, address_city, "
        "SV_number, tax_id, salary_month,gross_payment, real_payment, net_payment,bank_account, "
        "bank_name, title_name,company_name, address_additional"
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