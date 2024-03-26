import csv
import json
import os

# Convert to COCO format
coco_format = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "supercategory": "hands",
            "id": 1,
            "name": "person",
            "keypoints": ["wrist"],
            "skeleton": []
        }
    ]
}

# Specify the path to your CSV file
csv_file = r'C:\Users\lahir\Downloads\test_kpts.csv'
# Open the CSV file
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    # Read all lines from the CSV file
    lines = list(csv_reader)

for i,line in enumerate(lines):
    if i==0:
        continue
    if len(line)!=8:
        continue
    img_name=line[4].split('-')[-1]
    data=line[5]
    data_dict = json.loads(data)[0]
    x_pos=float(data_dict['x'])
    y_pos=float(data_dict['y'])
    
    img_width=int(data_dict['original_width'])
    img_height=int(data_dict['original_height'])

    x_pos=x_pos/100*img_width
    y_pos=y_pos/100*img_height

    pad=0
    bbs=[0,0,img_width,img_height]

    coco_format["images"].append({"file_name": 'test/'+img_name,
        "height": img_height,
        "width": img_width,
        "id": i
    })
    ar=[[int(x_pos),int(y_pos),1]]*21
    coco_format["annotations"].append({
        "num_keypoints": 1,
        "keypoints": ar,
        "image_id": i,
        "id": i,
        "category_id": 16,
        "bbox":bbs
    })

# Specify the path to save the JSON file
json_file = r'C:\Users\lahir\Downloads\test.json'
# Save the coco_format dictionary as a JSON file
with open(json_file, 'w') as file:
    json.dump(coco_format, file)


