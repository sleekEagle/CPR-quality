from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import numpy as np
import os
import sys
sys.path.append('C:\\Users\\lahir\\code\\CPR-quality\\')
import utils
import json

def get_bb(results):
    bbx_list=results.xyxy
    conf_list=results.confidence
    max_conf_arg=np.argmax(conf_list)
    bb=bbx_list[max_conf_arg]
    return bb

base_model = GroundingDINO(ontology=CaptionOntology({"hand": "hand"}))
root_dir='D:\\CPR_data_raw'
subj_dirs=[os.path.join(root_dir,item,'extracted') for item in utils.list_subdirectories(root_dir) if item[0].lower()=='p']
for subj_dir in subj_dirs:
    session_dirs=[os.path.join(subj_dir,session_dir) for session_dir in utils.list_subdirectories(subj_dir) if session_dir[0].lower()=='s']
    for session_dir in session_dirs:
        hand_bbs={}
        print(session_dir)
        hand_bbs_path=os.path.join(session_dir,'hand_bbs.json')
        if os.path.exists(hand_bbs_path):
            print(f'{hand_bbs_path} exists, skipping')
            continue
        img_dir=os.path.join(session_dir,'color')
        img_files=utils.list_files(img_dir,'jpg')
        for i,img_file in enumerate(img_files):
            print(f'Processing {i}/{len(img_files)}')
            img_path=os.path.join(img_dir,img_file)
            results = base_model.predict(img_path)
            bb=get_bb(results)
            assert len(bb)>0, f'No bounding box found for {img_file}'
            bbstr= ','.join([str(int(item)) for item in list(bb)])
            hand_bbs[img_file.split('.')[0]]=bbstr
            # Write the JSON string to the output file
        with open(hand_bbs_path, 'w') as file:
            file.write(json.dumps(hand_bbs))

            