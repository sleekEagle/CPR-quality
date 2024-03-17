from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import json
import time
import logging

# Set up logging configuration
logging.basicConfig(filename='output.log', level=logging.INFO)
# Add the following line at $PLACEHOLDER$
logging.info('This is a log message')


def get_bb(results):
    bbx_list=results.xyxy
    conf_list=results.confidence
    if len(conf_list)==0:
        return []
    max_conf_arg=np.argmax(conf_list)
    bb=bbx_list[max_conf_arg]
    return bb

def main():
    if len(sys.argv)==1:
        root_dir=r'\\samba.cs.virginia.edu\p\blurdepth\data\canon_images'
    else:
        root_dir = sys.argv[1]
    base_model = GroundingDINO(ontology=CaptionOntology({"hand": "hand"}))
    subj_dirs=[os.path.join(root_dir,item) for item in utils.list_subdirectories(root_dir) if item[0].lower()=='p']
    for subj_dir in subj_dirs:
        session_dirs=[os.path.join(subj_dir,session_dir) for session_dir in utils.list_subdirectories(subj_dir) if session_dir[0].lower()=='s']
        sleep=True
        for session_dir in session_dirs:
            hand_bbs={}
            print(session_dir)
            logging.info(f"Processing session directory: {session_dir}")
            hand_bbs_path=os.path.join(session_dir,'kinect','hand_bbs.json')
            if os.path.exists(hand_bbs_path):
                print(f'{hand_bbs_path} exists, skipping')
                sleep=False
                continue
            img_dir=os.path.join(session_dir,'kinect','color')
            img_files=utils.list_files(img_dir,'jpg')
            for i,img_file in enumerate(img_files):
                print(f'Processing {i}/{len(img_files)}')
                logging.info(f'Processing {i}/{len(img_files)}')
                img_path=os.path.join(img_dir,img_file)
                results = base_model.predict(img_path)
                bb=get_bb(results)
                if len(bb)==0:
                    print(f'No bounding box found for {img_file}')
                    hand_bbs[img_file.split('.')[0]]=""
                else:
                    bbstr= ','.join([str(int(item)) for item in list(bb)])
                    hand_bbs[img_file.split('.')[0]]=bbstr
            # Write the JSON string to the output file
            with open(hand_bbs_path, 'w') as file:
                file.write(json.dumps(hand_bbs))

if __name__ == "__main__":
    main()

            