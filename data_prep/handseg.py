import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append('C:\\Users\\lahir\\code\\CPR-quality\\')
import utils
import json
import cv2
import torch
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor
import numpy as np
from functools import reduce
from scipy.ndimage.morphology import binary_fill_holes
import sys
import logging
# Set up logging configuration
logging.basicConfig(filename='output.log', level=logging.INFO)
# Add the following line at $PLACEHOLDER$
logging.info('This is hand segmentation code')

def main():
    # model_path e.g: C:\Users\lahir\code\CPR-quality\sam_vit_h_4b8939.pth
    # data_path e.g : D:\CPR_data_raw
    if len(sys.argv)==1:
        model_path=r'C:\Users\lahir\code\CPR-quality\sam_vit_h_4b8939.pth'
        data_path=r'D:\CPR_extracted'
    else:
        model_path = sys.argv[1]
        data_path = sys.argv[2]

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    CHECKPOINT_PATH=model_path
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    mask_predictor = SamPredictor(sam)

    root_dir=data_path
    subj_dirs=[os.path.join(root_dir,item) for item in utils.list_subdirectories(root_dir) if item[0].lower()=='p']
    for subj_dir in subj_dirs:
        session_dirs=[os.path.join(subj_dir,session_dir) for session_dir in utils.list_subdirectories(subj_dir) if session_dir[0].lower()=='s']
        for session_dir in session_dirs:
            if session_dir!=r'D:\CPR_extracted\P21\s_6':
                continue
            print(session_dir)
            logging.info(f'{session_dir}')
            hand_bb_path=os.path.join(session_dir,'hand_bbs.json')
            if not os.path.exists(hand_bb_path):
                print(f'{hand_bb_path} does not exist')
                logging.info(f'{hand_bb_path} does not exist')
                continue
            with open(hand_bb_path, 'r') as file:
                data = json.load(file)
            hand_mask_dir=os.path.join(session_dir,'hand_mask')
            os.makedirs(hand_mask_dir,exist_ok=True)

            img_files=utils.list_files(os.path.join(session_dir,'color'),'jpg')
            last_bbx=0
            for img_file in img_files:
                print(img_file)
                hand_mask_img_path=os.path.join(hand_mask_dir,os.path.splitext(img_file)[0]+'.png')
                bbx_str=data[img_file.split('.')[0]]
                if bbx_str=="":
                    bbx=last_bbx
                else:
                    bbx=[int(item) for item in bbx_str.split(',')]
                    last_bbx=bbx
                if os.path.exists(hand_mask_img_path):
                    continue

                image_path = os.path.join(session_dir, 'color', img_file)
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask_predictor.set_image(image_rgb)
                masks, scores, logits = mask_predictor.predict(
                box=np.array(bbx),
                multimask_output=True
                )

                mask=np.any(masks, axis=0)*255
                mask=mask.astype(np.uint8)
                cv2.imwrite(hand_mask_img_path, mask)            


if __name__ == "__main__":
    main()

        







    


