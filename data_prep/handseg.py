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
import argparse
# Set up logging configuration
logging.basicConfig(filename='output.log', level=logging.INFO)
# Add the following line at $PLACEHOLDER$
logging.info('This is hand segmentation code')

def cannon(root_dir,model_path):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"

    CHECKPOINT_PATH=model_path
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    mask_predictor = SamPredictor(sam)

    subj_dirs=[os.path.join(root_dir,item) for item in utils.list_subdirectories(root_dir) if item[0].lower()=='p']
    for subj_dir in subj_dirs:
        session_dirs=[os.path.join(subj_dir,session_dir) for session_dir in utils.list_subdirectories(subj_dir) if session_dir[0].lower()=='s']
        for session_dir in session_dirs:
            print(session_dir)
            logging.info(f"Processing session directory: {session_dir}")
            img_dir_=utils.list_subdirectories(session_dir)
            if len(img_dir_)==0:
                print(f'{session_dir} does not exist. Continuing...')
                continue
            img_dir=os.path.join(session_dir,img_dir_[0])
            img_files=utils.list_files(img_dir,'jpg')
            last_bbx=0
            
            hand_bb_path=os.path.join(session_dir,'hand_bbs.json')
            if not os.path.exists(hand_bb_path):
                print(f'{hand_bb_path} does not exist')
                logging.info(f'{hand_bb_path} does not exist')
                continue
            with open(hand_bb_path, 'r') as file:
                data = json.load(file)
            hand_mask_dir=os.path.join(session_dir,'hand_mask')
            # if os.path.exists(hand_mask_dir):
            #     print(f'{hand_mask_dir} exists, skipping')
            #     continue
            if not os.path.exists(hand_mask_dir):
                os.makedirs(hand_mask_dir)

            for i,img_file in enumerate(img_files):
                print(f'Processing {i}/{len(img_files)}',end='\r')
                hand_mask_img_path=os.path.join(hand_mask_dir,os.path.splitext(img_file)[0]+'.png')
                if os.path.exists(hand_mask_img_path):
                    continue
                bbx_str=data[img_file.split('.')[0]]
                if bbx_str=="":
                    bbx=last_bbx
                else:
                    bbx=[int(item) for item in bbx_str.split(',')]
                    last_bbx=bbx

                image_path = os.path.join(img_dir, img_file)
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
            print(session_dir)
            logging.info(f'{session_dir}')
            hand_bb_path=os.path.join(session_dir,'kinect','hand_bbs.json')
            if not os.path.exists(hand_bb_path):
                print(f'{hand_bb_path} does not exist')
                logging.info(f'{hand_bb_path} does not exist')
                continue
            with open(hand_bb_path, 'r') as file:
                data = json.load(file)
            hand_mask_dir=os.path.join(session_dir,'kinect','hand_mask')
            # if os.path.exists(hand_mask_dir):
            #     print(f'{hand_mask_dir} exists, skipping')
            #     continue
            if not os.path.exists(hand_mask_dir):
                os.makedirs(hand_mask_dir)  

            img_files=utils.list_files(os.path.join(session_dir,'kinect','color'),'jpg')
            last_bbx=0
            for i,img_file in enumerate(img_files):
                print(f'Processing {i}/{len(img_files)}',end='\r')
                hand_mask_img_path=os.path.join(hand_mask_dir,os.path.splitext(img_file)[0]+'.png')
                if os.path.exists(hand_mask_img_path):
                    continue
                bbx_str=data[img_file.split('.')[0]]
                if bbx_str=="":
                    bbx=last_bbx
                else:
                    bbx=[int(item) for item in bbx_str.split(',')]
                    last_bbx=bbx

                image_path = os.path.join(session_dir, 'kinect','color', img_file)
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=r'/p/blurdepth/models/sam_vit_h_4b8939.pth',help="Path to the model checkpoint file")
    parser.add_argument("--data", default=r'/p/blurdepth/data/canon_images/', help="Path to the data directory")
    args = parser.parse_args()

    cannon(args.data, args.model)







    


