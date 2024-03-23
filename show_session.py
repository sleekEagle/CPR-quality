import os
import utils
import json
import cv2
import numpy as np
import sys
import argparse

def show_video(dir_path,show_kypts,bb):
    keypt_file=os.path.join(dir_path,'kinect','wrist_keypts','hand_keypts_mediapipe.json')
    img_dir=os.path.join(dir_path,'kinect','color')
    imgs=utils.get_files_with_str(img_dir,'.jpg')
    if show_kypts:
        with open(keypt_file, 'r') as json_file:
            kypt_dict = json.load(json_file)
    if bb:
        bb_file=os.path.join(dir_path,'kinect','hand_bbs.json')
        with open(bb_file, 'r') as json_file:
            bb_dict = json.load(json_file)

    last_bb=-1
    for i,img in enumerate(imgs):
        print(f'Processing {i}/{len(imgs)}',end='\r')
        img_key=os.path.basename(img).split('.')[0]
        image = cv2.imread(img)
        kypts=kypt_dict[img_key]
        if show_kypts and len(kypts)>0:
            x, y = np.array(kypts['x']), np.array(kypts['y'])
            for x_,y_ in zip(x,y):
                _=cv2.circle(image, (int(x_), int(y_)), radius=5, color=(0, 255, 0), thickness=-1)
        if bb:
            if img_key in bb_dict:
                bb=bb_dict[img_key]
                last_bb=bb
            else:
                bb=last_bb
            bb=[int(vals) for vals in bb.split(',')]
            _=cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
        # Scale down the image
        # scale_percent = 20  # Adjust the scale percentage as needed
        # width = int(image.shape[1] * scale_percent / 100)
        # height = int(image.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow('Image with Circles', image)
        cv2.waitKey(500)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="D:\\CPR_extracted\\", help="Path to data directory")
    parser.add_argument("-p", type=str, default="P18", help="Path to participant directory")
    parser.add_argument("-s", type=str, default="s_0", help="Path to session directory")
    parser.add_argument("--kypts", type=bool, default=True, help="Should kypts be displayed")
    parser.add_argument("--bb", type=bool, default=True, help="Should bounding bx be displayed")

    args = parser.parse_args()

    dir_path=os.path.join(args.root_dir,args.p,args.s)  

    show_video(dir_path,args.kypts,args.bb)
