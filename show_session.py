import os
import utils
import json
import cv2
import numpy as np
import sys
import argparse

def show_video(dir_path,show_kypts):
    keypt_file=os.path.join(dir_path,'kinect','wrist_keypts','hand_keypts_mediapipe.json')
    img_dir=os.path.join(dir_path,'kinect','color')
    imgs=utils.get_files_with_str(img_dir,'.jpg')
    with open(keypt_file, 'r') as json_file:
        kypt_dict = json.load(json_file)

    for i,img in enumerate(imgs):
        print(f'Processing {i}/{len(imgs)}',end='\r')
        img_key=os.path.basename(img).split('.')[0]
        image = cv2.imread(img)
        kypts=kypt_dict[img_key]
        if show_kypts and len(kypts)>0:
            x, y = np.array(kypts['x']), np.array(kypts['y'])
            for x_,y_ in zip(x,y):
                _=cv2.circle(image, (x_, y_), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.imshow('Image with Circles', image)
        cv2.waitKey(500)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="D:\\CPR_extracted\\", help="Path to data directory")
    parser.add_argument("-p", type=str, default="P0", help="Path to participant directory")
    parser.add_argument("-s", type=str, default="s_1", help="Path to session directory")
    parser.add_argument("--kypts", type=bool, default=False, help="Should kypts be displayed")

    args = parser.parse_args()

    dir_path=os.path.join(args.root_dir,args.p,args.s)

    show_video(dir_path,args.kypts)
