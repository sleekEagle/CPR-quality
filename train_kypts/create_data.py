import json
import os
import numpy as np
import shutil
import cv2
import sys
sys.path.append('C:\\Users\\lahir\\code\\CPR-quality\\')
import utils
import sys
import matplotlib.pyplot as plt

data_root='D:\CPR_extracted'
kypt_data_root="D:\kypt_dataset"
split_file=os.path.join(kypt_data_root,'splits.json')

# Read the JSON file
with open(split_file, 'r') as file:
    splits = json.load(file)

len(splits['test'])


pad=80
#copy data from the source dir to kypt dir
def copy_images(split):
    dest_dir=os.path.join(kypt_data_root,'images',split)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    file_names=[]
    for session_dir in splits[split]:
        print(session_dir)
        if session_dir.startswith('/'):
                session_dir = session_dir[1:]
        img_dir=os.path.join(data_root,session_dir,'kinect','color')
        files=os.listdir(img_dir)
        files.sort()
        idx=np.arange(0,len(files))
        selected_idx = idx[::int(len(idx)/25)]
        files_sel=[files[i] for i in selected_idx]

        file_list = [os.path.join(os.path.normpath(img_dir),file) for file in files_sel if file.endswith('.jpg')]
        #read hand bbs
        hand_bb_file=os.path.normpath(os.path.join(data_root,session_dir,'kinect','hand_bbs.json'))
        with open(hand_bb_file, 'r') as file:
            hand_bbs = json.load(file)
        last_bb=-1
        for file in file_list:
            imgname=os.path.basename(file).split('.')[0]
            if imgname in hand_bbs:
                bb = hand_bbs[imgname]
                last_bb=bb
            else:
                bb=last_bb
            dest_name = '_'.join(os.path.dirname(file).split(os.path.sep)[-4:-2])+'_'+os.path.basename(file)
            destination = os.path.join(dest_dir,dest_name)
            file_names.append(dest_name)
            image = cv2.imread(file)
            bb=[int(vals) for vals in bb.split(',')]
            img_crop= utils.crop_img_bb(image,bb,pad)
            cv2.imwrite(destination, img_crop)
    return file_names




def copy_all_images():
    dest_dir=os.path.join(kypt_data_root,'images')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    subj_dirs=utils.get_dirs_with_str(data_root, 'P')
    for sub_dir in subj_dirs:
        if os.path.basename(sub_dir) in  ['P0','P1','P10','P11']:
            continue
        session_dirs=utils.get_dirs_with_str(sub_dir,'s')
        for session_dir in session_dirs:
            print(session_dir)
            img_dir=os.path.join(data_root,sub_dir,session_dir,'kinect','color')
            img_files=utils.list_files(img_dir,'jpg')
            #read hand bbs
            hand_bb_file=os.path.normpath(os.path.join(data_root,session_dir,'kinect','hand_bbs.json'))
            with open(hand_bb_file, 'r') as file:
                hand_bbs = json.load(file)
            last_bb=-1
            for file in img_files:
                imgname=os.path.basename(file).split('.')[0]
                if imgname in hand_bbs:
                    bb = hand_bbs[imgname]
                    if bb=='':
                        bb=last_bb
                    else:
                        last_bb=bb
                dest_name = os.path.basename(sub_dir)+'_'+os.path.basename(session_dir)+'_'+os.path.basename(file)
                destination = os.path.join(dest_dir,dest_name)
                image = cv2.imread(os.path.join(img_dir,file))
                bb=[int(vals) for vals in bb.split(',')]
                img_crop= utils.crop_img_bb(image,bb,pad)
                cv2.imwrite(destination, img_crop)

def plot_bbs():
    subj_dirs=utils.get_dirs_with_str(data_root, 'P')
    for sub_dir in subj_dirs:
        session_dirs=utils.get_dirs_with_str(sub_dir,'s')
        for session_dir in session_dirs:
            img_dir=os.path.join(data_root,sub_dir,session_dir,'kinect','color')
            img_files=utils.list_files(img_dir,'jpg')
            #read hand bbs
            hand_bb_file=os.path.normpath(os.path.join(data_root,session_dir,'kinect','hand_bbs.json'))
            with open(hand_bb_file, 'r') as file:
                hand_bbs = json.load(file)
            keys=[img.split(',')[0] for img in img_files]
            bbs_list=[hand_bbs[k].split(',') for k in list(hand_bbs.keys()) if hand_bbs[k]!='']

            x_list=[int(bbs[0]) for bbs in bbs_list]
            y_list=[int(bbs[1]) for bbs in bbs_list]

            x_diff=np.max(x_list)-np.min(x_list)
            y_diff=np.max(y_list)-np.min(y_list)
            diff=max(x_diff,y_diff)
            print(session_dir,diff)


'''
create train test split data for the kypt dataset
'''
names={}
train_names=copy_images('train')
test_names=copy_images('test')
names['train']=train_names
names['test']=test_names
# Save the names dictionary as a JSON file
json_file = os.path.join(kypt_data_root, 'split_files.json')
with open(json_file, 'w') as file:
    json.dump(names, file)

