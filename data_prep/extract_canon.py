import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import shutil
from datetime import datetime
import re
import logging
# Set up logging configuration
logging.basicConfig(filename='ext_canon.log', level=logging.INFO)
# Add the following line at $PLACEHOLDER$
logging.info('This is extract canon code')

def extract_time_str(txt):
    x = re.findall("[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]", txt)
    if len(x)>0:
        return x[0]
    else: 
         return None

def get_ts_google(image_path):
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
            content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if len(texts)==0:
        return None
    for text in texts:
        match=extract_time_str(text.description)
        if type(match)==str:
            break
    return match


# get_ts_google(r'D:\CPR_data_raw\P0\s1\canon\MVI_1007\1955.png')

#308 : 46001.781   12:46:41.781
#1: 45991.533  12:46:31.533

#start idx is 1. not 0
# F_RATE=30
# time_str='12:47:38.109'
# time_idx=308
# end_idx=4077
# f_time=utils.get_float_time(datetime.strptime(time_str.strip(), "%H:%M:%S.%f").time())
# #create sequence of timestamps and idx

# start_f_time=f_time-(time_idx-1)*1/F_RATE
# idx_range = range(0, end_idx, 1)
# timestamps = [start_f_time + idx * 1/F_RATE for idx in idx_range]
                        
def extract_ts():
    root_path='D:\\CPR_data_raw'
    subj_dirs=utils.get_dirs_with_str(root_path, 'P')
    for subj in subj_dirs:
        session_dirs=utils.get_dirs_with_str(subj, 's')
        for sess in session_dirs:
            canon_dir=os.path.join(sess, 'canon')
            mvi_dirs=utils.get_dirs_with_str(canon_dir, 'MVI',i=0,j=5)
            for mvi in mvi_dirs:
                print('processing: ', mvi)
                img_files=utils.list_files(mvi, 'png')
                out_file=os.path.join(mvi, 'timestamps.txt')
                with open(out_file, 'w') as f:
                    for i,img in enumerate(img_files):
                        print(f'{i} out of {len(img_files)} is done.',end='\r')
                        img_path = os.path.join(mvi, img)
                        ts = get_ts_google(img_path)
                        if ts:
                            with open(out_file, 'a') as f:
                                f.write(img + ',' + ts + '\n')

def copy_mask_files():
    sounrce_path='/p/blurdepth/data/CPR_extracted_maks'
    dest_path='/p/blurdepth/data/hand_masks'
    sub_dirs=utils.get_dirs_with_str(sounrce_path, 'P')
    for s in sub_dirs:
        session_dirs=utils.get_dirs_with_str(os.path.join(sounrce_path,s), 's')
        for session in session_dirs:
            print('processing: ', session)
            logging.info(f'{session}')
            mask_dir=os.path.join(session, 'hand_mask')
            if os.path.isdir(mask_dir):
                dest_dir = os.path.join(dest_path, os.path.basename(s),os.path.basename(session), 'hand_mask')
                if os.path.exists(dest_dir):
                    shutil.rmtree(dest_path)
                print('copping to ', dest_dir)
                shutil.copytree(mask_dir, dest_dir)

def get_ffmpeg_cmds():
    path='D:\\CPR_data_raw\\'
    out_path='D:\\canon_images\\'
    sub_dirs=utils.get_dirs_with_str(path, 'P')
    for s in sub_dirs:
        session_dirs=utils.get_dirs_with_str(os.path.join(path,s), 's')
        for session in session_dirs:
            canon_dir=os.path.join(session, 'canon')
            if os.path.isdir(canon_dir):
                vid_files=utils.get_files_with_str(canon_dir, 'MOV')
                for vid in vid_files: 
                    out_dir=os.path.join(out_path,os.path.basename(s),os.path.basename(session), os.path.basename(vid.split('.')[0]))
                    if os.path.exists(out_dir):
                        shutil.rmtree(out_dir)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    file_name = os.path.basename(vid)
                    cmd=f'ffmpeg -i {vid} -vf fps=30 {out_dir}\%%04d.jpg'
                    print(cmd)

def rearrange_main_dir():
    ext_dir='D:\\CPR_extracted'
    sub_dirs=utils.get_dirs_with_str(ext_dir, 'P')
    for sd in sub_dirs:
        session_dirs=utils.get_dirs_with_str(sd, 's')
        base=os.path.basename(sd)
        if base=='P5':
            print('p5')
        for sess_dir in session_dirs:
            kinect_dir=os.path.join(sess_dir, 'kinect')
            color_dir=os.path.join(sess_dir, 'color')
            depth_dir=os.path.join(sess_dir, 'depth')
            try:
                utils.move_into_dir(color_dir, kinect_dir)
                utils.move_into_dir(depth_dir, kinect_dir)
            except Exception as e:
                print(e)
                logging.error(f'{sess_dir}: {e}')
            try:
                file=os.path.join(sess_dir, 'hand_bbs.json')
                utils.move_into_dir(file, kinect_dir)
            except Exception as e:
                print(e)
                logging.error(f'{sess_dir}: {e}')
            try:
                file=os.path.join(sess_dir, 'hand_keypts_XYZ.json')
                utils.move_into_dir(file, kinect_dir)
            except Exception as e:
                print(e)
                logging.error(f'{sess_dir}: {e}')
            try:
                file=os.path.join(sess_dir, 'kinect_depth_interp.txt')
                utils.move_into_dir(file, kinect_dir)
            except Exception as e:
                print(e)
                logging.error(f'{sess_dir}: {e}')
            try:
                file=os.path.join(sess_dir, 'kinect_ts.txt')
                utils.move_into_dir(file, kinect_dir)
            except Exception as e:
                print(e)
                logging.error(f'{sess_dir}: {e}')
            try:
                file=os.path.join(sess_dir, 'mean_person_bbx.txt')
                utils.move_into_dir(file, kinect_dir)
            except Exception as e:
                print(e)
                logging.error(f'{sess_dir}: {e}')

def move_masks_into_data_dir():
    mask_dir = r"\\samba.cs.virginia.edu\p\blurdepth\data\hand_masks"
    data_path='D:\\CPR_extracted'
    subj_dirs=utils.get_dirs_with_str(mask_dir, 'P')
    for s in subj_dirs:
        s_base=os.path.basename(s)
        session_dirs=utils.get_dirs_with_str(s,'s')
        for session in session_dirs:
            print('processing: ', session)
            sess_base=os.path.basename(session)
            mask_dir=os.path.join(session, 'hand_mask')
            target_dir=os.path.join(data_path,s_base,sess_base,'kinect')
            subdir=os.path.join(target_dir, 'hand_mask')
            color_dir=os.path.join(target_dir, 'color')
            if os.path.exists(subdir):
                file_list = os.listdir(subdir)
                color_dir_list=os.listdir(color_dir)
                if len(file_list)==len(color_dir_list):
                    print('its already threre. Continuing...')
                    continue
            try:
                utils.move_into_dir(mask_dir, target_dir)
            except Exception as e:
                print(e)
                logging.error(f'{session}: {e}')

def get_ms_from_ts(timestamp):
    # Split the timestamp into hours, minutes, seconds, and milliseconds
    hours, minutes, seconds = timestamp.split(":")
    seconds, milliseconds = seconds.split(".")

    # Convert each part into milliseconds
    hours_in_ms = int(hours) * 3600000
    minutes_in_ms = int(minutes) * 60000
    seconds_in_ms = int(seconds) * 1000
    milliseconds = int(milliseconds)
    total_milliseconds = hours_in_ms + minutes_in_ms + seconds_in_ms + milliseconds
    return total_milliseconds


def get_ts_from_images(root_path,original_fps=30,target_fps=1):
    import numpy as np
    import time
    skip_num=original_fps//target_fps
    sub_dirs = utils.list_subdirectories(root_path)
    for dir in sub_dirs:
        ts_list=[]
        img_list=[]

        out_file=os.path.join(root_path,dir,'timestamps.txt')
        # if os.path.exists(out_file):
        #     print(f'{dir} is already processed. Continuing...')
        #     continue
        print(f'processing: {dir}')
        img_files=utils.list_files(os.path.join(root_path,dir), 'jpg')
        img_files_int=[int(file.split('.')[0]) for file in img_files]
        img_files_int.sort()
        img_files = [str(file).zfill(4) + '.jpg' for file in img_files_int]
        process_ar=np.zeros(len(img_files))
        process_ar[::skip_num]=1
        selected_img_files=[img_files[i] for i in range(len(img_files)) if process_ar[i]==1]
        #get ts with OCR
        for i,img in enumerate(selected_img_files):
            # print(f'{i} out of {len(selected_img_files)} is done.',end='\r')
            img_file=os.path.join(root_path,dir,img)
            ts=get_ts_google(img_file) 
            if ts:
                img_list.append(int(img.split('.')[0]))
                # print(img,int(img.split('.')[0]))
                ts_list.append(ts)

        ms_list=[get_ms_from_ts(ts) for ts in ts_list]
        intterp_ts_list=np.zeros(len(img_files_int))
        for i in range(len(img_list)):
            intterp_ts_list[img_list[i]-1]=ms_list[i]
        #interpolate
        args=np.argwhere(intterp_ts_list!=0)
        args.sort()
        for i in range(len(intterp_ts_list)):
            if intterp_ts_list[i]==0:
                if i<=args[0]:
                    I1,I2=args[0:2][:,0]
                elif i>=args[-1]:
                    I1,I2=args[-2:][:,0]                    
                else:
                    I1=max(args[args<i])
                    I2=min(args[args>i])
                T1,T2=intterp_ts_list[I1],intterp_ts_list[I2]
                t_interp=T2 + (T2-T1)/(I2-I1)*(i-I2)
                intterp_ts_list[i]=t_interp
        #save to file
        with open(out_file, 'w') as f:
            for i in range(len(intterp_ts_list)):
                f.write(str(img_files[i])+','+str(intterp_ts_list[i])+'\n')


if __name__ == "__main__":
    # move_masks_into_data_dir()
    get_ts_from_images(r'D:\hand_depth_dataset\canon')


























                
                



