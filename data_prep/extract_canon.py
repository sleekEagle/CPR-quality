import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import shutil
from datetime import datetime
import re
import logging
# Set up logging configuration
logging.basicConfig(filename='output.log', level=logging.INFO)
# Add the following line at $PLACEHOLDER$
logging.info('This is hand segmentation code')

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
                dest_path = os.path.join(dest_path, os.path.basename(s),os.path.basename(session), 'hand_mask')
                if not os.path.exists(dest_path):
                    shutil.copytree(mask_dir, dest_path)

def get_ffmpeg_cmds():
    path='D:\\CPR_data_raw\\'
    sub_dirs=utils.get_dirs_with_str(path, 'P')
    for s in sub_dirs:
        session_dirs=utils.get_dirs_with_str(os.path.join(path,s), 's')
        for session in session_dirs:
            canon_dir=os.path.join(session, 'canon')
            if os.path.isdir(canon_dir):
                vid_files=utils.get_files_with_str(canon_dir, 'MOV')
                for vid in vid_files:                    
                    out_dir=os.path.join(canon_dir, vid.split('.')[0])
                    if os.path.exists(out_dir):
                        shutil.rmtree(out_dir)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    file_name = os.path.basename(vid)
                    cmd=f'ffmpeg -i {vid} -vf fps=30 {out_dir}\%%04d.png'
                    print(cmd)

if __name__ == "__main__":
    copy_mask_files()












                
                



