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
import numpy as np
import json

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



def get_ts_from_images(root_path,original_fps=30,target_fps=1):
    import numpy as np
    import time
    skip_num=original_fps//target_fps
    sub_dirs = utils.list_subdirectories(root_path)
    for dir in sub_dirs:
        ts_list=[]
        img_list=[]

        out_file=os.path.join(root_path,dir,'timestamps.txt')
        if os.path.exists(out_file):
            print(f'{dir} is already processed. Continuing...')
            continue
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
            ts=-1
            try:
                while(ts==-1):
                    ts=utils.get_ts_google(img_file) 
                    if ts:
                        img_list.append(int(img.split('.')[0]))
                        # print(img,int(img.split('.')[0]))
                        ts_list.append(ts)
                    if ts==-1:
                        time.sleep(1)
            except Exception as e:
                print(e)

        ms_list=[utils.get_ms_from_ts(ts) for ts in ts_list]
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

def get_ts_canon_dataset_main(path):
    pdirs=utils.list_subdirectories(path)
    for p in pdirs:
        sdirs=utils.get_dirs_with_str(os.path.join(path,p),'s',i=0,j=1)
        for s in sdirs:
            out_file=os.path.join(s,'timestamps.txt')
            if os.path.exists(out_file):
                print(f'{s} is already processed. Continuing...')
                continue
            print(f'processing: {s}')
            img_dir=[s for s in utils.list_subdirectories(s) if (s.startswith('img') or s.startswith('MVI'))][0]
            imgs=[os.path.basename(img) for img in utils.get_files_with_str(os.path.join(s,img_dir),'jpg')]
            imgs.sort()
            ts1=utils.get_ts_google(os.path.join(s,img_dir,imgs[0]),wait=5)
            ts2=utils.get_ts_google(os.path.join(s,img_dir,imgs[-1]),wait=5)
            assert ts1 and ts2, 'Error in getting timestamps'
            ts1=utils.get_ms_from_ts(ts1)
            ts2=utils.get_ms_from_ts(ts2)
            assert (ts1 and ts2) , 'Error in getting timestamps'
        
            # ts_range = np.linspace(ts1, ts2, len(imgs))
            # ts1=ts2 - 1000/30.4*len(imgs)
            ts_range = np.linspace(ts1, ts2, len(imgs))

            out_file=os.path.join(s,'timestamps.txt')
            with open(out_file, 'w') as f:
                for i in range(len(imgs)):
                    f.write(imgs[i]+','+str(ts_range[i])+'\n')

def select_inrange_canon_data():
    kinect_path=r'D:\CPR_extracted'
    canon_path=r'D:\CPR_dataset\canon_images'
    canon_out_path=r'D:\CPR_dataset\canon_images_selected2'

    kinect_part_dirs=utils.get_dirs_with_str(kinect_path, 'P')
    for kp in kinect_part_dirs:
        kinect_s_dirs=utils.get_dirs_with_str(kp, 's')
        for sd in kinect_s_dirs:
            print(f'processing {sd}...')
            if sd!='D:\\CPR_extracted\\P23\\s_10':
                continue
            ts_path=os.path.join(sd, 'kinect','kinect_ts.txt')
            if not os.path.exists(ts_path):
                print(f'{ts_path} does not exist. Continuing...')
                continue
            ts_list=np.array(utils.read_allnum_lines(ts_path))

            out_dir=os.path.join(canon_out_path,os.path.basename(kp),os.path.basename(sd))
            if os.path.exists(out_dir):
                print(f'{sd} is already processed. Continuing...')
                # continue

            cdir_=os.path.join(canon_path, os.path.basename(kp))
            if not os.path.exists(cdir_):
                print(f'{cdir_} does not exist. Continuing...')
                continue
            canon_s_dir=utils.get_dirs_with_str(cdir_,'s',i=0,j=1)
            c_dir_ts=[]
            ts_vals=[]

            for s in canon_s_dir:
                ts_dir=os.path.join(s,'timestamps.txt')
                with open(ts_dir, 'r') as f:
                    canon_ts_list = f.readlines()
                canon_ts_list=np.array([float(val.strip().split(',')[-1])/1000 for val in canon_ts_list])
                c1=canon_ts_list[canon_ts_list>ts_list[0]]
                c2=c1[c1<ts_list[-1]]
                c_dir_ts.append((ts_dir,len(c2)))
                ts_vals.append(canon_ts_list)

            best_arg=np.argmax(np.array([t[1] for t in c_dir_ts]))
            if c_dir_ts[best_arg][1]<30*5:
                print('Not sufficient data in the canon dir. continuing...')
                continue
            best_dir=c_dir_ts[best_arg][0]
            best_ts_vals=ts_vals[best_arg]
            condition1=best_ts_vals>=ts_list[0]
            condition2=best_ts_vals<=ts_list[-1]
            and_cond=condition1 & condition2
            select_cond=np.argwhere(and_cond)[:,0]
            selected_ts=best_ts_vals[select_cond]
            img_dir=[s for s in utils.list_subdirectories(os.path.dirname(best_dir)) if (s.startswith('img') or s.startswith('MVI'))][0]
            imgs = utils.list_files(os.path.join(os.path.dirname(best_dir),img_dir), 'jpg')
            imgs.sort()
            selected_imgs=[imgs[i] for i in select_cond]
            # selected_depth_anything_imgs=[os.path.join(os.path.dirname(best_dir),'depth_anything_depth',img.split('.')[0]+'.png') for img in selected_imgs]
            # selected_handmask_imgs=[os.path.join(os.path.dirname(best_dir),'hand_mask',img.split('.')[0]+'.png') for img in selected_imgs]

            print(f'canon source dir: {best_dir}')
            out_color_dir=os.path.join(out_dir,'color')
            out_depth_anything_depth_dir=os.path.join(out_dir,'depth_anything_depth')
            out_hand_mask_dir=os.path.join(out_dir,'hand_mask')
            out_kypt_dir=os.path.join(out_dir,'hand_keypts')
            if not os.path.exists(out_color_dir): os.makedirs(out_color_dir)
            if not os.path.exists(out_depth_anything_depth_dir): os.makedirs(out_depth_anything_depth_dir)
            if not os.path.exists(out_hand_mask_dir): os.makedirs(out_hand_mask_dir)
            if not os.path.exists(out_kypt_dir): os.makedirs(out_kypt_dir)
            
            for i in range(len(selected_imgs)):
                try:
                    shutil.copy(os.path.join(os.path.dirname(best_dir),img_dir,selected_imgs[i]), os.path.join(out_dir,'color',selected_imgs[i]))
                except:
                    print(f'Error in copying images {selected_imgs[i]}')
                try:
                    shutil.copy(os.path.join(os.path.dirname(best_dir),'depth_anything_depth',selected_imgs[i].split('.')[0]+'.png'), os.path.join(out_depth_anything_depth_dir,selected_imgs[i].split('.')[0]+'.png'))
                except:
                    print(f'Error in copying depth anything {selected_imgs[i]}')
                try:
                    shutil.copy(os.path.join(os.path.dirname(best_dir),'hand_mask',selected_imgs[i].split('.')[0]+'.png'), os.path.join(out_hand_mask_dir,selected_imgs[i].split('.')[0]+'.png'))
                except:
                    print(f'Error in copying hand mask {selected_imgs[i]}')
            
            img_keys=[k.split('.')[0] for k in imgs]
            with open(os.path.join(os.path.dirname(best_dir),'wrist_keypts','hand_keypts_mediapipe.json'), 'r') as file:
                data = json.load(file)
                k=[k for k in data.keys() if k in img_keys]
                data={k_:data[k_] for k_ in k}
            with open(os.path.join(out_kypt_dir,'hand_keypts_mediapipe.json'), 'w') as file:
                json.dump(data, file)

            with open(os.path.join(os.path.dirname(best_dir),'hand_bbs.json'), 'r') as file:
                data = json.load(file)
                k=[k for k in data.keys() if k in img_keys]
                data={k_:data[k_] for k_ in k}
            with open(os.path.join(out_dir,'hand_bbs.json'), 'w') as file:
                json.dump(data, file)
            #save ts
            out_file=os.path.join(canon_out_path,os.path.basename(kp),os.path.basename(sd),'timestamps.txt')
            with open(out_file, 'w') as f:
                for i in range(len(selected_imgs)):
                    f.write(selected_imgs[i]+','+str(selected_ts[i])+'\n')


def copy_canon_data():
    source_path=r'D:\CPR_dataset\canon_images'
    dest_path=r'D:\CPR_dataset\canon_images_selected'
    part_dirs=utils.get_dirs_with_str(source_path, 'P')
    for pd in part_dirs:
        s_dirs=utils.get_dirs_with_str(os.path.join(source_path,pd), 's')
        for sd in s_dirs:
            print(f'processing {sd}...')
            source_color_path=os.path.join(source_path,os.path.basename(pd),os.path.basename(sd))
            img_dir=[s for s in utils.list_subdirectories(source_color_path) if (s.startswith('img') or s.startswith('MVI'))][0]
            source_color_path=os.path.join(source_color_path,img_dir)
            imgs = utils.list_files(os.path.join(os.path.dirname(source_color_path),img_dir), 'jpg')

def check_canon():
    root=r'D:\CPR_dataset\canon_images'
    part_dirs=utils.get_dirs_with_str(root,'P')
    for p in part_dirs:
        sub_dirs=utils.get_dirs_with_str(p,'s')
        for s in sub_dirs:
            print(s)
            img_dirs=[dir for dir in utils.list_subdirectories(s) if (dir.startswith('img') or dir.startswith('MVI'))]
            print(img_dirs)


def copy_canon_ts():
    source_path=r'D:\canon_images_original'
    dest_path=r'D:\CPR_dataset\canon_images'
    part_dirs=utils.get_dirs_with_str(source_path, 'P')
    for pd in part_dirs:
        s_dirs=utils.get_dirs_with_str(os.path.join(source_path,pd), 's')
        for sd in s_dirs:
            print(f'processing {sd}...')
            dest_ts_path=os.path.join(dest_path,os.path.basename(pd),os.path.basename(sd),'timestamps.txt')
            if os.path.exists(dest_ts_path):
                print(f'{sd} is already processed. Continuing...')
                continue
            source_ts_path=os.path.join(source_path,os.path.basename(pd),os.path.basename(sd),'timestamps.txt')
            shutil.copy(source_ts_path,dest_ts_path)



if __name__ == "__main__":
    # move_masks_into_data_dir()
    # get_ts_from_images(r'D:\hand_depth_dataset\canon')
    # get_ts_canon_dataset_main(r'D:\CPR_dataset\canon_images')
    select_inrange_canon_data()
    # check_canon()


























                
                



