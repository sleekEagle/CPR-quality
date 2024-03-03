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

kinect_ts_format = "%H_%M_%S.%f"
root_dir='D:\\CPR_data_raw\\'

def get_subject_smartwatch_data(part_dir):
    session_dirs=utils.get_dirs_with_str(part_dir, 's',i=0,j=1)
    smartwatch_data=[]
    for sess_dir in session_dirs:
        smartwatch_dir=os.path.join(sess_dir, 'smartwatch')
        smartwatch_files=utils.get_files_with_str(smartwatch_dir, 'txt')
        for file in smartwatch_files:
            with open(file) as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                ts_s = [l.split(',')[-1] for l in lines]
                ts=utils.get_ts_list(ts_s)
                smartwatch_data.append([file,ts, lines])
    return smartwatch_data

# participants=utils.get_dirs_with_str(root_dir, 'P')
# for part_dir in participants:
#     smartwatch_data=get_subject_smartwatch_data(part_dir)

ext_dir='D:\\CPR_extracted\\'

def extract_data():
    participants=utils.get_dirs_with_str(ext_dir, 'P')
    for part_dir in participants:
        #get all smartwatch data for this participant
        root_part_dir=os.path.join(root_dir,os.path.basename(part_dir))
        smartwatch_data=get_subject_smartwatch_data(root_part_dir)
        session_dirs=utils.get_dirs_with_str(part_dir, 's',i=0)
        for sess_dir in session_dirs:
            # if sess_dir=='D:\\CPR_extracted\\P12\\s_1':
            #     print('here')
            print('Processing session: ', sess_dir)
            ts_file=os.path.join(sess_dir, 'kinect_ts.txt')
            with open(ts_file) as f:
                ts_list = f.readlines()
                ts_list = [float(line.strip()) for line in ts_list]
            #find the matching smartwatch data file
            start_ts,end_ts=ts_list[0],ts_list[-1]
            found_data=0
            for data in smartwatch_data:
                if data[1][0]<=start_ts and data[1][-1]>=end_ts:
                    #extract the relavent section of the smartwatch data
                    indices = [i for i, ts in enumerate(data[1]) if ((ts >= start_ts) and (ts <= end_ts))]
                    indices.sort()
                    found_data = [data[-1][i] for i in indices]
                    break
            if found_data:
                smartwatch_out_dir=os.path.join(sess_dir, 'smartwatch')
                os.makedirs(smartwatch_out_dir,exist_ok=True)
                out_file=os.path.join(smartwatch_out_dir, 'smartwatch.txt')
                if os.path.exists(out_file):
                    os.remove(out_file)
                print('len found data: ', len(found_data))
                n=0
                with open(out_file, 'w') as f:
                    for item in found_data:
                        if len(item)>0 and len(item)<110:
                            f.write("%s\n" % item)
                            n+=1
                print('len written data: ', n)
            else:
                print('No data found for session: ', sess_dir)

extract_data()  



        
        
        
        








