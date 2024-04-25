import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor
import torch
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import shutil
import time
import argparse

skip=120
#object detection model
base_model = GroundingDINO(ontology=CaptionOntology({"hand": "hand"}))

#hand segmentation model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),'sam_vit_h_4b8939.pth')
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_predictor = SamPredictor(sam)

def get_interp_ts_list(files):
    process_ar=[0]*len(files)
    ind_list,ts_list=[],[]
    
    for i in range(len(files)):
        if i%skip!=0:
            continue
        # ts=None
        # j=i
        # while not ts:
        #     color_file=os.path.join(color_dir,files[j])
        #     ts=utils.get_ts_google(color_file)
        #     j+=1
        #     if j==i+skip:
        #         break

        ts=utils.get_ts_google(files[i])
        if ts:
            process_ar[i]=1
            ind_list.append(i)
            ts_list.append(ts)

    ms_list=[utils.get_ms_from_ts(ts) for ts in ts_list]
    intterp_ts_list=[0]*len(files)
    for i,idx in enumerate(ind_list):
        intterp_ts_list[idx]=ms_list[i]
    #interpolate
    ind_list.sort()
    ind_list=np.array(ind_list)
    for i in range(len(intterp_ts_list)):
        if intterp_ts_list[i]==0:
            if i<=ind_list[0]:
                I1,I2=ind_list[0:2]
            elif i>=ind_list[-1]:
                I1,I2=ind_list[-2:]                
            else:
                I1=max(ind_list[ind_list<i])
                I2=min(ind_list[ind_list>i])
            T1,T2=intterp_ts_list[I1],intterp_ts_list[I2]
            t_interp=T2 + (T2-T1)/(I2-I1)*(i-I2)
            intterp_ts_list[i]=t_interp
    return intterp_ts_list


def get_ts_kinect():
    dirs=utils.list_subdirectories(kinect_root) 
    for dir in dirs:
        print(dir)
        out_file=os.path.join(kinect_root,dir,'ts.txt')
        if os.path.exists(out_file):
            print('ts file already exists. continuing...')
            continue
        color_dir=os.path.join(kinect_root,dir,'color')
        files=utils.list_files(color_dir,'jpg')
        files=[os.path.join(kinect_root,dir,'color',f) for f in files]
        
        intterp_ts_list=get_interp_ts_list(files)
        #save to file
        with open(out_file, 'w') as f:
            for i in range(len(intterp_ts_list)):
                f.write(str(files[i])+','+str(intterp_ts_list[i])+'\n')

def get_ts_canon():
    dirs=utils.list_subdirectories(canon_root)
    for dir in dirs:
        print(dir)
        out_file=os.path.join(canon_root,dir,'ts.txt')
        if os.path.exists(out_file):
            print('ts file already exists. continuing...')
            continue
        files=utils.list_files(os.path.join(canon_root,dir),'jpg')
        files=[os.path.join(canon_root,dir,f) for f in files]
        intterp_ts_list=get_interp_ts_list(files)
        #save to file
        with open(out_file, 'w') as f:
            for i in range(len(intterp_ts_list)):
                f.write(str(files[i])+','+str(intterp_ts_list[i])+'\n')

def get_ms_ts(file):
    ts=utils.get_ts_google(file)
    if ts:
        ts=utils.get_ms_from_ts(ts)
    return ts

def get_seg(file,bb):
    image = cv2.imread(file)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    area=(bb[3]-bb[1])*(bb[2]-bb[0])

    bb_=[int(b) for b in bb]

    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(image_rgb)

    if area<200000:
        input_point = np.array([[(bb_[0]+bb_[2])*0.5, (bb_[1]+bb_[3])*0.5]])
        input_label = np.array([1])
        mask, scores, logits = mask_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
        )
        mask=np.squeeze(mask)
        #remove mask outside the bb
        anti_mask=np.zeros_like(mask)
        anti_mask[bb_[1]:bb_[3],bb_[0]:bb_[2]]=1
        mask=mask*anti_mask
    else:
        mask_predictor.set_image(image_rgb)
        mask, scores, logits = mask_predictor.predict(
            box=np.array(bb_),
            multimask_output=False
            )
        mask=np.squeeze(mask)


    # bb_=[int(b) for b in bb]
    # pad=0
    # bb_=[bb_[0]-pad,bb_[1]-pad,bb_[2]+pad,bb_[3]+pad]
    # # img_cropped=utils.crop_img_bb(image_rgb,bb,50)
    # # utils.draw_bb(file,bb)

    # mask_predictor.set_image(image_rgb)
    # masks, scores, logits = mask_predictor.predict(
    #     box=np.array(bb_),
    #     multimask_output=True
    #     )


    mask=mask*255
    mask=mask.astype(np.uint8)

    # mask=np.any(mask, axis=0)*255
    # mask=mask.astype(np.uint8)
    # plt.imshow(mask*anti_mask)
    return mask

def get_point_cloud(file,mask=None):
    depth_file=os.path.join(file.replace('jpg','png').replace('color','depth'))
    depth_img = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if mask is not None:
        depth_img=depth_img*(mask/mask.max())
    d=np.argwhere(depth_img>0)
    x,y=d[:,0],d[:,1]
    depth_valus=depth_img[x,y]
    X,Y,Z=utils.get_XYZ_kinect(x,y,depth_valus)

    # points = np.vstack((X, Y, Z)).T
    # pcd_lower = o3d.geometry.PointCloud()
    # pcd_lower.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd_lower], window_name="Aligned Point Clouds")


    return X,Y,Z

def resize_sparse_points(image, new_size):
    # Create a blank image of the new size
    resized_image = np.zeros(new_size, dtype=image.dtype)
    
    # Calculate scaling factors
    y_scale = new_size[0] / image.shape[0]
    x_scale = new_size[1] / image.shape[1]
    
    # Set points in the new image based on their scaled positions
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x]:  # If the pixel is a point
                new_y = int(y * y_scale)
                new_x = int(x * x_scale)
                resized_image[new_y, new_x] = image[y, x]  # Set the point in the new image
    
    return resized_image

def get_centroid(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assuming the largest contour corresponds to the object of interest
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return np.array([cx, cy])


def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    # Convert angle from degrees to radians
    angle_rad = np.radians(angle)

    # Calculate the cosine and sine of the rotation angle
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Translate point back to origin:
    px -= ox
    py -= oy

    # Rotate point
    new_x = px * cos_angle - py * sin_angle
    new_y = px * sin_angle + py * cos_angle

    # Translate point back:
    new_x += ox
    new_y += oy

    return new_x, new_y


import tkinter as tk
from tkinter import messagebox

# Function to be called when the first button is pressed
def on_accept_press():
    global save_data
    save_data = True
    root.destroy()

# Function to be called when the second button is pressed
def on_reject_press():
    global save_data
    save_data = False
    root.destroy()
            
def sync_imgs(data_root,out_path):
    kinect_root=os.path.join(data_root,'kinect')
    canon_root=os.path.join(data_root,'canon')
    dirs=utils.list_subdirectories(kinect_root)
    for dir in dirs:
        k_bb_path=os.path.join(out_path,'kinect','bbs.txt')
        canon_bb_path=os.path.join(out_path,'canon','bbs.txt')
        part_name=dir.split('_')[0]
        # if part_name in ['P0','P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13']:
        #     continue
        ts_path=os.path.join(kinect_root,dir,'ts.txt')
        with open(ts_path, 'r') as f:
            lines = f.readlines()
        kinect_files=[l.split(',')[0] for l in lines]
        kinect_ts=np.array([float(l.split(',')[1].strip()) for l in lines])
        
        indices = list(range(0, len(kinect_files), 30))
        selected_kinect_files=[k for i,k in enumerate(kinect_files) if i in indices]
        selected_kinect_ts_list=np.array([k for i,k in enumerate(kinect_ts) if i in indices])

        #get all canon files and ts
        part=dir.split('_')[0]
        canon_dirs=utils.get_dirs_with_str(canon_root,part,i=0,j=1)
        if len(canon_dirs)==0:
            print('no canon dirs found')
            continue
        canon_files_list,canon_ts_list=[],[]
        for d in canon_dirs:
            canon_ts_file=os.path.join(canon_root,d,'ts.txt')
            with open(canon_ts_file, 'r') as f:
                lines = f.readlines()
            canon_files=[os.path.join(d,os.path.basename(l.split(',')[0])) for l in lines]
            canon_ts=[float(l.split(',')[1].strip()) for l in lines]
            canon_files_list.extend(canon_files)
            canon_ts_list.extend(canon_ts)

        for ind in indices:
            canon_color_file=os.path.join(out_path,'canon','color',part_name+'_'+str(ind)+'.jpg')
            canon_depth_file=os.path.join(out_path,'canon','depth',part_name+'_'+str(ind)+'.png')
            canon_seg_file=os.path.join(out_path,'canon','seg',part_name+'_'+str(ind)+'.png')
            kinect_color_file=os.path.join(out_path,'kinect','color',part_name+'_'+str(ind)+'.jpg')
            kinect_depth_file=os.path.join(out_path,'kinect','depth',part_name+'_'+str(ind)+'.png')
            kinect_seg_file=os.path.join(out_path,'kinect','seg',part_name+'_'+str(ind)+'.png')
            print(os.path.basename(canon_color_file))
            if os.path.exists(canon_color_file) and os.path.exists(canon_depth_file) and os.path.exists(canon_seg_file) and os.path.exists(kinect_color_file) and os.path.exists(kinect_depth_file) and os.path.exists(kinect_seg_file):
                print('files already exist. continuing...')
                # continue

            ts=kinect_ts[ind]
            k_file=os.path.basename(kinect_files[ind])
            print('k_file:',k_file)
            #find the closest canon image
            closest_idx=np.argmin(np.abs(canon_ts_list-ts))
            canon_ts=canon_ts_list[closest_idx]
            closest_file=canon_files_list[closest_idx]
            #get the kinect images around this canon image
            v=kinect_ts-canon_ts
            if len(v[v<=0])==0:
                kinect_closest_lower_idx=ind
            else:
                kinect_closest_lower_idx=max(np.argwhere(v<=0))[0]
            if len(v[v>=0])==0:
                kinect_closest_upper_idx=ind
            else:
                kinect_closest_upper_idx=min(np.argwhere(v>=0))[0]

            k_lower_file=os.path.join(kinect_root,dir,'color',kinect_files[kinect_closest_lower_idx])
            k_upper_file=os.path.join(kinect_root,dir,'color',kinect_files[kinect_closest_upper_idx])
            k_file=os.path.join(kinect_root,dir,'color',k_file)
            canon_file=closest_file

            #get actual ts from images
            ts=get_ms_ts(k_file)
            k_ts = ts if ts else kinect_ts[ind]

            if k_lower_file==k_file:
                k_lower_ts=k_ts
            else:
                ts=get_ms_ts(k_lower_file)
                k_lower_ts = ts if ts else kinect_ts[kinect_closest_lower_idx]

            if k_upper_file==k_file:
                k_upper_ts=k_ts
            else:
                ts=get_ms_ts(k_upper_file)
                k_upper_ts = ts if ts else kinect_ts[kinect_closest_upper_idx]

            ts=get_ms_ts(canon_file)
            c_ts = ts if ts else canon_ts

            num=0
            while k_upper_ts<=c_ts:
                kinect_closest_upper_idx+=1
                k_upper_file=os.path.join(kinect_root,dir,'color',kinect_files[kinect_closest_upper_idx])
                ts=get_ms_ts(k_upper_file)
                k_upper_ts = ts if ts else kinect_ts[kinect_closest_upper_idx]
                num+=1
                if num>10:
                    break
            num=0
            while k_lower_ts>c_ts:
                kinect_closest_lower_idx-=1
                k_lower_file=os.path.join(kinect_root,dir,'color',kinect_files[kinect_closest_lower_idx])
                ts=get_ms_ts(k_lower_file)
                k_lower_ts = ts if ts else kinect_ts[kinect_closest_lower_idx]
                num+=1
                if num>10:
                    break

            if not(k_lower_ts<=c_ts<=k_upper_ts):
                print('timestamps not in order')
                continue
            
            print(k_lower_ts,k_ts,k_upper_ts,c_ts)
            print('lower , upper:',kinect_closest_lower_idx,kinect_closest_upper_idx)
            
            #detect bbs
            results = base_model.predict(k_file)
            k_bb=utils.get_bb(results)
            results = base_model.predict(k_lower_file)
            lower_bb=utils.get_bb(results)
            results = base_model.predict(k_upper_file)
            upper_bb=utils.get_bb(results)
            results = base_model.predict(closest_file)
            canon_bb=utils.get_bb(results)

            if len(k_bb)*len(lower_bb)*len(upper_bb)*len(canon_bb)==0:
                print('at least one bb is missing')
                continue
                
            #detect hand segmentation
            k_mask=get_seg(k_file,k_bb)
            lower_mask=get_seg(k_lower_file,lower_bb)
            upper_mask=get_seg(k_upper_file,upper_bb)
            canon_mask=get_seg(closest_file,canon_bb)

            # utils.draw_bb(closest_file,canon_bb)

            # plt.imshow(upper_mask)

            # Read the image
            X_lower,Y_lower,Z_lower=get_point_cloud(k_lower_file,lower_mask)
            X_upper,Y_upper,Z_upper=get_point_cloud(k_upper_file,upper_mask)
            #remove outlier depth values
            def remove_outliers(ar):
                mad=np.median(np.abs(ar-np.median(ar)))  
                threshold=5*mad
                filtered_entries = (np.abs(ar-np.median(ar))) < threshold
                return np.where(filtered_entries)[0]

            inlier_args=remove_outliers(Z_lower)
            X_lower,Y_lower,Z_lower=X_lower[inlier_args],Y_lower[inlier_args],Z_lower[inlier_args]
            inlier_args=remove_outliers(Z_upper)
            X_upper,Y_upper,Z_upper=X_upper[inlier_args],Y_upper[inlier_args],Z_upper[inlier_args]

            points = np.vstack((X_lower, Y_lower, Z_lower)).T
            pcd_lower = o3d.geometry.PointCloud()
            pcd_lower.points = o3d.utility.Vector3dVector(points)

            points = np.vstack((X_upper, Y_upper, Z_upper)).T
            pcd_upper = o3d.geometry.PointCloud()
            pcd_upper.points = o3d.utility.Vector3dVector(points)

            upper_center=np.asarray(pcd_upper.points).mean(axis=0)
            lower_center=np.asarray(pcd_lower.points).mean(axis=0)
            tr=upper_center-lower_center
            # np.asarray(pcd_lower.points) + (upper_center-lower_center)
            # p = o3d.geometry.PointCloud()
            # p.points=o3d.utility.Vector3dVector(np.asarray(pcd_lower.points) + (upper_center-lower_center))

            # o3d.visualization.draw_geometries([pcd_lower,pcd_upper], window_name="Original Point Clouds")

            init_transformation = np.eye(4)
            init_transformation[:3,-1]=tr
            icp_result = o3d.pipelines.registration.registration_icp(
            pcd_lower, pcd_upper, max_correspondence_distance=0.002,  # Set according to your data scale
            init=init_transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
            )
            transformation_matrix = icp_result.transformation
            print("Transformation Matrix:")
            print(transformation_matrix)

            frac=(c_ts-k_lower_ts)/(k_upper_ts-k_lower_ts)
            assert frac>=0, 'fraction should be positive'

            Rot=transformation_matrix[:3,:3]
            T=transformation_matrix[:3,3]

            T_new=T*frac

            r = R.from_matrix(Rot.copy())
            euler_angles = r.as_euler('xyz', degrees=True)
            new_angles=euler_angles*frac
            new_rot=R.from_euler('xyz', new_angles, degrees=True).as_matrix()

            T_int = np.eye(4)
            T_int[:3,:3]=new_rot
            T_int[:3,3]=T_new

            pcd_int = o3d.geometry.PointCloud()
            pcd_int.points = o3d.utility.Vector3dVector(pcd_lower.points)
            pcd_int.transform(T_int)

            # o3d.visualization.draw_geometries([pcd_int], window_name="Aligned Point Clouds")

            #transform into canon coordinate frame
            pcd_int.transform(utils.kinect_to_canon)
            #project the point cloud into the canon image
            X,Y,Z=np.array(pcd_int.points).T
            x,y=utils.project_3d_to_2d(X,Y,Z,utils.canon_k)
            canon_proj=np.zeros(utils.canon_original_res)
            for i in range(len(x)):
                if int(x[i])<0 or int(y[i])<0 or int(y[i])>lower_mask.shape[1]-1 or int(x[i])>lower_mask.shape[0]-1:
                    continue
                canon_proj[int(x[i]),int(y[i])]=Z[i]
            if len(canon_proj[canon_proj>0])==0:
                print('no points projected')
                continue
            canon_proj_thres=canon_proj.copy()
            canon_proj_thres[canon_proj_thres>0]=1
            # canon_proj_thres=canon_proj_thres.astype(np.uint8)
            # density_map_blurred = cv2.GaussianBlur(canon_proj_thres, (0, 0), sigmaX=2, sigmaY=2)
            # density_map_blurred[density_map_blurred>0]=1
            canon_proj_thres=canon_proj_thres.astype(np.uint8)
            h,w=canon_mask.shape
            h_,w_=canon_proj_thres.shape
            # Pad the density map to match the size of the canon mask
            pad_h = (h - h_)
            pad_w = (w - w_)
            canon_proj_thres = np.pad(canon_proj_thres, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            canon_proj = np.pad(canon_proj, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

            aligned_img=canon_proj_thres
            aligned_depth=canon_proj

            '''
            align the depth map to the canon mask
            *************************************************************************************************************************
            '''
            target_points=np.argwhere(canon_mask>0)
            source_points=np.argwhere(aligned_depth>0)
            depth_vals=[aligned_depth[pt[0],pt[1]] for pt in source_points]
            #estimate initial params
            source_points_norm=(source_points-source_points.mean(axis=0))/source_points.std(axis=0)
            source_points_norm_scaled=source_points_norm*target_points.std(axis=0)+target_points.mean(axis=0)

            source_points_norm_scaled.mean(axis=0),source_points_norm_scaled.std(axis=0)
            target_points.mean(axis=0),target_points.std(axis=0)

            target_centroid=np.mean(target_points,axis=0)
            source_centroid=np.mean(source_points,axis=0)
            target_std=np.std(target_points,axis=0)
            source_std=np.std(source_points,axis=0)
            #normalize source points
            source_points[:,0]=(source_points[:,0]-source_centroid[0])/source_std[0]
            source_points[:,1]=(source_points[:,1]-source_centroid[1])/source_std[1]

            source_points[:,0]=source_points[:,0]/source_std[0]*target_std[0]+(target_centroid[0]-source_centroid[0])
            source_points[:,1]=source_points[:,1]/source_std[1]*target_std[1]+(target_centroid[1]-source_centroid[1])
            '''
            *************************************************************************************************************************
            '''
            canon_depth=np.zeros_like(canon_mask,dtype=np.uint16)
            h,w=canon_mask.shape
            for i,pt in enumerate(source_points_norm_scaled):
                x, y = pt
                if x>=0 and x<h and y>=0 and y<w:
                    canon_depth[int(x),int(y)]=depth_vals[i]
                        
            canon_mask[canon_mask>0]=255
            k_mask[k_mask>0]=255
            canon_img=utils.draw_bb(closest_file,canon_bb,show=False)
            mask_depth_canon=utils.show_img_overlay(canon_mask,canon_depth,alpha=0.7,show=False)

            k_img=utils.draw_bb(k_file,k_bb,show=False)
            k_depth=cv2.imread(k_file.replace('color','depth').replace('jpg','png'), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

            # fig, axs = plt.subplots(2, 3, figsize=(12, 8))
            # axs[0, 0].imshow(canon_img)
            # axs[0, 0].set_title('Canon Image')
            # axs[0, 1].imshow(mask_depth_canon)
            # axs[0, 1].set_title('Canon Mask and Depth')
            # axs[0, 2].imshow(k_img)
            # axs[0, 2].set_title('Kinect Image')
            # axs[1, 0].imshow(k_depth)
            # axs[1, 0].set_title('Kinect Depth')
            # axs[1, 1].imshow(k_mask)
            # axs[1, 1].set_title('Kinect Mask')
            # axs[1, 2].axis('off')  # Empty subplot
            # plt.tight_layout()
            # plt.show()

            # # Create the main window
            # global root
            # root = tk.Tk()
            # root.title("OpenCV Actions")
            # # Create and place the buttons
            # button1 = tk.Button(root, text="Accept", command=on_accept_press)
            # button1.pack(padx=20, pady=10)
            # button2 = tk.Button(root, text="Reject", command=on_reject_press)
            # button2.pack(padx=20, pady=10)
            # # Start the GUI event loop
            # root.mainloop()

            #save data
            kinect_dirs=[os.path.join(out_path,'kinect',dir) for dir in ['depth','color','seg']]
            for d in kinect_dirs:
                if not os.path.exists(d):
                    os.makedirs(d)
            canon_dirs=[os.path.join(out_path,'canon',dir) for dir in ['depth','color','seg']]
            for d in canon_dirs:
                if not os.path.exists(d):
                    os.makedirs(d)

            shutil.copy(closest_file, canon_color_file)
            cv2.imwrite(canon_depth_file, (canon_depth).astype(np.uint16))
            cv2.imwrite(canon_seg_file, canon_mask)
            shutil.copy(k_file, kinect_color_file)
            shutil.copy( k_file.replace('color','depth').replace('jpg','png'),kinect_depth_file)
            cv2.imwrite(kinect_seg_file, k_mask)

            canon_bb_str=part_name+'_'+str(ind)+',' + ','.join([str(int(bb)) for bb in canon_bb])
            kinect_bb_str=part_name+'_'+str(ind)+',' + ','.join([str(int(bb)) for bb in k_bb])
            with open(k_bb_path, 'a') as f:
                f.write(kinect_bb_str+'\n')
            with open(canon_bb_path, 'a') as f: 
                f.write(canon_bb_str+'\n')
            time.sleep(60)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Preparation')
    parser.add_argument('--data_root', type=str, default='D:/hand_depth_dataset/', help='Root directory of Canon data')
    parser.add_argument('--output_path', type=str, default='D:/hand_depth_extracted/', help='Output path for synchronized images')
    args = parser.parse_args()
    sync_imgs(args.data_root,args.output_path)



