import os
import json
import cv2
import numpy as np
import sys
sys.path.append('C:\\Users\\lahir\\code\\CPR-quality\\')
import utils
import argparse
import matplotlib.pyplot as plt

k=np.array([[615.873673811006,0,640.803032851225],[0,615.918359977960,365.547839233105],[0,0,1]])

def get_XYZ(x,y,depth,k):
    X=(x-k[0,2])*depth/k[0,0]
    Y=(y-k[1,2])*depth/k[1,1]
    Z=depth
    return X,Y,Z

def get_kypt_XYZ(x,y,depth_file,hand_mask_file):
    depth_img = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    # plt.scatter(x,y, c='r', marker='x')
    # plt.imshow(depth_img)
    # plt.show()
    lookup_window=10
    depth_sel=depth_img[int(y-lookup_window):int(y+lookup_window),int(x-lookup_window):int(x+lookup_window)]
    condition=(depth_sel>100.0) & (depth_sel<2000.0)
    depth_sel=depth_sel[condition]
    if len(depth_sel)>0:
        wrist_depth=np.median(depth_sel)
    else:
        #get the closest valid depth value
        valid_sensor_depth=depth_img>0
        if not os.path.exists(hand_mask_file):
            wrist_depth=-1
        else:
            hand_mask=cv2.imread(hand_mask_file,cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
            valid_depth_mask=(valid_sensor_depth) & (hand_mask>0)
            valid_args=np.argwhere(valid_depth_mask>0)
            closest_arg=np.argmin(np.sum(np.abs(valid_args-np.array([y,x])),axis=1))
            closest_xy=valid_args[closest_arg,:]
            wrist_depth=depth_img[closest_xy[0],closest_xy[1]]
    if not (wrist_depth>0 and wrist_depth<2000.0):
        print(f'Invalid depth value: {wrist_depth}')
        return -1
    X,Y,Z=get_XYZ(x,y,wrist_depth,k)
    return X,Y,Z


def mask_XYZ():
    root=r'D:\CPR_extracted\P1\s_2\kinect'
    XYZ_list=[]
    img_path=os.path.join(root,'color')
    hand_mask_dir=os.path.join(root,'hand_mask')
    depth_dir=os.path.join(root,'depth')
    with open(os.path.join(root,'hand_bbs.json'), 'r') as f:
        hand_bbs = json.load(f)
    img_files=utils.list_files(img_path,'jpg')
    for i,img_file in enumerate(img_files):
        if i<10:
            print('skipping...')
            continue

        img_key=img_file.split('.')[0]
        print(img_key)
        depth_path=os.path.join(depth_dir,img_key+'.png')
        hand_mask_file=os.path.join(hand_mask_dir,img_key+'.png')
        hand_mask=cv2.imread(hand_mask_file,cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        valid_mask_args=np.argwhere(hand_mask>0)
        mask_x_min,mask_x_max=np.min(valid_mask_args[:,1]),np.max(valid_mask_args[:,1])
        mask_y_min,mask_y_max=np.min(valid_mask_args[:,0]),np.max(valid_mask_args[:,0])
        kypt_x_vals=int(valid_mask_args[:,1][np.argwhere(valid_mask_args[:,0]==mask_y_min)].mean())
        kypt=(kypt_x_vals,mask_y_min)
        pt1=get_kypt_XYZ(kypt[0],kypt[1],depth_path,hand_mask_file)
        XYZ_list.append(pt1)
    

# pt1=np.array(XYZ_list[0])
# depth_list=[]
# for i in range(1,len(XYZ_list)):
#     pt_=np.array(XYZ_list[i])
#     d=np.sqrt(np.square(pt_-pt1).sum())
#     depth_list.append(d)

# plt.plot(depth_list)


def sift_XYZ():

    sift = cv2.SIFT_create(contrastThreshold=0.02)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    root=r'D:\CPR_extracted\P1\s_2\kinect'
    img_path=os.path.join(root,'color')
    hand_mask_dir=os.path.join(root,'hand_mask')
    depth_dir=os.path.join(root,'depth')
    with open(os.path.join(root,'hand_bbs.json'), 'r') as f:
        hand_bbs = json.load(f)
    img_files=utils.list_files(img_path,'jpg')
    first_desc=-1
    first_kypts=-1
    first_depth_path=-1
    first_mask_path=-1
    n=0
    max_matches=20
    first_img=-1
    last_bb=-1
    depth_list=[]
    for i,img_file in enumerate(img_files):
        print(img_file)
        if i<10:
            print('skipping...')
            continue
        if i==161:
            pass
        image = cv2.imread(os.path.join(img_path,img_file))
        img_key=img_file.split('.')[0]
        hand_mask_file=os.path.join(hand_mask_dir,img_key+'.png')
        hand_mask=cv2.imread(hand_mask_file,cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        depth_path=os.path.join(depth_dir,img_key+'.png')
        #get bb
        if img_key in hand_bbs:
            bb=hand_bbs[img_key]
            last_bb=bb
        else:
            bb=last_bb
        bb=[int(item) for item in bb.split(',')]
        # bb[-1]=int(bb[1]+(bb[-1]-bb[1])*0.5)
        cropped_image = utils.crop_img_bb(image,bb,0)
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        # plt.imshow(gray_image), plt.axis('off'), plt.show()
        
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        # img_keypoints = cv2.drawKeypoints(gray_image, keypoints, None)
        # plt.imshow(img_keypoints), plt.axis('off'), plt.show()

        # valid_mask_args=np.argwhere(hand_mask>0)
        # mask_x_min,mask_x_max=np.min(valid_mask_args[:,1]),np.max(valid_mask_args[:,1])
        # mask_y_min,mask_y_max=np.min(valid_mask_args[:,0]),np.max(valid_mask_args[:,0])
        # mask_y_new_max=mask_y_min+(mask_y_max-mask_y_min)*0.5
        # new_hand_mask=np.zeros_like(hand_mask)
        # new_hand_mask[int(mask_y_min):int(mask_y_new_max),int(mask_x_min):int(mask_x_max)]=1
        # hand_mask=new_hand_mask*hand_mask

        #select keypints that are in the mask area
        # valid_idx=[i for i,k in enumerate(keypoints) if hand_mask[int(k.pt[1]),int(k.pt[0])]>0]
        # keypoints=[keypoints[i] for i in valid_idx]
        # descriptors=descriptors[valid_idx]
        if n==0:
            first_desc=descriptors
            first_kypts=keypoints
            first_depth_path=depth_path
            first_mask_path=hand_mask_file
            first_img=gray_image
            n+=1
            continue
        matches = bf.match(first_desc,descriptors)
        dists=np.array([match.distance for match in matches])
        sorted_args=np.argsort(dists)
        selected_args=sorted_args[:min(len(dists),max_matches)]
        queryIdx=[matches[arg].queryIdx for arg in selected_args]
        trainIdx=[matches[arg].trainIdx for arg in selected_args]
        keypoints1=[first_kypts[i] for i in queryIdx]
        keypoints2=[keypoints[i] for i in trainIdx]

        # img3 = cv2.drawMatches(first_img, first_kypts, gray_image, keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3), plt.show()

        # img_keypoints = cv2.drawKeypoints(gray_image, keypoints2, None)
        # plt.imshow(img_keypoints), plt.axis('off'), plt.show()
        # img_keypoints = cv2.drawKeypoints(first_img, keypoints1, None)
        # plt.imshow(img_keypoints), plt.axis('off'), plt.show()

        #get the top most keypoint
        arg=np.argmin([k.pt[1] for k in keypoints1])
        
        top_keypoint1=keypoints1[arg].pt
        top_keypoint2=keypoints2[arg].pt




        # plt.scatter(int(top_keypoint2[0]),int(top_keypoint2[1]), c='r', marker='x')
        # plt.imshow(image)
        # plt.show()

        pt1=get_kypt_XYZ(top_keypoint1[0],top_keypoint1[1],depth_path,hand_mask_file)
        pt2=get_kypt_XYZ(top_keypoint2[0],top_keypoint2[1],first_depth_path,first_mask_path)
        depth=np.sqrt(np.square(np.array(pt1)-np.array(pt2)).sum())
        print(depth)
        depth_list.append(depth)

        # Plotting two images side by side
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(first_img)
        # axs[0].scatter(top_keypoint1[0], top_keypoint1[1], c='r', marker='x')
        # axs[0].set_title('First Image')

        # axs[1].imshow(image)
        # axs[1].scatter(top_keypoint2[0], top_keypoint2[1], c='r', marker='x')
        # axs[1].set_title('Current Image')

        # plt.show()

    plt.plot(depth_list)
    plt.show()

    pass


def extract_3Dpts(model_name,method):
    root_dir='D:\\CPR_extracted'
    subj_dirs=utils.get_dirs_with_str(root_dir, 'P')
    for subj_dir in subj_dirs:
        session_dirs=utils.get_dirs_with_str(subj_dir,'s')
        for session_dir in session_dirs:
            if not session_dir==r'D:\CPR_extracted\P0\s_2':
                continue

            img_dir=os.path.join(session_dir,'kinect','color')
            depth_dir=os.path.join(session_dir,'kinect','depth')
            ts_file=os.path.join(session_dir,'kinect','kinect_ts.txt')
            hand_mask_dir=os.path.join(session_dir,'kinect','hand_mask')

            if method=='mask':
                XYZ_pt_file=os.path.join(session_dir,'kinect','wrist_keypts',f'hand_keypts_XYZ_mask.json')
            elif method=='wrist_kypt':
                XYZ_pt_file=os.path.join(session_dir,'kinect','wrist_keypts',f'hand_keypts_{model_name}_XYZ.json')
            if os.path.exists(XYZ_pt_file):
                print(XYZ_pt_file+' exists. Continuing')
                continue
            print(session_dir)
            with open(ts_file, 'r') as file:
                # Read all lines into a list
                ts_lines = file.readlines()
            ts_list=np.array([float(line.strip()) for line in ts_lines])
            #read keypoint data
            if method=='wrist_kypt':
                with open(os.path.join(session_dir,'kinect','wrist_keypts',f'hand_keypts_{model_name}.json'), 'r') as json_file:
                    kypt_dict = json.load(json_file)                

            img_list=os.listdir(img_dir)
            img_list.sort()
            kypt_XYZ_dict={}
            
            for i,img in enumerate(img_list):
                print(f'Processing {i}/{len(img_list)}',end='\r')
                if img.endswith('.jpg'):
                    img_key=img.split('.')[0]
                    depth_file=os.path.join(depth_dir,img_key+'.png')
                    # depth_img = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
                    hand_mask_file=os.path.join(hand_mask_dir,img_key+'.png')
                    hand_mask=cv2.imread(hand_mask_file,cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
                    if method=='wrist_kypt':
                        kypts=kypt_dict[img_key]
                    elif method=='mask':
                        valid_mask_args=np.argwhere(hand_mask>0)
                        mask_x_min,mask_x_max=np.min(valid_mask_args[:,1]),np.max(valid_mask_args[:,1])
                        mask_y_min,mask_y_max=np.min(valid_mask_args[:,0]),np.max(valid_mask_args[:,0])
                        kypt_x_vals=int(valid_mask_args[:,1][np.argwhere(valid_mask_args[:,0]==mask_y_min)].mean())
                        kypt=(kypt_x_vals,mask_y_min)
                        kypts={'x':[kypt[0]],'y':[kypt[1]]}

                    if len(kypts)==0:
                        print(f'No keypoints detected for {img_key}')
                        kypt_XYZ_dict[img_key]=''
                    else:
                        kypt={}
                        for i in range(len(kypts['x'])):
                            x,y=kypts['x'][i],kypts['y'][i]
                            pts=get_kypt_XYZ(x,y,depth_file,hand_mask_file)
                            if pts==-1:
                                kypt[i]=''
                            else:
                                X,Y,Z=pts
                                kypt[i]=(str(X),str(Y),str(Z))
                        kypt_XYZ_dict[img_key]=kypt
            with open(XYZ_pt_file, 'w') as f:
                json.dump(kypt_XYZ_dict, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mmpose_finetuned_RHD2D", help="name of the model. mediapipe, mmpose_RHD2D, mmpose_onehand10k, mmpose_coco")
    parser.add_argument("--method", type=str, default="mask", help="mask_centroid: use the centroid of the mask as the keypoint,\
                        wrist_kypt: use the wrist keypoint as the keypoint")
    args = parser.parse_args()
    # mask_XYZ() 
    extract_3Dpts(args.model,args.method)
        
# t=np.arange(0,len(XYZ_list))

# xvals=np.array([XYZ[0] for XYZ in XYZ_list])
# xvals_s=xvals[xvals>-1]


# XYZ_list_cleaned=[]
# t_selected_list=[]
# for i,XYZ in enumerate(XYZ_list):
#     if XYZ[0]>0:
#         XYZ_list_cleaned.append(XYZ)
#         t_selected_list.append(ts_list[i])
# XYZ_list_cleaned=np.array(XYZ_list_cleaned)
# t_selected_list=np.array(t_selected_list)

# #interpolate points
# first_t=ts_list[0]
# input_ts=t_selected_list-first_t
# output_ts=ts_list-first_t
# last_valid_idx=np.max(np.argwhere(output_ts<input_ts[-1]))
# output_ts=output_ts[:last_valid_idx]

# out_interp=utils.interpolate_between_ts(XYZ_list_cleaned[:,1],input_ts,output_ts,fit_window=50,deg=4)

# input_ts[0],input_ts[-1]
# output_ts[0],output_ts[-1]

# plt.plot(input_ts,XYZ_list_cleaned[:,1])
# plt.plot(output_ts,out_interp)


# plt.scatter(t_selected_list-first_t,XYZ_list_cleaned[:,2],s=5)
# plt.show()

# utils.animate_pt_seq(XYZ_list_cleaned, interval=0.01)


# t_list=np.array([t_list]).T

# plt.scatter(XYZ_list_cleaned[:,0],XYZ_list_cleaned[:,1],XYZ_list_cleaned[:,2])
# plt.show()

# ransac = RANSACRegressor()
# ransac.fit(XYZ_list_cleaned[:,0:2], XYZ_list_cleaned[:,2])
# pred=ransac.predict(XYZ_list_cleaned[:,0:2])

# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)


# plt.plot(XYZ_list_cleaned[:,2])
# plt.plot(pred)
# plt.show()

# from matplotlib.animation import FuncAnimation

# def update_graph(num):
#     data=XYZ_list_cleaned[num,:]
#     graph._offsets3d = (data[0], data[1], data[2])
#     title.set_text('3D Test, time={}'.format(num))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# title = ax.set_title('3D Test')

# data=XYZ_list_cleaned[0,:]
# graph = ax.scatter(data[0], data[1], data[2])
# ani = FuncAnimation(fig, update_graph, 19, 
#                                interval=40, blit=False)
# plt.show()



# from sklearn import datasets, linear_model
# X, y, coef = datasets.make_regression(
#     n_samples=100,
#     n_features=1,
#     n_informative=1,
#     noise=10,
#     coef=True,
#     random_state=0,
# )





# XYZ_list
# x_vals=np.array([XYZ[0] for XYZ in XYZ_list])
# y_vals=np.array([XYZ[1] for XYZ in XYZ_list])
# z_vals=np.array([XYZ[2] for XYZ in XYZ_list])

# x_vals=x_vals[x_vals>0]
# y_vals=y_vals[y_vals>0] 
# z_vals=z_vals[z_vals>0] 


# plt.plot(y_vals[y_vals>0])
# plt.show()

# first_point=XYZ_list[0]
# for i in range(1,len(XYZ_list)):
#     if XYZ_list[i]==[-1]:
#         XYZ_list[i]=first_point
#     else:
#         first_point=XYZ_list[i]

# ransac = RANSACRegressor()
# ransac.fit(np.array(XYZ_list), np.array(XYZ_list))


# depth_list=np.array(depth_list)
# depth_list=depth_list[depth_list>0]
# plt.plot(depth_list)
# plt.show()

# depth_files=[os.path.join(depth_dir,item) for item in os.listdir(img_dir) if item.endswith('.jpg')]
