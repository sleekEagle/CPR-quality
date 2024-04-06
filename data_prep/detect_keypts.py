import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import json
import numpy as np
import argparse
import cv2

def show_img(image):
    # Check if the image was successfully loaded
    if image is not None:
        # Display the image
        cv2.imshow('Image', image)

        # Wait for a key press and then close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to load the image.")

def plot_points(image,points):
        # Draw circles at the specified points
        for point in points:
            cv2.circle(image, point, 5, (0, 255, 0), -1)
        return image

class WristDet_mediapipe:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp_hands = mp.solutions.hands
    
    def get_kypts(self,image):
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
                # image = cv2.imread(path)
                height, width, _ = image.shape
                # Detect hands in the image
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                xy_vals=[]
                z_vals=[]
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(21):
                            x = int(hand_landmarks.landmark[i].x * width)
                            y = int(hand_landmarks.landmark[i].y * height)
                            z=hand_landmarks.landmark[i].z
                            xy_vals.append((x,y))
                            z_vals.append(z)
                if len(xy_vals)==42:
                    closest_hand=np.argmin([z_vals[0],z_vals[21]])
                    start_coord=0 if closest_hand==0 else 21
                    xy_vals=xy_vals[start_coord:start_coord+21]
        return image,xy_vals
    
def detect_object(path,obejct_name='person'):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=r'C:\Users\lahir\Downloads\efficientdet_lite0.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                        score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)
    image = mp.Image.create_from_file(path)
    detection_result = detector.detect(image)
    # image_copy = np.copy(image.numpy_view())
    # annotated_image = visualize(image_copy, detection_result)
    # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    # show_img(rgb_annotated_image)'
    detection_result=detection_result.detections
    bbx=-1
    for det in detection_result:
        if det.categories[0].category_name==obejct_name:
             bbx=det.bounding_box
             break
    return bbx
    
# wrst=WristDet_mediapipe()
# session_dir='D:\\CPR_data_raw\\P0\\extracted\\s_10\\'
# img_dir=os.path.join(session_dir,'color')
# img_files=utils.list_files(img_dir,'jpg')
# destination_directory=os.path.join(session_dir,'wrist_keypts')
# if not os.path.exists(destination_directory):
#     os.makedirs(destination_directory)
# destination_file=os.path.join(destination_directory,'hand_keypts_mediapipe.json')
# with open(destination_file,'w') as file:
#     output={}
#     for img_file in img_files:
#         img_path=os.path.join(img_dir,img_file)
#         image,xy_vals=wrst.get_kypts(img_path)
#         if len(xy_vals)==0:
#             output[img_file.split('.')[0]]={}
#             continue
#         assert len(xy_vals)==21, f'Number of key points is not 21 for {img_file}'
#         x_vals=[val[0] for val in xy_vals]
#         y_vals=[val[1] for val in xy_vals]
#         sub_dict={"x":x_vals,"y":y_vals}
#         output[img_file.split('.')[0]]=sub_dict
#     json.dump(output, file)

# wrst=WristDet_mediapipe()
# path=r'D:\CPR_data_raw\P13\extracted\s_0\color\01319.jpg'
# image,xy_vals=wrst.get_kypts(path)

def detect_kypts_mp(root_dir):
    wrst=WristDet_mediapipe()
    subj_dirs=utils.get_dirs_with_str(root_dir, 'P')
    for subj_dir in subj_dirs:
        session_dirs=utils.get_dirs_with_str(subj_dir,'s')
        for session_dir in session_dirs:
            print(session_dir)
            handbb_path=os.path.join(session_dir,'kinect','hand_bbs.json')
            if os.path.exists(handbb_path):
                with open(handbb_path, 'r') as file:
                    hand_bbs = json.load(file)
            else:
                print(f'{handbb_path} does not exist')
                continue

            img_dir=os.path.join(session_dir,'kinect','color')
            img_files=utils.list_files(img_dir,'jpg')
            destination_directory=os.path.join(session_dir,'kinect','wrist_keypts')
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
            destination_file=os.path.join(destination_directory,'hand_keypts_mediapipe.json')
            if os.path.exists(destination_file):
                with open(destination_file, 'r') as file:
                    lines = file.readlines()
                    if len(lines)>0:
                        print(f'{destination_file} is not empty. Continuing...')
                        continue
                        # Delete the destination file if it exists
                os.remove(destination_file)
                   
            with open(destination_file,'w') as file:
                output={}
                last_bb=0
                for img_file in img_files:
                    img_path=os.path.join(img_dir,img_file)
                    img=cv2.imread(img_path)
                    if hand_bbs[img_file.split('.')[0]]=='':
                        hand_bb=last_bb
                    else:
                        hand_bb=[int(val) for val in hand_bbs[img_file.split('.')[0]].split(',')]
                    last_bb=hand_bb
                    pad=80
                    img_crop= utils.crop_img_bb(img,hand_bb,pad)

                    image,xy_vals=wrst.get_kypts(img_crop)
                    # if len(xy_vals)==0:
                    #     output[img_file.split('.')[0]]={}
                    #     continue
                    # assert len(xy_vals)==21, f'Number of key points is not 21 for {img_file}'
                    
                    x_vals=[val[0]+hand_bb[0]-pad for val in xy_vals]
                    y_vals=[val[1]+hand_bb[1]-pad for val in xy_vals]
                    # points=[]
                    # for i in range(len(x_vals)):
                    #     if x_vals[i]==0:
                    #         print('zero detected')
                    #     points.append((x_vals[i],y_vals[i]))
                    # utils.plot_points(img,points)
                    # utils.show_img(img)

                    sub_dict={"x":x_vals,"y":y_vals}
                    output[img_file.split('.')[0]]=sub_dict
                json.dump(output, file)

'''
different models and their performance 
https://github.com/open-mmlab/mmpose/tree/5a3be9451bdfdad2053a90dc1199e3ff1ea1a409/configs/hand_2d_keypoint
selected topdown heatmap method from 
https://github.com/open-mmlab/mmpose/tree/5a3be9451bdfdad2053a90dc1199e3ff1ea1a409/configs/hand_2d_keypoint/topdown_heatmap

selected model: HRNETv2-w18+dark
trained on: 
    RHD2D : td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256
    coco : td-hm_hrnetv2-w18_dark-8xb32-210e_coco-wholebody-hand-256x256
    onehand10k : td-hm_hrnetv2-w18_dark-8xb64-210e_onehand10k-256x256
'''

def detec_keypoints_dir_mmpose(data_path,model_name):
    from mmdet.apis import init_detector, inference_detector,DetInferencer
    from mmpose.apis import MMPoseInferencer
    import numpy as np
    from mmcv.image import imread
    from PIL import Image
    import cv2
    # from mmpose.apis import init_pose_model, inference_top_down_pose_model

    config_file = r'C:\Users\lahir\code\mmpose\configs\hand_2d_keypoint\topdown_heatmap\rhd2d\td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256_cpr.py'
    checkpoint_file = r'C:\Users\lahir\code\mmpose\work_dirs\td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256_cpr\best_AUC_epoch_60.pth'
    # pose_model=init_pose_model(config_file,checkpoint_file)

    inferencer = MMPoseInferencer(
        pose2d=config_file,
        pose2d_weights=checkpoint_file
    )
    # inferencer = MMPoseInferencer('td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256')

    img_files=utils.list_files(data_path,'jpg')
    for img_file in img_files:
        img=cv2.imread(os.path.join(data_path,img_file))
        result_generator = inferencer(img, show=False)
        result=next(result_generator)
        kypts=result['predictions'][0][0]['keypoints'][0]
        # Draw a red point on the image
        cv2.circle(img, (int(kypts[0]),int(kypts[1])), 5, (0, 0, 255), -1)
        # Show the image
        cv2.imshow('Image', img)
        # Wait for a key press and then close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# annot_file=r'C:\Users\lahir\Downloads\cpr_data\annotations\train.json'
# def read_json_file(file_path):
#     import json
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

# # Example usage
# import cv2
# import os
# data_path=r'C:\Users\lahir\Downloads\cpr_data'
# data = read_json_file(annot_file)
# img_name='train/P22_s_12_01063.jpg'
# values=[item for item in data['images'] if item['file_name']==img_name]
# id=values[0]['id']  
# h,w=values[0]['height'],values[0]['width']
# [item for item in data['annotations'] if item['image_id']==id][0]['keypoints'][0]
# img = cv2.imread(os.path.join(data_path,img_name))
# cv2.imshow('Image', img)
# # Wait for a key press and then close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()



def detect_kypts_mmpose(model_name,root_dir):
    from mmdet.apis import init_detector, inference_detector,DetInferencer
    from mmpose.apis import MMPoseInferencer
    import numpy as np
    from mmcv.image import imread
    from PIL import Image
    import cv2

    models={
        'RHD2D' : 'td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256',
        'coco' : 'td-hm_hrnetv2-w18_dark-8xb32-210e_coco-wholebody-hand-256x256',
        'onehand10k' : 'td-hm_hrnetv2-w18_dark-8xb64-210e_onehand10k-256x256'
    }
    if model_name == 'finetuned_RHD2D':
        config_file = r'C:\Users\lahir\code\mmpose\configs\hand_2d_keypoint\topdown_heatmap\rhd2d\td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256_cpr.py'
        checkpoint_file = r'C:\Users\lahir\code\mmpose\work_dirs\td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256_cpr\best_AUC_epoch_110.pth'
    elif 'RHD2D' in model_name:
        model_str=models[model_name]
        print(f'Using model: {model_str}')

    subj_dirs=utils.get_dirs_with_str(root_dir, 'P')
    for subj_dir in subj_dirs:
        session_dirs=utils.get_dirs_with_str(subj_dir,'s')
        for session_dir in session_dirs:
            # if session_dir!=r'D:\CPR_extracted\P10\s_5':
            #     continue
            print(session_dir)
            handbb_path=os.path.join(session_dir,'kinect','hand_bbs.json')
            if os.path.exists(handbb_path):
                with open(handbb_path, 'r') as file:
                    hand_bbs = json.load(file)
            else:
                print(f'{handbb_path} does not exist')
                continue
            img_dir=os.path.join(session_dir,'kinect','color')
            img_files=utils.list_files(img_dir,'jpg')
            destination_directory=os.path.join(session_dir,'kinect','wrist_keypts')
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
            destination_file=os.path.join(destination_directory,f'hand_keypts_{model_name}.json')
            if os.path.exists(destination_file):
                print(f'{destination_file} exists. Continuing...')
                continue
            if model_name=='finetuned_RHD2D':
                inferencer = MMPoseInferencer(
                pose2d=config_file,
                pose2d_weights=checkpoint_file
                )
            elif 'RHD2D' in model_name:
                inferencer = MMPoseInferencer(model_str)

            with open(destination_file,'w') as file:
                output={}
                gray0=-1
                for i,img_file in enumerate(img_files):
                    print(f'Processing {i}/{len(img_files)}',end='\r')
                    img_path=os.path.join(img_dir,img_file)
                    img=cv2.imread(img_path)

                    if 'RHD2D' in model_name:
                        if hand_bbs[img_file.split('.')[0]]=='':
                            hand_bb=last_bb
                        else:
                            hand_bb=[int(val) for val in hand_bbs[img_file.split('.')[0]].split(',')]
                        last_bb=hand_bb
                        pad=80
                        img_crop= utils.crop_img_bb(img,hand_bb,pad)

                        result_generator = inferencer(img_crop, show=False)
                        result=next(result_generator)
                        kypts=result['predictions'][0][0]['keypoints']
                        x_vals=[val[0]+hand_bb[0]-pad for val in kypts]
                        y_vals=[val[1]+hand_bb[1]-pad for val in kypts]
                    elif model_name=='tracking':
                        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        if i==0:
                            gray0=gray
                            p=utils.select_points(gray)
                            point = np.array([[p[:2]]], dtype=np.float32)
                            x_vals=[int(p[0])]
                            y_vals=[int(p[1])]
                        else:
                            # Parameters for Lucas-Kanade optical flow
                            lk_params = {
                                'winSize': (15, 15),
                                'maxLevel': 2,
                                'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                            }
                            new_points, status, _ = cv2.calcOpticalFlowPyrLK(gray0, gray, point, None, **lk_params)
                            x, y = new_points.ravel()
                            x_vals=[int(x)]
                            y_vals=[int(y)]
                            # cv2.circle(gray, (int(x), int(y)), 5, (0, 255, 0), -1)
                            # utils.show_img(gray)
                            # continue

                    sub_dict={"x":x_vals,"y":y_vals}
                    output[img_file.split('.')[0]]=sub_dict

                    # Display the image
                    # for k in kypts:
                    #     cv2.circle(img_crop, (int(k[0]), int(k[1])), 5, (0, 255, 0), -1)
                    # cv2.imshow('Image', img_crop)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                json.dump(output, file)

#object detection
def get_personBB(root_dir):
    root_dir='D:\CPR_data_raw'
    subj_dirs=[os.path.join(root_dir,item,'extracted') for item in utils.list_subdirectories(root_dir) if item[0].lower()=='p']
    for subj_dir in subj_dirs:
        session_dirs=[os.path.join(subj_dir,session_dir) for session_dir in utils.list_subdirectories(subj_dir) if session_dir[0].lower()=='s']
        out_file=os.path.join(session_dir,'mean_person_bbx.txt')
        if os.path.exists(out_file):
            continue
        for session_dir in session_dirs:
            img_dir=os.path.join(session_dir,'color')
            img_files=utils.list_files(img_dir,'jpg')
            x,y,width,height=0,0,0,0
            n=0
            print(session_dir)
            for img_file in img_files:
                img_path=os.path.join(img_dir,img_file)
                bbx=detect_object(img_path,'person')
                if bbx==-1:
                    continue
                n+=1
                x+=bbx.origin_x
                y+=bbx.origin_y
                width+=bbx.width
                height+=bbx.height
            if n>100:
                mean_x=x/n
                mean_y=y/n
                mean_width=width/n
                mean_height=height/n
                with open(out_file,'w') as file:
                    file.write("mean_x,mean_y,mean_width,mean_height \n")
                    file.write(f'{int(mean_x)},{int(mean_y)},{int(mean_width)},{int(mean_height)}')
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect Keypoints')
    parser.add_argument('--data_path', type=str, default=r'D:\CPR_extracted', help='Path to data directory')
    parser.add_argument('--model_name', type=str, default='tracking', help='model used to detect keypoints. finetuned_RHD2D: mmpose model trained on RHD2D finetuned on our dataset, tracking: use opencv tracking')
    args=parser.parse_args()
    
    detect_kypts_mmpose(args.model_name,args.data_path)
    # detect_kypts_mp(args.data_path)
    # detec_keypoints_dir_mmpose(r'C:\Users\lahir\Downloads\cpr_data\test','RHD2D')



# from mmdet.apis import init_detector, inference_detector,DetInferencer
# from mmpose.apis import MMPoseInferencer
# import numpy as np
# from mmcv.image import imread
# from PIL import Image
# import cv2

# models={
#     'RHD2D' : 'td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256',
#     'coco' : 'td-hm_hrnetv2-w18_dark-8xb32-210e_coco-wholebody-hand-256x256',
#     'onehand10k' : 'td-hm_hrnetv2-w18_dark-8xb64-210e_onehand10k-256x256'
# }
# model_str=models['RHD2D']
# inferencer = MMPoseInferencer(model_str)
# result_generator = inferencer(r'C:\Users\lahir\Downloads\c.jpg', show=True)
# result = next(result_generator)








