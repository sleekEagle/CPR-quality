import cv2
import mediapipe as mp
import numpy as np
import sys
sys.path.append('C:\\Users\\lahir\\code\\CPR-quality\\')
import utils
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import os

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




def detect_kypts_mp():
    wrst=WristDet_mediapipe()
    root_dir='D:\CPR_extracted'
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



#object detection
def get_personBB():
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

# get_personBB()
detect_kypts_mp()








