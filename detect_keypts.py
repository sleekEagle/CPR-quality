import cv2
import mediapipe as mp
import numpy as np
import utils
import os
import json

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
            cv2.circle(image, point, 5, (0, 255, 0), -1)  # (0, 255, 0) is the color in BGR format, -1 is to fill the circle

class WristDet_mediapipe:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
    
    def get_kypts(self,path):
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
                image = cv2.imread(path)
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
    
wrst=WristDet_mediapipe()
root_dir='D:\CPR_data_raw'
subj_dirs=[os.path.join(root_dir,item,'extracted') for item in utils.list_subdirectories(root_dir) if item[0].lower()=='p']
for subj_dir in subj_dirs:
    session_dirs=[os.path.join(subj_dir,session_dir) for session_dir in utils.list_subdirectories(subj_dir) if session_dir[0].lower()=='s']
    for session_dir in session_dirs:
        print(session_dir)
        img_dir=os.path.join(session_dir,'color')
        img_files=utils.list_files(img_dir,'jpg')
        destination_directory=os.path.join(session_dir,'wrist_keypts')
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)
        destination_file=os.path.join(destination_directory,'hand_keypts_mediapipe.json')
        if os.path.exists(destination_file):
            continue
        with open(destination_file,'w') as file:
            output={}
            for img_file in img_files:
                img_path=os.path.join(img_dir,img_file)
                image,xy_vals=wrst.get_kypts(img_path)
                if len(xy_vals)==0:
                    output[img_file.split('.')[0]]={}
                    continue
                assert len(xy_vals)==21, f'Number of key points is not 21 for {img_file}'
                x_vals=[val[0] for val in xy_vals]
                y_vals=[val[1] for val in xy_vals]
                sub_dict={"x":x_vals,"y":y_vals}
                output[img_file.split('.')[0]]=sub_dict
            json.dump(output, file)


