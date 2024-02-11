# from mmpose.apis import MMPoseInferencer

# class PredictKypt:
#     '''
#     model_name                                                              AUC            EPE
#     _________________________________________________________________________________________________
#     td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256
#     td-hm_hrnetv2-w18_dark-8xb32-210e_coco-wholebody-hand-256x256
#     td-hm_hrnetv2-w18_dark-8xb64-210e_onehand10k-256x256                    0.572           23.96
#     td-hm_hrnetv2-w18_udp-8xb64-210e_onehand10k-256x256                     0.571           23.88
#     '''
#     def __init__(self,model_name='td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256'):
#         self.model_name=model_name
#         self.inferencer = MMPoseInferencer(model_name)

#     def show_kypts(self):
#         import matplotlib.pyplot as plt
#         import matplotlib.image as mpimg

#         keypoints=self.result['keypoints']
#         x=[k[0] for k in keypoints]
#         y=[k[1] for k in keypoints]
#         image = mpimg.imread(self.img_path)
#         plt.imshow(image)
#         plt.scatter(x,y)
#         plt.show()

#     def save_kypts(self,path):
#         import cv2

#         keypoints=self.result['keypoints']
#         if self.result['bbox_score']<0.8:
#             image = cv2.imread(self.img_path)
#             for point in keypoints:
#                 cv2.circle(image, (round(point[0]),round(point[1])), radius=5, color=(0, 0, 255), thickness=-1)
#             cv2.imwrite(path, image)
        
    
#     def get_kypts(self,img_path):
#         self.img_path=img_path
#         result_generator = self.inferencer(img_path, show=False)
#         result = next(result_generator)
#         self.result=result['predictions'][0][0]
#         return self.result 



# #how to use this class
# img_path=r'C:\Users\lahir\data\kinect_hand_data\hands.png'        
# kp=PredictKypt()
# pred=kp.get_kypts(img_path)
# #get keypoints from the prediction result
# keypoints=pred['keypoints']
# #show the hand image with keypoints
# kp.show_kypts()


# import cv2
# from mmpose.apis import init_model,inference_topdown
# img = cv2.imread(img_path)

# # Initialize the model
# cfg = 'configs/hand/onehand10k/coco/256x256.py'
# checkpoint = 'onehand10k_coco_256x256-8c7b4f7c.pth'
# model = init_model(cfg, checkpoint, device='cpu')


#using ikomia
# from ikomia.dataprocess.workflow import Workflow
# from ikomia.utils import ik
# from ikomia.utils.displayIO import display


# wf = Workflow()

# algo = wf.add_task(name = 'infer_mmlab_pose_estimation', auto_connect=True)

# wf.run_on(url="https://cdn.nba.com/teams/legacy/www.nba.com/bulls/sites/bulls/files/jordan_vs_indiana.jpg")

# display(algo.get_image_with_graphics())


# wf = Workflow()
# pose = wf.add_task(ik.infer_mmlab_pose_estimation(
#         config_file = "configs/hand_2d_keypoint/topdown_regression/onehand10k/td-reg_res50_8xb64-210e_onehand10k-256x256.py",
#         conf_thres = '0.5',
#         detector = "Hand"
#         ),
#         auto_connect=True
# )
# wf.run_on(path=r"C:\Users\lahir\Downloads\00108_crp.jpg")
# display(pose.get_image_with_graphics())


# from ikomia.dataprocess.workflow import Workflow
# from ikomia.utils import ik
# from ikomia.utils.displayIO import display

# wf = Workflow()

# yolov7 = wf.add_task(ik.infer_yolo_v7_instance_segmentation(), auto_connect=True)

# # wf.run_on(path="path/to/your/image.png")
# wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_dog.png")

# display(yolov7.get_image_with_graphics())
# display(yolov7.get_image_with_mask())
# display(yolov7.get_image_with_mask_and_graphics())
# ik

# from mmdet.apis import init_detector, inference_detector,DetInferencer
# from mmpose.apis import MMPoseInferencer
# import numpy as np
# from mmcv.image import imread
# from PIL import Image

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

# img=imread(r'C:\Users\lahir\Downloads\00868.jpg')
# inferencer = DetInferencer('yolof_r50_c5_8x8_1x_coco')
# pred=inferencer(img, show=True)
# s=256

# labels=pred['predictions'][0]['labels']
# scores=pred['predictions'][0]['scores']
# person_index=labels.index(0)
# person_score=scores[person_index]
# if person_score>0.4:
#     #person detected
#     bbox=pred['predictions'][0]['bboxes'][person_index]
#     x0,x1,y0,y1=int(bbox[0]),int(bbox[2]),int(bbox[1]),int(bbox[3])
#     x0=max(x0,0)
#     x1=min(x1,img.shape[1])
#     y0=max(y0,0)
#     y1=min(y1,img.shape[0])
#     #create padding
#     x_len,y_len=x1-x0,y1-y0
#     x_=int((x0+x1)*0.5)
#     hnd_img=img[y1-s:y1+s,x_-s:x_+s,:]

# hnd_img=Image.fromarray(img).resize((512,512))
# hnd_img=np.array(hnd_img)
# hnd_img.show()

# # inferencer = MMPoseInferencer('td-hm_hrnet-w32_8xb64-210e_coco-256x192')
# inferencer = MMPoseInferencer('rtmpose-m_8xb256-210e_hand5-256x256')
# result_generator = inferencer(hnd_img, show=True)
# result = next(result_generator)

# result['predictions'][0][0]['bbox']

import cv2
import mediapipe as mp
import numpy as np

class WristDet:
    def __init__(self):
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands()
        mp_drawing = mp.solutions.drawing_utils
    
    def get_kypts(self,path):
        image = cv2.imread(path)
        height, width, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect hands in the image
        results = self.hands.process(rgb_image)
        xy_vals=[]
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(21):
                    x = int(hand_landmarks.landmark[i].x * width)
                    y = int(hand_landmarks.landmark[i].y * height)
                    xy_vals.append((x,y))
        return image,xy_vals

def plot_points(image,points):
        # Draw circles at the specified points
        for point in points:
            cv2.circle(image, point, 5, (0, 255, 0), -1)  # (0, 255, 0) is the color in BGR format, -1 is to fill the circle
    
wrst=WristDet()
image,xy_vals=wrst.get_kypts(r'D:\CPR_data_raw\P15\extracted\s_2\color\00479.jpg')
plot_points(image,xy_vals)
show_img(image)


x_vals=[item[0] for item in xy_vals]
y_vals=[item[1] for item in xy_vals]

min_x,max_x=min(x_vals),max(x_vals)
min_y,max_y=min(y_vals),max(y_vals)
padding=int((max_x-min_x+max_y-min_y)*0.5*1)
w,h,_=image.shape
hand_img=image[max(0,min_y-padding):min(w,max_y+padding),max(0,min_x-padding):min(h,max_x+padding+20),:]
show_img(hand_img)


# hand_img = cv2.imread(r'D:\CPR_data_raw\P15\extracted\s_3\color\01103.jpg')
# img=imread(r'D:\CPR_data_raw\P19\extracted\s_2\color\01787.jpg')
inferencer = MMPoseInferencer('td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256')
# inferencer = MMPoseInferencer('hand')
result_generator = inferencer(hand_img, show=True)
result = next(result_generator)
wrist=result['predictions'][0][0]['keypoints'][0]
wrist=[int(w) for w in wrist]
cv2.circle(hand_img, (wrist[0],wrist[1]), 5, (0, 255, 0), 2)  # (0, 255, 0) is the color in BGR format, and 2 is the thickness
show_img(hand_img)










