import json
import numpy as np
import matplotlib.pyplot as plt

# Specify the path to your JSON file
file_path = r'D:\CPR_extracted\P1\s_2\kinect\wrist_keypts\kypts.json'

# Open the JSON file
with open(file_path, 'r') as f:
    data = [json.loads(line) for line in f]

data[0]['data_row']['details']
['external_id']

x_list,y_list=[],[]
img_name_list=[]
for i,d in enumerate(data):
    cords=d['projects'][list(d['projects'].keys())[0]]['labels'][0]['annotations']['objects'][0]['point']
    x_list.append(cords['x'])
    y_list.append(cords['y'])
    img_name_list.append(int(d['data_row']['external_id'].split('.')[0]))

img_name_list=np.array(img_name_list)
sorted_ind = np.argsort(img_name_list)

#write bach to nive json file
out_dict={}
for i in range(len(x_list)):
    x_vals=[x_list[i]]
    x_vals.extend([-1]*20)
    y_vals=[y_list[i]]
    y_vals.extend([-1]*20)
    
    out_dict[str(img_name_list[i]).zfill(5)]={'x':x_vals,'y':y_vals}

# Write out_dict to a JSON file
output_file = r'D:\CPR_extracted\P1\s_2\kinect\wrist_keypts\hand_keypts_test.json'
with open(output_file, 'w') as f:
    json.dump(out_dict, f)



import utils
import os
import matplotlib.pyplot as plt

root_dir='D:\\CPR_extracted'
subj_dirs=utils.get_dirs_with_str(root_dir, 'P')
for subj_dir in subj_dirs:
    session_dirs=utils.get_dirs_with_str(subj_dir,'s')
    for session_dir in session_dirs:
        try:
            print(session_dir)
            img_dir=os.path.join(session_dir,'kinect','color')
            img_list=os.listdir(img_dir)
            img_list.sort()
            img1=img_list[0]
            img2=img_list[-1]

            img1_path = os.path.join(img_dir, img1)
            img2_path = os.path.join(img_dir, img2)

            img1_data = plt.imread(img1_path)
            img2_data = plt.imread(img2_path)

            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(img1_data)
            plt.title(img1)

            plt.subplot(1, 2, 2)
            plt.imshow(img2_data)
            plt.title(img2)

            plt.show()
        except:
            print('exception')

