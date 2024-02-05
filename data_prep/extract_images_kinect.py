import os
import subprocess

command="python azure_kinect_mkv_reader.py --input "

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def get_subdirectories(directory):
    # Get a list of all entries in the directory
    entries = os.listdir(directory)

    # Filter only directories
    subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]

    return subdirectories

def delete_empty_directory(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        try:
            # Check if the directory is empty
            if not os.listdir(directory_path):
                os.rmdir(directory_path)
                print(f"Empty directory '{directory_path}' deleted successfully.")
            else:
                print(f"Directory '{directory_path}' is not empty.")
        except Exception as e:
            print(f"Error deleting directory '{directory_path}': {e}")
    else:
        print(f"Directory '{directory_path}' does not exist or is not a directory.")


root_dir='D:\CPR_data_raw'
for i in range(5,24):
    sub_dir=os.path.join(root_dir,"P"+str(i))
    subdirectories_list = get_subdirectories(sub_dir)
    session_dirs=[dir for dir in subdirectories_list if ("s" in dir[0]) or ("S" in dir[0])]
    for dir in session_dirs:
        dir_path=os.path.join(sub_dir,dir,'kinect')
        files=os.listdir(dir_path)
        if len(files)==0:
            continue
        try:
            mkv_file=[f for f in files if f.split('.')[-1]=='mkv'][0]
        except:
            continue
        mkv_file_path=os.path.join(dir_path,mkv_file)
        output_dir=os.path.join(dir_path,"images")
        # delete_empty_directory(output_dir)
        this_cmd=command+mkv_file_path+' --output ' + output_dir
        print(this_cmd)
        #excec command
        result = subprocess.run(this_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)









