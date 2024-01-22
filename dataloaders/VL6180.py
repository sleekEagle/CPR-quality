import matplotlib.pyplot as plt

path='D:\\CPR_data_raw\\P0\\s1\\arduino\\2023-12-03-12_46_42.txt'

def get_data(path):
    with open(path, 'r') as file:
        # Read all lines into a list
        all_lines = file.readlines()

    ts_list,depth_list=[],[]
    found_first_line=False
    for l in all_lines:
        if 'Sensor found!' in l:
            found_first_line=True
            continue
        if not found_first_line:
            continue
        line=l.strip()
        splt=line.split(' ')
        ts=''.join(splt[0:2])
        depth=float(splt[-1])
        ts_list.append(ts)
        depth_list.append(depth)
    return ts_list,depth_list

ts_list,depth_list=get_data(path)
plt.plot(depth_list)
plt.show()





