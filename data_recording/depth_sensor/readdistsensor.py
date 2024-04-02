import csv
import matplotlib.pyplot as plt
from datetime import datetime

file=r'C:\Users\lahir\Documents\CPR Quality Data\CoolTerm Capture 2022-10-20 15-15-25.txt'

def getms(tsstring):
    dt_obj = datetime.strptime(tsstring,'%H:%M:%S.%f')
    ms=(dt_obj.hour*60*60+dt_obj.minute*60+dt_obj.second)+dt_obj.microsecond/100
    return ms

def get_depthdata(file):
    ts,val=[],[]
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            try:
                splt= (', '.join(row)).split()
                ts_=splt[0]
                ts_=getms(ts_)
                val_=int(splt[1])
                ts.append(ts_)
                val.append(val_)
            except Exception as e:
                print(str(e))
    return ts,val
