import csv
import numpy as np
import os
import sys
import csv

def mykmeans(kjz1):    #2分类
    bj=1;kjz1=np.sort(kjz1)
    while(True):
        if bj==1:
            kj=np.mean([kjz1[0],kjz1[len(kjz1)-1]]) #初始分组均值使用最小值和最大值的平均值
        else:
            k1=s1;k2=s2;
            kj=np.mean([k1,k2]);
        kjz2=[[],[]];
        for j in kjz1:
            if j<=kj:
                kjz2[0].append(j)
            else:
                kjz2[1].append(j)
        s1=np.mean(kjz2[0]);
        s2=np.mean(kjz2[1]);
        if bj==2:
            if s1==k1 and s2==k2:
                break
        bj=2;
    return kjz2

file_path = '/srv/wuyichao/Super-Resolution/KPSAGAN/BasicSR-master/BasicSR-master-c/codes/csv/'
f = os.listdir(file_path)
f.sort
for file in f:
    print (file)
    total_path = os.path.join(file_path,file)
    myfile =open(total_path,'r')
    lines=myfile.readlines()
    myfile.close()
    row=[]
    for line in lines:
        row.append(line.split(','))
    co32 = 0.0
    co31 = 0.0
    co43 = 0.0
    co42 = 0.0
    co41 = 0.0
    for i in range(159,169):
        c32 = float(row[i][2])/(float(row[i][2])+float(row[i][3]))
        c31 = float(row[i][3])/(float(row[i][2])+float(row[i][3]))
        c43 = float(row[i][4])/(float(row[i][4])+float(row[i][5])+float(row[i][6]))
        c42 = float(row[i][5])/(float(row[i][4])+float(row[i][5])+float(row[i][6]))
        c41 = float(row[i][6])/(float(row[i][4])+float(row[i][5])+float(row[i][6]))
        co32 = co32 +c32
        co31 = co31 +c31
        co43 = co43 +c43
        co42 = co42 +c42
        co41 = co41 +c41
    co32 = co32/10
    co31 = co31/10
    c3 =[co32, co31]
    co43 = co43/10
    co42 = co42/10
    co41 = co41/10
    c4 = [co43, co42, co41]
    if (max(c3)<(1/2*1.05)) & (max(c3)>(1/2*0.95)):
        print('retain all the connection to x3')
    else:
        x = mykmeans(c3)
        for i in range (len(x[0])):
            ind = c3.index(x[0][i])
            if ind==0:
                print('cut x2 to x3')
            elif ind ==1:
                print('cut x1 to x3')
    if (max(c4)<(1/3*1.05)) & (max(c4)>(1/3*0.95)):
        print('retain all the connection to x4')
    else:
        y = mykmeans(c4)
        for i in range (len(y[0])):
            ind = c4.index(y[0][i])
            if ind==0:
                print('cut x3 to x4')
            elif ind ==1:
                print('cut x2 to x4')
            elif ind ==2:
                print('cut x1 to x4')
