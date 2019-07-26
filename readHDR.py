# coding:utf-8
import numpy as np
import cv2
import glob, argparse, math
import OpenEXR
import Imath
import imageio
import os, sys
import datetime
import imageio
import scipy.io as scio
imageio.plugins.freeimage.download()

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', help='Directory path of hdr images.', default='./hdr')     
parser.add_argument('-o', help='Directory path of ldr images.', default='./training_samples') 

args = parser.parse_args()

dir_in_path_list = glob.glob(args.i+'/*')
dir_in_path_list = dir_in_path_list[:] 

# 确定输出文件路径
dir_out_path = glob.glob(args.o)
start = datetime.datetime.now()
N = len(dir_in_path_list)
for i in range(N):
    print('第%d个文件夹'%i)
    dir_in_path = dir_in_path_list[i]      
    print('文件夹名字：',dir_in_path)
    filename_root = os.path.basename(dir_in_path)   
    files_hdr_path_list = glob.glob(dir_in_path+'/*.hdr')
    for file in files_hdr_path_list:
        filename_hdr, file_format = os.path.splitext(file)  
        filename_sub = os.path.basename(filename_hdr)                       
        hdr = imageio.imread(file)        
        hdr_max = np.max(hdr)
        hdr_mean = np.mean(hdr)
        hdr_min = np.min(hdr[hdr!=0.0])
        print('file :{}, min: {}, mean:{}, max:{}'.format(filename_sub,hdr_min,hdr_mean,hdr_max))     