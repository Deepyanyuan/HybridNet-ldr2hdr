# coding:utf-8
'''
generate training data pairs,
including one log-domain HDR image, one original-domain HDR image and their normalization images, total four HDR images,
9 multi-exposure LDR images
'''

import numpy as np
import cv2
import glob, argparse, math
import OpenEXR
import Imath
import imageio
import os, sys
import datetime
from tqdm import tqdm
import random

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', help='Directory path of hdr images.', default='./basedata/hdr')       # load directory path of hdr images
parser.add_argument('-o', help='Directory path of ldr images.', default='./training_samples')   # save directory path of ldr images

args = parser.parse_args()


# define camera response curve function
# def func_0(x):
#     result = 0.02075*np.power(x, 3) + 0.5034 * np.power(x, 2) + 0.4727 * x - 0.001136
#     result[result>1.0]=1.0
#     result[result<0.0]=0.0
#     return result
def func_1(x):
    result = 0.9491 * np.power(x, 3) - 2.97 * np.power(x, 2) + 3.114 * x - 0.1031
    result[result > 1.0] = 1.0
    result[result < 0.0] = 0.0
    return result


def func_2(x):
    result = 0.2108 * np.power(x, 3) - 0.9448 * np.power(x, 2) + 1.711 * x + 0.0246
    result[result > 1.0] = 1.0
    result[result < 0.0] = 0.0
    return result


def func_3(x):
    result = 2.909 * np.power(x, 3) - 5.858 * np.power(x, 2) + 3.908 * x + 0.0883
    result[result > 1.0] = 1.0
    result[result < 0.0] = 0.0
    return result


def func_4(x):
    result = 1.462 * np.power(x, 3) - 3.16 * np.power(x, 2) + 2.618 * x + 0.1047
    result[result > 1.0] = 1.0
    result[result < 0.0] = 0.0
    return result


func_dict = {'mark1': func_1, 'mark2': func_2, 'mark3': func_3, 'mark4': func_4}
mark_list = ['mark1', 'mark2', 'mark3', 'mark4']  


# define exposure time function
def exposure_times(tao, T):
    delt_t = list()
    tt = int(T / 2 + 1)
    for t in range(tt):
        delt_t_ = math.pow(1 / tao, t)
        delt_t.append(delt_t_)
    delt_t.reverse()
    for t in range(tt - 1):
        delt_t_ = math.pow(tao, t + 1)
        delt_t.append(delt_t_)
    delt_t = np.array(delt_t)

    return delt_t


tao = math.sqrt(2)  
T = 8  
normal_value = 3  
dir_in_path_list = glob.glob(args.i + '/*')

dir_in_path_list = dir_in_path_list[:] 
max_hdrs = [33, 200, 27]

dir_out_path = glob.glob(args.o)
Times = exposure_times(tao, T)  

start = datetime.datetime.now()
N = len(dir_in_path_list)
for i in tqdm(range(N)):
    dir_in_path = dir_in_path_list[i] 
    filename_root = os.path.basename(dir_in_path) 
    files_hdr_path_list = glob.glob(dir_in_path + '/*.hdr')
    current_hdr_max = max_hdrs[i]

    for file_num, file in enumerate(files_hdr_path_list):
        if file_num % 1 == 0:
            hdr = cv2.imread(file, flags=cv2.IMREAD_ANYDEPTH)           # read HDR dataset 

            hdr_0 = hdr + (10 ** -8)  
            filename_hdr, file_format = os.path.splitext(file) 
            filename_sub = os.path.basename(filename_hdr) 
            print('file name:', filename_sub)
            hdr_log = np.log10(hdr_0 + 1)
            hdr_log_norm = (hdr_log - np.min(hdr_log)) / (np.log10(current_hdr_max + 1)) 
            hdr_norm = (hdr_0 - np.min(hdr_0)) / (3 * np.mean(hdr_0) - np.min(hdr_0)) 
            hdr_0_norm = hdr_0 / current_hdr_max

            hdr_norm_exposure = list()
            for i in range(T + 1):
                hdr_norm_exposure.append(hdr_norm * Times[i])
            hdr_norm_exposure = np.array(hdr_norm_exposure)

            for i in range(len(mark_list)):
                mark = mark_list[i]
                ldr_norm_temp = func_dict[mark](hdr_norm_exposure)  
                save_root_path = dir_out_path[0] + '/' + filename_root + '_' + filename_sub + '_' + mark + '_sub'

                count = 0
                image_each = 3
                exposure_N, height, width, channel = np.shape(ldr_norm_temp)
                img_patch = np.min([height, width])
                if img_patch > 1023:  
                    while count < image_each:
                        width1 = random.randint(0, width - img_patch)
                        height1 = random.randint(0, height - img_patch)
                        width2 = width1 + img_patch
                        height2 = height + img_patch
                        cut_hdr_temp_0 = hdr_log[height1:height2, width1:width2, :]  
                        cut_hdr_temp_1 = hdr_log_norm[height1:height2, width1:width2, :] 
                        cut_hdr_temp_2 = hdr_0[height1:height2, width1:width2, :]  
                        cut_hdr_temp_3 = hdr_0_norm[height1:height2, width1:width2, :] 
                        cut_ldr_temp = ldr_norm_temp[:, height1:height2, width1:width2, :] 

                        re_size = (256, 256)
                        shrink_cut_hdr_temp_0 = cv2.resize(cut_hdr_temp_0, re_size, interpolation=cv2.INTER_AREA)
                        shrink_cut_hdr_temp_1 = cv2.resize(cut_hdr_temp_1, re_size, interpolation=cv2.INTER_AREA)
                        shrink_cut_hdr_temp_2 = cv2.resize(cut_hdr_temp_2, re_size, interpolation=cv2.INTER_AREA)
                        shrink_cut_hdr_temp_3 = cv2.resize(cut_hdr_temp_3, re_size, interpolation=cv2.INTER_AREA)

                        num_str = str(count + 1).rjust(2, '0')
                        savepath = save_root_path + num_str
                        class_H_path = savepath + '/HDR'
                        class_L_path = savepath + '/LDR'

                        os.makedirs(class_H_path)
                        os.makedirs(class_L_path)
                        cv2.imwrite(class_H_path + '/0.hdr', shrink_cut_hdr_temp_0)  # save log-domain HDR image
                        cv2.imwrite(class_H_path + '/1.hdr', shrink_cut_hdr_temp_1)  # save log-domain HDR image, normalization
                        cv2.imwrite(class_H_path + '/2.hdr', shrink_cut_hdr_temp_2)  # save original-domain HDR image
                        cv2.imwrite(class_H_path + '/3.hdr', shrink_cut_hdr_temp_3)  # save original-domain HDR image， normalization

                        for n in range(exposure_N):
                            shrink_cut_ldr_temp = cv2.resize(cut_ldr_temp[n] * 255, re_size, interpolation=cv2.INTER_AREA)
                            cv2.imwrite(class_L_path + '/' + str(n) + '.png', shrink_cut_ldr_temp)  # save multi-exposure LDR images
                        count += 1
                else:
                    re_size = (256, 256)
                    shrink_cut_hdr_temp_0 = cv2.resize(hdr_log, re_size, interpolation=cv2.INTER_AREA)
                    shrink_cut_hdr_temp_1 = cv2.resize(hdr_log_norm, re_size, interpolation=cv2.INTER_AREA)
                    shrink_cut_hdr_temp_2 = cv2.resize(hdr_0, re_size, interpolation=cv2.INTER_AREA)
                    shrink_cut_hdr_temp_3 = cv2.resize(hdr_0_norm, re_size, interpolation=cv2.INTER_AREA)
                    cut_ldr_temp = ldr_norm_temp

                    num_str = str(count + 1).rjust(2, '0')
                    savepath = save_root_path + num_str
                    class_H_path = savepath + '/HDR'
                    class_L_path = savepath + '/LDR'

                    os.makedirs(class_H_path)
                    os.makedirs(class_L_path)
                    cv2.imwrite(class_H_path + '/0.hdr', shrink_cut_hdr_temp_0)  # save log-domain HDR image
                    cv2.imwrite(class_H_path + '/1.hdr', shrink_cut_hdr_temp_1)  # save log-domain HDR image, normalization
                    cv2.imwrite(class_H_path + '/2.hdr', shrink_cut_hdr_temp_2)  # save original-domain HDR image
                    cv2.imwrite(class_H_path + '/3.hdr', shrink_cut_hdr_temp_3)  # save original-domain HDR image， normalization

                    for n in range(exposure_N):
                        cut_norm_temp_ = cut_ldr_temp[n] * 255
                        shrink_cut_ldr_temp = cv2.resize(cut_norm_temp_, re_size, interpolation=cv2.INTER_AREA)

                        cv2.imwrite(class_L_path + '/' + str(n) + '.png', shrink_cut_ldr_temp)  # save multi-exposure LDR images

end = datetime.datetime.now()
print(end - start)
print('success!')