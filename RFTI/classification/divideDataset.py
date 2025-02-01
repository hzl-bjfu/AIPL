import os
import shutil
import numpy as np
import time


# '''
# images_allCamera
# '''
# time_start = time.time()
#
# # 文件路径
#
# f1 = open('./data/allCamera/images_label.txt','r')
# f2 = open('./data/allCamera/train_allCamera.txt','w')
# f3 = open('./data/allCamera/test_allCamera.txt','w')
#
# imagesList = f1.readlines()
# np.random.shuffle(imagesList)
#
# train_len = int(len(imagesList)*0.8)
# for i in range(train_len):  #1947
#     # originalImagesList[i] = originalImagesList[i].strip()
#     f2.write('{}'.format(imagesList[i]))
# for i in range(train_len,len(imagesList)):  #487
#     # originalImagesList[i] = originalImagesList[i].strip()
#     f3.write('{}'.format(imagesList[i]))
#
# f1.close()
# f2.close()
# f3.close()
#
# time_end = time.time()
# print('allCamera训练集和测试集划分完毕, 耗时%s!!' % (time_end - time_start))



# '''
# images_eachCamera
# '''
# time_start = time.time()
# cameraNumber = '089'
# # 文件路径
# #004、089
# f1 = open('./data/eachCamera/images_label_beishanshuju'+ cameraNumber +'.txt','r')
# f2 = open('./data/eachCamera/train_beishanshuju'+ cameraNumber +'.txt','w')
# f3 = open('./data/eachCamera/test_beishanshuju'+ cameraNumber +'.txt','w')
#
# imagesList = f1.readlines()
# np.random.shuffle(imagesList)
#
# train_len = int(len(imagesList)*0.8)
# for i in range(train_len):  #1947
#     # originalImagesList[i] = originalImagesList[i].strip()
#     f2.write('{}'.format(imagesList[i]))
# for i in range(train_len,len(imagesList)):  #487
#     # originalImagesList[i] = originalImagesList[i].strip()
#     f3.write('{}'.format(imagesList[i]))
#
# f1.close()
# f2.close()
# f3.close()
#
# time_end = time.time()
# print('eachCamera训练集和测试集划分完毕, 耗时%s!!' % (time_end - time_start))



'''
images_allCamera_2n
'''
time_start = time.time()

# 文件路径

f1 = open('./data/allCamera/images_label_2n.txt','r')
f2 = open('./data/allCamera/train_allCamera_2n.txt','w')
f3 = open('./data/allCamera/test_allCamera_2n.txt','w')

imagesList = f1.readlines()
np.random.shuffle(imagesList)

train_len = int(len(imagesList)*0.8)
for i in range(train_len):  #1947
    # originalImagesList[i] = originalImagesList[i].strip()
    f2.write('{}'.format(imagesList[i]))
for i in range(train_len,len(imagesList)):  #487
    # originalImagesList[i] = originalImagesList[i].strip()
    f3.write('{}'.format(imagesList[i]))

f1.close()
f2.close()
f3.close()

time_end = time.time()
print('allCamera_2n训练集和测试集划分完毕, 耗时%s!!' % (time_end - time_start))