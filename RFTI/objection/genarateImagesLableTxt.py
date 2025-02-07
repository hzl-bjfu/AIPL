# -*- coding: utf-8  -*-
import os
from PIL import Image
import cv2


# '''
# 将未整理的原始图像数据保存到images.txt
# '''
# root = 'F:\SoftwareProgram\Data\dataset\snow leopard/original'
# f1 = open('F:\SoftwareProgram\Data\dataset\snow leopard/original/images.txt', 'w')
# idx = 0
# for r, d, files in os.walk(root):
#     if files != []:
#         for i in files:
#             if i.split('.')[-1] == 'JPG':
#                 fp = os.path.join(r, i)
#                 # print(fp)
#                 f1.write('{}\n'.format(fp))
#                 idx += 1
#
# print(idx)  #1424张
# f1.close()



'''
按照文件夹 读取正负样本到各自的列表变量
'''
n_1_path = 'F:\SoftwareProgram\Data\dataset\snow leopard\整理/n/1'
original_1_path = 'F:\SoftwareProgram\Data\dataset\snow leopard/original\\1'
images = os.listdir(n_1_path)
n_1_imageName = []
for image_name in images:
    image_path = original_1_path + '\\' + image_name
    n_1_imageName.append(image_path)
    # if image_path in original_imglist:
print('1'*20)

n_2_path = 'F:\SoftwareProgram\Data\dataset\snow leopard\整理/n/2'
original_2_path = 'F:\SoftwareProgram\Data\dataset\snow leopard/original\\2'
images = os.listdir(n_2_path)
n_2_imageName = []
for image_name in images:
    image_path = original_2_path + '\\' + image_name
    n_2_imageName.append(image_path)
    # if image_path in original_imglist:
print('2'*20)

p_100MEDIA_path = 'F:\SoftwareProgram\Data\dataset\snow leopard\整理/p/100MEDIA'
original_100MEDIA_path = 'F:\SoftwareProgram\Data\dataset\snow leopard/original\\100MEDIA'
images = os.listdir(p_100MEDIA_path)
p_100MEDIA_imageName = []
for image_name in images:
    image_path = original_100MEDIA_path + '\\' + image_name
    p_100MEDIA_imageName.append(image_path)
    # if image_path in original_imglist:
print('3'*20)


'''
将images.txt中的每一行附上标签,生成images_lable.txt
'''
f2 = open('F:\SoftwareProgram\Data\dataset\snow leopard/original/images_lable.txt', 'w')
fid =open('F:\SoftwareProgram\Data\dataset\snow leopard/original/images.txt', 'r')
original_imglist = fid.readlines()

for image in original_imglist:
    image = image.strip()
    if image.split('original\\')[-1][:2] == '1\\':
        # if 'HED-061A-0026.JPG' in image:
        #     print('look at me!')
        if image in n_1_imageName:
            f2.write('{},{}\n'.format(image,0))
        else:
            f2.write('{},{}\n'.format(image,1))
    if image.split('original\\')[-1][:2] == '2\\':
        if image in n_2_imageName:
            f2.write('{},{}\n'.format(image,0))
        else:
            f2.write('{},{}\n'.format(image,1))
    if image.split('original\\')[-1][:8] == '100MEDIA':
        if image in p_100MEDIA_imageName:
            f2.write('{},{}\n'.format(image,1))
        else:
            f2.write('{},{}\n'.format(image,0))

f2.close()
fid.close()