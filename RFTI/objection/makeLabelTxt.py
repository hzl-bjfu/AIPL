# -*- coding: utf-8  -*-
import os
from PIL import Image
import cv2


'''
处理original
将未整理的原始图像数据保存到images.txt
'''
rootPath = 'F:\SoftwareProgram\Data\dataset\snow leopard'
f1 = open(rootPath+'/original/original/images.txt', 'w')
idx = 0
for r, d, files in os.walk(rootPath+'/original/original/'):
    if files != []:
        for i in files:
            if i.split('.')[-1] == 'JPG':
                fp = os.path.join(r, i)
                # print(fp)
                f1.write('{}\n'.format(fp))
                idx += 1

print(idx)  #1424张
f1.close()



'''
按照文件夹 读取正负样本到各自的列表变量
'''
n_1_path = rootPath+'\整理/n/1'
original_1_path = rootPath+'/original/original\\1'
images = os.listdir(n_1_path)
n_1_imageName = []
for image_name in images:
    image_path = original_1_path + '\\' + image_name
    n_1_imageName.append(image_path)
    # if image_path in original_imglist:
print('1'*20)

n_2_path = rootPath+'\整理/n/2'
original_2_path = rootPath+'/original/original\\2'
images = os.listdir(n_2_path)
n_2_imageName = []
for image_name in images:
    image_path = original_2_path + '\\' + image_name
    n_2_imageName.append(image_path)
    # if image_path in original_imglist:
print('2'*20)

p_100MEDIA_path = rootPath+'\整理/p/100MEDIA'
original_100MEDIA_path = rootPath+'/original/original\\100MEDIA'
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
f2 = open(rootPath+'/original/original/images_lable.txt', 'w')
fid =open(rootPath+'/original/original/images.txt', 'r')
original_imglist = fid.readlines()

for image in original_imglist:
    image = image.strip()
    if image.split('original/')[-1][:2] == '1\\':
        if image in n_1_imageName:
            f2.write('{},,,{}\n'.format(image,0))
        else:
            f2.write('{},,,{}\n'.format(image,1))
    if image.split('original/')[-1][:2] == '2\\':
        if image in n_2_imageName:
            f2.write('{},,,{}\n'.format(image,0))
        else:
            f2.write('{},,,{}\n'.format(image,1))
    if image.split('original/')[-1][:8] == '100MEDIA':
        if image in p_100MEDIA_imageName:
            f2.write('{},,,{}\n'.format(image,1))
        else:
            f2.write('{},,,{}\n'.format(image,0))

f2.close()
fid.close()


'''
处理np_hard_samples_images
'''
# time_start = time.time()

# '''
# 将未整理的原始图像数据保存到images.txt
# '''
# rootPath = 'F:\SoftwareProgram\Data\dataset\snow leopard'
# f1 = open(rootPath+'\original\hard_samples/hard_samples_images.txt', 'w')
# idx = 0
# for r, d, files in os.walk(rootPath+'\original\hard_samples\hard_samples'):
#     if files != []:
#         for i in files:
#                 fp = os.path.join(r, i)
#                 # print(fp)
#                 f1.write('{}\n'.format(fp))
#                 idx += 1
#
# print(idx)  #1424张
# f1.close()

# '''
# 按照文件夹 读取正负样本到各自的列表变量
# '''
# n_hard_samples_path = rootPath+'\整理/n/hard_samples'
# original_hard_samples_path = rootPath+'\\original\\hard_samples\\hard_samples'
# images = os.listdir(n_hard_samples_path)
# n_hard_samples_imageName = []
# for image_name in images:
#     image_path = original_hard_samples_path + '\\' + image_name
#     n_hard_samples_imageName.append(image_path)
#     # if image_path in original_imglist:
# print('1'*20)

# p_hard_samples_path1 = rootPath+'\整理/p/hard_samples'
# # original_hard_samples_path = rootPath+'/original\\hard_samples\\hard_samples'
# images = os.listdir(p_hard_samples_path1)
# p_hard_samples_imageName1 = []
# for image_name in images:
#     image_path = original_hard_samples_path + '\\' + image_name
#     p_hard_samples_imageName1.append(image_path)
#     # if image_path in original_imglist:
# print('2'*20)
#
# p_hard_samples_path2 = rootPath+'\整理/p/hard_samples'
# # original_100MEDIA_path = rootPath+'/original\\100MEDIA'
# images = os.listdir(p_hard_samples_path2)
# p_hard_samples_imageName2 = []
# for image_name in images:
#     image_path = original_hard_samples_path + '\\' + image_name
#     p_hard_samples_imageName2.append(image_path)
#     # if image_path in original_imglist:
# print('3'*20)
#
# '''
# hard_samples_images.txt中的每一行附上标签,hard_samples_images_lable.txt
# '''
# f2 = open(rootPath+'/original\\hard_samples\\hard_samples_images_lable.txt', 'w')
# fid =open(rootPath+'/original\\hard_samples\\hard_samples_images.txt', 'r')
# original_hard_samples_imglist = fid.readlines()
#
# for image in original_hard_samples_imglist:
#     image = image.strip()
#     if image in n_hard_samples_imageName:
#         f2.write('{},,,{}\n'.format(image,0))
#     else:
#         f2.write('{},,,{}\n'.format(image,1))
#
# f2.close()
# fid.close()




# 文件路径

# dataset_len = 844  #其中猫与狗的图片数量各占一半
# # index = np.arange(dataset_len)
# # np.random.shuffle(index)#打断顺序
# # train_len = int(len(index)*0.8)
# # index_train = index[:train_len] #取前len（index）的80%作为index2
# # index_test = index[train_len:]
# rootPath = 'F:\SoftwareProgram\Data\dataset\snow leopard/original/hard_samples'
# f1 = open(rootPath+'\hard_samples_images_lable.txt','r')
# f2 = open(rootPath+'\\hard_samples_train.txt','w')
# f3 = open(rootPath+'\\hard_samples_test.txt','w')
#
# originalImagesList = f1.readlines()
# np.random.shuffle(originalImagesList)
#
# train_len = int(dataset_len*0.8)
# for i in range(train_len):
#     # originalImagesList[i] = originalImagesList[i].strip()
#     f2.write('{}'.format(originalImagesList[i]))
# for i in range(train_len,dataset_len):
#     # originalImagesList[i] = originalImagesList[i].strip()
#     f3.write('{}'.format(originalImagesList[i]))
#
# f1.close()
# f2.close()
# f3.close()
# time_end = time.time()
# print('雪豹训练集和测试集划分完毕, 耗时%s!!' % (time_end - time_start))