import os
import numpy as np


# rootPath = 'F:\cfy_mistake_trigger'


# '''
# images_allCamera
# '''
# idx = 0
# f1 = open(rootPath+'/images.txt', 'w')
# for r, d, files in os.walk(rootPath):
#     if files != []:
#         for i in files:
#             if i.split('.')[-1] == 'JPG':
#                 fp = os.path.join(r, i)
#                 # print(fp)
#                 f1.write('{}\n'.format(fp))
#                 idx += 1
#
# print(idx)  #1004张
# f1.close()

# '''
# 将images.txt中的每一行附上标签,生成images_label.txt
# '''
# idx = 0
# f2 = open(rootPath+'/images_label.txt', 'w')
# fid =open(rootPath+'/images.txt', 'r')
# original_imglist = fid.readlines()
#
# for image in original_imglist:
#     image = image.strip()
#     if 'have_animals' in image:
#         f2.write('{},,,{}\n'.format(image,0))
#     else:
#         f2.write('{},,,{}\n'.format(image,1))
#     idx += 1
#
# print(idx)  #1004张
# f2.close()
# fid.close()









# '''
# images_eachCamera
# '''
# cameraNumber = '089'
# idx = 0
# f1 = open(rootPath+'/txt/images_beishanshuju'+ cameraNumber +'.txt', 'w')
# for r, d, files in os.walk(rootPath+'/txt/'):
#     if files != []:
#         for i in files:
#             if i.split('.')[-1] == 'JPG':
#                 fp = os.path.join(r, i)
#                 # print(fp)
#                 f1.write('{}\n'.format(fp))
#                 idx += 1
#
# print(idx)  #1004张
# f1.close()
#
# '''
# 将images.txt中的每一行附上标签,生成images_label.txt
# '''
# idx = 0
# f2 = open(rootPath+'/txt/images_label_beishanshuju'+ cameraNumber +'.txt', 'w')
# fid =open(rootPath+'/txt/images_beishanshuju'+ cameraNumber +'.txt', 'r')
# original_imglist = fid.readlines()
#
# for image in original_imglist:
#     image = image.strip()
#     if 'have_animals' in image:
#         f2.write('{},,,{}\n'.format(image,0))
#     else:
#         f2.write('{},,,{}\n'.format(image,1))
#     idx += 1
#
# print(idx)  #1004张
# f2.close()
# fid.close()




# '''
# images_eachCamera_test
# '''
# cameraNumber = '004'
# # rootPath = 'F:\cfy_mistake_trigger/txt\eachCamera'
# f1 = open(rootPath+'/txt/eachCamera/images_test_beishanshuju'+ cameraNumber +'.txt', 'w')
# for r, d, files in os.walk(rootPath+'/txt/eachCamera/'):
#     if files != []:
#         for i in files:
#             if i.split('.')[-1] == 'JPG':
#                 fp = os.path.join(r, i)
#                 # print(fp)
#                 f1.write('{}\n'.format(fp))
#                 idx += 1
#
# print(idx)  #1004张
# f1.close()
#
# '''
# 将images.txt中的每一行附上标签,生成images_label.txt
# '''
# idx = 0
# f2 = open(rootPath+'/txt/eachCamera/images_label_test_beishanshuju'+ cameraNumber +'.txt', 'w')
# fid =open(rootPath+'/txt/eachCamera/images_test_beishanshuju'+ cameraNumber +'.txt', 'r')
# original_imglist = fid.readlines()
#
# for image in original_imglist:
#     image = image.strip()
#     if 'have_animals' in image:
#         f2.write('{},,,{}\n'.format(image,0))
#     else:
#         f2.write('{},,,{}\n'.format(image,1))
#     idx += 1
#
# print(idx)  #1004张
# f2.close()
# fid.close()



# rootPath = 'F:\cfy_mistake_trigger'
#
# '''
# images_allCamera_2n
# '''
# # a = 10
# # b = 3
# # c = 11
# # d = 1
# #25
# idx = 0
# cameraNumber = 0
# # f1 = open(rootPath+'/txt/constrastiveLearn/images_test_label_2n.txt', 'w')
# f1 = open(rootPath+'/txt/constrastiveLearn/images_label_2n.txt', 'w')
# for r, d, files in os.walk(rootPath+'/train'):
#     # if r == 'F:\cfy_mistake_trigger/test':
#     #     # del d[1]
#     #     d.remove('txt')
#     # if 'images_label_beishanshuju004.txt' in files:
#     #     files.remove('images_label_beishanshuju004.txt')
#     if files != []:
#
#         for i in files:
#             if i.split('.')[-1] == 'JPG' or i.split('.')[-1] == 'jpg':
#                 fp = os.path.join(r, i)
#                 # print(fp)
#                 f1.write('{},,,{}\n'.format(fp,cameraNumber))
#                 idx += 1
#         cameraNumber += 1
# print(idx)  #1014张
# f1.close()


rootPath = 'F:\cfy_mistake_trigger'
'''
images_allCamera_2n_train_test
'''
f1 = open(rootPath+'/txt/constrastiveLearn/images_label_2n.txt', 'r')
f2 = open(rootPath+'/txt/constrastiveLearn/images_label_2n_train.txt','w')
f3 = open(rootPath+'/txt/constrastiveLearn/images_label_2n_test.txt','w')


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
