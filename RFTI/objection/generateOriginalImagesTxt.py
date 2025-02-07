# -*- coding: utf-8  -*-
import os
from PIL import Image
import cv2


'''
将未整理的原始图像数据保存到images.txt
'''
root = 'F:\SoftwareProgram\Data\dataset\snow leopard/original'
f1 = open('F:\SoftwareProgram\Data\dataset\snow leopard/original/images.txt', 'w')
idx = 0
for r, d, files in os.walk(root):
    if files != []:
        for i in files:
            if i.split('.')[-1] == 'JPG':
                fp = os.path.join(r, i)
                # print(fp)
                f1.write('{}\n'.format(fp))
                idx += 1

print(idx)  #1424张
f1.close()