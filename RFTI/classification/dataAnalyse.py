import json
import time



'''
统计各相机预测错误的数量分布
'''
f1 = open('./data/allCamera/test_negative_allCamera_focal_50_0.5_1000.txt','r')
testImagesList = f1.readlines()
cameraList = {}
for imageName in testImagesList:
    imageName = imageName.strip()
    cameraNumber = imageName.split('\\')[3]
    if cameraNumber in cameraList:
        cameraList[cameraNumber] += 1
    else:
        cameraList[cameraNumber] = 1

import matplotlib.pyplot as plt
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# waters = ('碳酸饮料', '绿茶', '矿泉水', '果汁', '其他')
# buy_number = [6, 7, 6, 1, 2]
print(cameraList.keys())
print(cameraList.values())
categoryKey=list(cameraList.keys())
categoryValue=list(cameraList.values())
plt.bar(categoryKey,categoryValue)
plt.title('各相机预测错误的数量分布')
plt.savefig('./data/allCamera/各相机预测错误的数量分布.jpg')
plt.show()