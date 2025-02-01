import PIL.Image as Image
import os
from torchvision import transforms as transforms
import time



outfile = 'F:\cfy_mistake_trigger/train\北山数据_22/102\have_animals'
im = Image.open('F:\cfy_mistake_trigger/train\北山数据_22/102\have_animals/PICT0105.JPG')
# outfile = 'F:\cfy_mistake_trigger/train\snow_leopard_12/2_1'
# im = Image.open('F:\cfy_mistake_trigger/train\snow_leopard_12/2_1/JDG-A099B-0728.JPG')
# im.save(os.path.join(outfile, 'test2.jpg'))
# print(os.path.join(outfile, 'test2.jpg'))

for i in range(1,10):
    a = 1.5
    new_im = transforms.ColorJitter(brightness=(0,a))(im)  #亮度，当为a时，从[max(0,1-a),1+a]中随机选择。当为（ab）时，从[a,b]中选择。
    new_im = transforms.ColorJitter(contrast=[0,a])(new_im)   #对比度，同上
    new_im = transforms.ColorJitter(saturation=[0,a])(new_im)  #饱和度，同上
    new_im = transforms.ColorJitter(hue=0.5)(new_im)    #色度，当为ａ时，从[-a,a]中选择参数，注：0<=a<=0.5
    new_im.save(os.path.join(outfile, 'PICT0105_'+str(i)+'.JPG'))

# new_im = transforms.RandomGrayscale(p=0.5)(im)    # 以0.5的概率进行灰度化
# new_im.save(os.path.join(outfile, '灰度化.JPG'))


# new_im = transforms.Resize((100, 200))(im)  #高100，宽200
# print(f'{im.size}---->{new_im.size}')
# new_im.save(os.path.join(outfile, '1_1.JPG'))

# new_im = transforms.RandomCrop(3000)(im)   # 裁剪出100x100的区域
# new_im.save(os.path.join(outfile, 'RandomCrop_1.JPG'))
# new_im = transforms.CenterCrop(5000)(im)
# new_im.save(os.path.join(outfile, 'CenterCrop_2.JPG'))

# new_im = transforms.RandomHorizontalFlip(p=0.5)(im)   # p表示概率
# new_im.save(os.path.join(outfile, '随机水平翻转_1.JPG'))
# new_im = transforms.RandomVerticalFlip(p=1)(im)
# new_im.save(os.path.join(outfile, '随机垂直翻转_2.JPG'))


# new_im = transforms.RandomRotation(90)(im)    #随机旋转，最大正负90度
# new_im.save(os.path.join(outfile, '随机旋转.JPG'))

