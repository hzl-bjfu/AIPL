import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn

from nets.facenet import Facenet as facenet
from torchvision.models import efficientnet_b0


#---------------------------------#
#   计算人脸距离
#---------------------------------#
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    # (n, )
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

#---------------------------------#
#   比较人脸
#---------------------------------#
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=1):
    dis = face_distance(known_face_encodings, face_encoding_to_check)  #欧氏距离
    return list(dis <= tolerance), dis


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和backbone需要修改！
# --------------------------------------------#
class myFacenet(object):
    _defaults = {
        # "model_path"    : "model_data/facenet_mobilenet.pth",
        # "model_path": my_model_path,
        # # "input_shape"   : (160, 160, 3),
        # "input_shape": my_input_shape,
        # "backbone": my_backbone,
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化myFacenet
    # ---------------------------------------------------#
    def __init__(self, my_model_path, my_input_shape, my_backbone, **kwargs):
        self.__dict__.update(self._defaults)
        self.model_path = my_model_path
        self.input_shape = my_input_shape
        self.backbone = my_backbone
        self.generate()


    def generate(self):
        # 载入模型
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = facenet(backbone=self.backbone, mode="predict")
        model.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net = model.eval()

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

        print('{} model loaded.'.format(self.model_path))

    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        if self.input_shape[-1] == 1:
            new_image = new_image.convert("L")
        return new_image

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_1):
        # ---------------------------------------------------#
        #   图片预处理，归一化
        # ---------------------------------------------------#
        with torch.no_grad():
            image_1 = self.letterbox_image(image_1, [self.input_shape[1], self.input_shape[0]])
            # image_2 = self.letterbox_image(image_2, [self.input_shape[1], self.input_shape[0]])

            photo_1 = torch.from_numpy(
                np.expand_dims(np.transpose(np.asarray(image_1).astype(np.float64) / 255, (2, 0, 1)), 0)).type(
                torch.FloatTensor)
            # photo_2 = torch.from_numpy(
            #     np.expand_dims(np.transpose(np.asarray(image_2).astype(np.float64) / 255, (2, 0, 1)), 0)).type(
            #     torch.FloatTensor)

            if self.cuda:
                photo_1 = photo_1.cuda()
                # photo_2 = photo_2.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            output1 = self.net(photo_1).cpu().numpy()
            # output2 = self.net(photo_2).cpu().numpy()
            #
            # # ---------------------------------------------------#
            # #   计算二者之间的距离
            # # ---------------------------------------------------#
            # l1 = np.linalg.norm(output1 - output2, axis=1)


        return output1

def main():
    # facenet_threhold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    facenet_threhold_list = [0.5]
    my_model_path = "models_pkl/myCheckpoint/mobilenet_a0.2_f0.5_e100_best.pth"
    my_input_shape = (448, 448, 3)
    # 可以选择使用 efficientnet_b0 作为骨干网络
    my_backbone = "efficientnet_b0"
    model = myFacenet(my_model_path, my_input_shape, my_backbone)
    # model.eval()
    for facenet_threhold in facenet_threhold_list:
        preWrong_num = 0
        haveAniTotal_num = 0
        noAniTotal_num = 0
        haveAniWrong_num = 0
        noAniWrong_num = 0
        # testImagePath = 'F:\cfy_mistake_trigger/txt/constrastiveLearn/images_test_label_2n.txt'  # 要测试的相机路径文件
        testImagePath = 'F:\cfy_mistake_trigger/txt/constrastiveLearn/images_label_2n_test.txt'  # 要测试的相机路径文件

        imglistAll = {}
        with open(testImagePath, 'r') as fid_org:
            testImagePathList = fid_org.readlines()
        testImageLen = len(testImagePathList)
        for testImagePath in testImagePathList:
            imagePath, cameraIndex = testImagePath.strip('\n').split(',,,')
            # cameraIndex = int(cameraIndex)
            # if int(cameraIndex)%2==0:
            if cameraIndex not in imglistAll.keys():
                eachCamera = []
                eachCamera.append(imagePath)
                imglistAll[cameraIndex] = eachCamera
            else:
                imglistAll[cameraIndex].append(imagePath)

        test_negative_haveAniWrong = open('./data/test/test_negative_haveAniWrong_' + str(facenet_threhold) + '.txt',
                                          'w')
        test_negative_noAniWrong = open('./data/test/test_negative_noAniWrong_' + str(facenet_threhold) + '.txt', 'w')
        for cameraIndex in imglistAll.keys():
            # known_face_encodings = np.loadtxt('F:\cfy_mistake_trigger/txt/constrastiveLearn/test/noAnimal/' + str(
            #     (int(cameraIndex) // 2) * 2) + '.txt')
            known_face_encodings = np.loadtxt('F:\cfy_mistake_trigger/txt/constrastiveLearn/train/noAnimal/' + str(
                (int(cameraIndex) // 2) * 2) + '.txt')
            imglist = imglistAll[cameraIndex]
            for path in imglist:
                iamge_pre = Image.open(path)  # F:\cfy_mistake_trigger\紫合数据双溪口_8\6_1\Ere 0323.JPG
                face_encoding = model.detect_image(iamge_pre)
                # -----------------------------------------------#
                #   特征比对-开始
                # -----------------------------------------------#
                matches, face_distances = compare_faces(known_face_encodings, face_encoding, tolerance=facenet_threhold)

                name = "have_animals"
                # 取出这个最近人脸的评分
                # 取出当前输入进来的人脸，最接近的已知人脸的序号
                if int(cameraIndex) % 2 == 0:
                    face_distances = np.delete(face_distances,
                                               np.argwhere(np.array(face_distances) == 0))  # 删除出现相似度阈值为0的图像（相同图像）
                    # best_match_index = np.argmin(face_distances)
                    # face_distances = np.delete(face_distances, best_match_index)
                try:
                    best_match_index = np.argmin(face_distances)
                except:
                    print(path)
                if face_distances[best_match_index] <= facenet_threhold:
                    name = "no_animal"
                if int(cameraIndex) % 2 == 0:
                    noAniTotal_num += 1
                    if name == "have_animals":
                        preWrong_num += 1
                        noAniWrong_num += 1
                        # print(path, " ", name)
                        # print(face_distances[best_match_index], face_distances)
                        test_negative_noAniWrong.write('{},,,{}\n'.format(path, cameraIndex))
                        test_negative_noAniWrong.write(
                            '{},,,{}\n'.format(face_distances[best_match_index], face_distances))
                if int(cameraIndex) % 2 == 1:
                    haveAniTotal_num += 1
                    if name == "no_animal":
                        preWrong_num += 1
                        haveAniWrong_num += 1
                        # print(path, " ", name)
                        # print(face_distances[best_match_index], face_distances)
                        test_negative_haveAniWrong.write('{},,,{}\n'.format(path, cameraIndex))
                        test_negative_haveAniWrong.write(
                            '{},,,{}\n'.format(face_distances[best_match_index], face_distances))
        print("相似度阈值:", facenet_threhold)
        print("识别错误的数量:" + str(preWrong_num) + "/" + str(testImageLen),
              " acc:" + str(1 - (preWrong_num / testImageLen)))
        if haveAniTotal_num != 0:
            print("有动物图像识别错误的数量:" + str(haveAniWrong_num) + "/" + str(haveAniTotal_num),
                  " acc:" + str(1 - (haveAniWrong_num / haveAniTotal_num)))
        else:
            print("有动物图像识别错误的数量:" + str(haveAniWrong_num) + "/" + str(haveAniTotal_num))
        if noAniTotal_num != 0:
            print("无动物图像识别错误的数量:" + str(noAniWrong_num) + "/" + str(noAniTotal_num),
                  " acc:" + str(1 - (noAniWrong_num / noAniTotal_num)))
        else:
            print("无动物图像识别错误的数量:" + str(noAniWrong_num) + "/" + str(noAniTotal_num))
        test_negative_haveAniWrong.close()
        test_negative_noAniWrong.close()


if __name__ == "__main__":
    main()