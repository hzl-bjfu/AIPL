import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import shutil

# from nets.facenet import Facenet as facenet
from nets.my_facenet import Facenet as facenet



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
        # plt.imshow(new_image)
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
    facenet_threhold = 0.5
    my_model_path = "models_pkl/myCheckpoint/mobilenet_a0.2_f0.5_e100_best.pth" #mobilenet
    # my_model_path = "logs/checkpoint0.6_train1_inception_resnetv1_08_29_15_40_14.pth" #inception_resnetv1
    # my_model_path = "logs/checkpoint0.6_train1_ResNet50_08_31_10_48_17.pth" #ResNet50
    my_input_shape = (448, 448, 3)
    my_backbone = "mobilenet"
    # my_backbone = "inception_resnetv1"
    # my_backbone = "ResNet50"
    # my_backbone = "ResNeSt50"
    model = myFacenet(my_model_path, my_input_shape, my_backbone)
    # model.eval()
    known_face_encodings = np.loadtxt('F:\cfy_mistake_trigger/txt\constrastiveLearn/train/noAnimal/26.txt') #选择相机背景的编码
    testImagePath = 'F:\cfy_mistake_trigger/txt\constrastiveLearn/images_oneCamera.txt'  # 待测试的相机文件路径（相机所有图像）
    cameraIndex = '001'
    resImagePath = './data/test_oneCamera/'+my_backbone+'/'+cameraIndex
    with open(testImagePath, 'r') as fid_org:
        testImagePathList = fid_org.readlines()
    for testImagePath in testImagePathList:
        imagePath = testImagePath.strip('\n')
        imagePath = "F:\cfy_all_trigger\北山数据_22/001/" + imagePath.split('001/')[-1]  # cfy,移动硬盘路径
        image_1 = Image.open(imagePath)
        face_encoding = model.detect_image(image_1)
        matches, face_distances = compare_faces(known_face_encodings, face_encoding, tolerance=facenet_threhold)
        # 取出这个最近人脸的评分
        # 取出当前输入进来的人脸，最接近的已知人脸的序号
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] <= facenet_threhold:
            shutil.copy(imagePath, resImagePath+'/no_animal/')
        else:
            shutil.copy(imagePath, resImagePath+'/have_animals/')
        # print(image_1, " ", name)
        # print(face_distances[best_match_index], face_distances)

if __name__ == "__main__": #预测单张
    main()

