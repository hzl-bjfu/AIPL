import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn

from nets.facenet import Facenet as facenet

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
    my_model_path = "models_pkl/myCheckpoint/mobilenet_a0.2_f0.5_e100_best.pth"
    my_input_shape = (448, 448, 3)
    my_backbone = "mobilenet"
    model = myFacenet(my_model_path, my_input_shape, my_backbone)
    # model.eval()
    imglistAll = {}
    # eachCamera = []
    # cameraIndex = 0
    # orgDataset_path = 'F:\cfy_mistake_trigger/txt/constrastiveLearn/images_test_label_2n.txt' #不同相机的路径文件，测泛化性
    orgDataset_path = 'F:\cfy_mistake_trigger/txt/constrastiveLearn/images_label_2n_train.txt'  # 编码训练集作为数据库
    with open(orgDataset_path, 'r') as fid_org:
        orgImagePathList = fid_org.readlines()
    for orgImagePath in orgImagePathList:
        imagePath, cameraIndex = orgImagePath.strip('\n').split(',,,')
        # cameraIndex = int(cameraIndex)
        if int(cameraIndex) % 2 == 0:
            if cameraIndex not in imglistAll.keys():
                eachCamera = []
                eachCamera.append(imagePath)
                imglistAll[cameraIndex] = eachCamera
            else:
                imglistAll[cameraIndex].append(imagePath)

    # noAnimalCamera_path = 'F:\cfy_mistake_trigger/txt/constrastiveLearn/noAnimal/'+str(cameraIndex)+'.txt'  #要编码的相机路径文件
    # with open(noAnimalCamera_path, 'r') as fid2:
    #     imglist = fid2.readlines() #imglist是一个列表，为一个相机的所有背景图片路径
    for cameraIndex in imglistAll.keys():
        known_face_encodings = []
        imglist = imglistAll[cameraIndex]  # 一个相机对应一个txt编码文件
        for path in imglist:
            iamge_known = Image.open(path.split(',,,')[0])  # F:\cfy_mistake_trigger\紫合数据双溪口_8\6_1\Ere 0323.JPG,,,48
            image_feature = model.detect_image(iamge_known)
            known_face_encodings.append(image_feature)
            # list_file.write(image_feature+'\n')
        known_face_encodings = np.array(known_face_encodings).squeeze()
        # np.savetxt('F:\cfy_mistake_trigger/txt/constrastiveLearn/test/noAnimal/' + cameraIndex + '.txt',
        #            known_face_encodings)
        np.savetxt('F:\cfy_mistake_trigger/txt/constrastiveLearn/train/noAnimal/' + cameraIndex + '.txt',
                   known_face_encodings)
        # known_face_encodings = np.loadtxt('F:\cfy_mistake_trigger\北山数据_22/004\known_face_encodings.txt')

if __name__ == "__main__":
    main()





