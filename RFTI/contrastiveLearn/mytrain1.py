import os
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision.models import efficientnet_b0
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

# from nets.facenet import Facenet
from nets.my_facenet import Facenet
# from nets.facenet_training import triplet_loss, LossHistory, weights_init
from nets.my_facenet_training import triplet_loss, LossHistory, weights_init
# from utils.dataloader import FacenetDataset, dataset_collate
from utils.my_dataloader import FacenetDataset, dataset_collate
from utils.eval_metrics import evaluate
from utils.LFWdataset import LFWDataset
from utils.myEarlyStopping import EarlyStopping

from mytest import face_distance,compare_faces,myFacenet
from PIL import Image

import datetime



def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def letterbox_image(image, input_shape):
    image = image.convert("RGB")
    iw, ih = image.size
    w, h = [input_shape[1], input_shape[0]]
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', [input_shape[1], input_shape[0]], (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    if input_shape[-1] == 1:
        new_image = new_image.convert("L")
    return new_image

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
def detect_image(model, image_1,input_shape,cuda):
    # ---------------------------------------------------#
    #   图片预处理，归一化
    # ---------------------------------------------------#
    with torch.no_grad():
        image_1 = letterbox_image(image_1, [input_shape[1], input_shape[0]])
        # image_2 = self.letterbox_image(image_2, [self.input_shape[1], self.input_shape[0]])

        photo_1 = torch.from_numpy(
            np.expand_dims(np.transpose(np.asarray(image_1).astype(np.float64) / 255, (2, 0, 1)), 0)).type(
            torch.FloatTensor)
        # photo_2 = torch.from_numpy(
        #     np.expand_dims(np.transpose(np.asarray(image_2).astype(np.float64) / 255, (2, 0, 1)), 0)).type(
        #     torch.FloatTensor)

        if cuda:
            photo_1 = photo_1.cuda()
            # photo_2 = photo_2.cuda()

        # ---------------------------------------------------#
        #   图片传入网络进行预测
        # ---------------------------------------------------#
        # output1 = net(photo_1).cpu().numpy()
        output1 = model(photo_1).cpu().numpy()
        # output2 = self.net(photo_2).cpu().numpy()
        #
        # # ---------------------------------------------------#
        # #   计算二者之间的距离
        # # ---------------------------------------------------#
        # l1 = np.linalg.norm(output1 - output2, axis=1)


    return output1

def encoding2(model,trainImagePath,cuda):
    # my_model_path = "models_pkl/myCheckpoint/mobilenet_a0.2_f0.5_e100_best.pth"
    # my_input_shape = (448, 448, 3)
    # my_backbone = "mobilenet"
    # model = myFacenet(my_model_path, my_input_shape, my_backbone)
    # model.eval()
    # net.eval()

    imglistAll = {}
    # eachCamera = []
    # cameraIndex = 0
    # orgDataset_path = 'F:\cfy_mistake_trigger/txt/constrastiveLearn/images_test_label_2n.txt'
    with open(trainImagePath, 'r') as fid_org:#用训练集的相机背景作为对比数据库
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
            image_feature = detect_image(model,iamge_known,input_shape,cuda)
            known_face_encodings.append(image_feature)
            # list_file.write(image_feature+'\n')
        known_face_encodings = np.array(known_face_encodings).squeeze()
        np.savetxt('F:\cfy_mistake_trigger/txt/constrastiveLearn/test/noAnimal/' + cameraIndex + '.txt',
                   known_face_encodings)
        # known_face_encodings = np.loadtxt('F:\cfy_mistake_trigger\北山数据_22/004\known_face_encodings.txt')
        # return known_face_encodings

def test2(model,trainImagePath,testImagePath,facenet_threhold,e,time_str,cuda):
    # model = myFacenet(my_model_path, my_input_shape, my_backbone)
    # model.eval()
    net.eval()
    preWrong_num = 0
    haveAniTotal_num = 0
    noAniTotal_num = 0
    haveAniWrong_num = 0
    noAniWrong_num = 0
    # testImagePath = 'F:\cfy_mistake_trigger/txt/constrastiveLearn/images_test_label_2n.txt'  # 要测试的相机路径文件

    encoding2(model, trainImagePath,cuda)

    test_negative = open('./data/test/test_negative_'+str(time_str)+'.txt', 'a') # 只追写文件。从文件底部添加内容 不存在则创建 。
    test_negative.write('{}:{},,,'.format("epoch", str(e)))

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

    # test_negative_haveAniWrong = open('./data/test/test_negative_haveAniWrong_' + str(facenet_threhold) + '.txt',
    #                                   'w')
    # test_negative_noAniWrong = open('./data/test/test_negative_noAniWrong_' + str(facenet_threhold) + '.txt', 'w')
    for cameraIndex in imglistAll.keys():
        known_face_encodings = np.loadtxt('F:\cfy_mistake_trigger/txt/constrastiveLearn/test/noAnimal/' + str(
            (int(cameraIndex) // 2) * 2) + '.txt')
        imglist = imglistAll[cameraIndex]
        for path in imglist:
            iamge_pre = Image.open(path)  # F:\cfy_mistake_trigger\紫合数据双溪口_8\6_1\Ere 0323.JPG
            face_encoding = detect_image(model,iamge_pre,input_shape,cuda)
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
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] <= facenet_threhold:
                name = "no_animal"
            if int(cameraIndex) % 2 == 0:
                noAniTotal_num += 1
                if name == "have_animals":
                    preWrong_num += 1
                    noAniWrong_num += 1
                    # print(path, " ", name)
                    # print(face_distances[best_match_index], face_distances)
                    # test_negative_noAniWrong.write('{},,,{}\n'.format(path, cameraIndex))
                    # test_negative_noAniWrong.write(
                    #     '{},,,{}\n'.format(face_distances[best_match_index], face_distances))
            if int(cameraIndex) % 2 == 1:
                haveAniTotal_num += 1
                if name == "no_animal":
                    preWrong_num += 1
                    haveAniWrong_num += 1
                    # print(path, " ", name)
                    # print(face_distances[best_match_index], face_distances)
                    # test_negative_haveAniWrong.write('{},,,{}\n'.format(path, cameraIndex))
                    # test_negative_haveAniWrong.write(
                    #     '{},,,{}\n'.format(face_distances[best_match_index], face_distances))
    # test_negative.write('{} {}\n'.format("相似度阈值:", facenet_threhold))
    test_negative.write('{}  {},,,'.format("识别错误的数量:" + str(preWrong_num) + "/" + str(testImageLen),
          " acc:" + str(1 - (preWrong_num / testImageLen))))
    if haveAniTotal_num != 0:
        test_negative.write('{}  {},,,'.format("有动物图像识别错误的数量:" + str(haveAniWrong_num) + "/" + str(haveAniTotal_num),
              " acc:" + str(1 - (haveAniWrong_num / haveAniTotal_num))))
    else:
        test_negative.write('{},,,'.format("有动物图像识别错误的数量:" + str(haveAniWrong_num) + "/" + str(haveAniTotal_num)))
    if noAniTotal_num != 0:
        test_negative.write('{}  {}\n'.format("无动物图像识别错误的数量:" + str(noAniWrong_num) + "/" + str(noAniTotal_num),
              " acc:" + str(1 - (noAniWrong_num / noAniTotal_num))))
    else:
        test_negative.write('{}\n'.format("无动物图像识别错误的数量:" + str(noAniWrong_num) + "/" + str(noAniTotal_num)))

    return 1 - (preWrong_num / testImageLen)


# def fit_ont_epoch(model, loss, epoch, epoch_size, gen, val_epoch_size, gen_val, Epoch, test_loader, cuda):
def fit_ont_epoch(model, loss, epoch, epoch_size,trainImagePath,testImagePath,facenet_threhold, gen,  Epoch, cuda):

    total_triple_loss = 0
    total_CE_loss = 0
    total_accuracy = 0

    test_total_triple_loss = 0
    test_total_CE_loss = 0
    test_total_accuracy = 0

    net.train()
    # model.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen): #iteration表示第几个batch
            if iteration >= epoch_size:
                break
            images, labels = batch
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    labels = Variable(torch.from_numpy(labels).long()).cuda()
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    labels = Variable(torch.from_numpy(labels).long())

            optimizer.zero_grad()
            before_normalize, outputs1 = model.forward_feature(images)
            outputs2 = model.forward_classifier(before_normalize)

            _triplet_loss = loss(outputs1, Batch_size)
            _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
            _loss = _triplet_loss + _CE_loss

            _loss.backward()
            optimizer.step()

            with torch.no_grad():
                accuracy = torch.mean(
                    (torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

            total_accuracy += accuracy.item()
            total_triple_loss += _triplet_loss.item()
            total_CE_loss += _CE_loss.item()

            pbar.set_postfix(**{'total_triple_loss': total_triple_loss / (iteration + 1),
                                'total_CE_loss': total_CE_loss / (iteration + 1),
                                'accuracy': total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    # net.eval()
    # model.eval()
    print('Start Validation')
    test_acc = test2(model,trainImagePath,testImagePath, facenet_threhold, epoch + 1, time_str,cuda)
    print("测试集准确率:", test_acc)
    print('Finish Validation')


    loss_history.append_loss(np.mean(accuracy.numpy()), (total_triple_loss + total_CE_loss) / (epoch_size + 1))

    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f' % ((total_triple_loss + total_CE_loss) / (epoch_size + 1)))
    print('Saving state, iter:', str(epoch + 1))
    # torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f.pth-Val_Loss%.4f.pth' % ((epoch + 1),
    #                                                                                      (
    #                                                                                              total_triple_loss + total_CE_loss) / (
    #                                                                                              epoch_size + 1),
    #                                                                                      (
    #                                                                                              val_total_triple_loss + val_total_CE_loss) / (
    #                                                                                              val_epoch_size + 1)))
    if_early_stopping = False
    early_stopping(test_acc, model)
    if early_stopping.early_stop:
        if_early_stopping = True
    # else:
    #     torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Test_acc%.4f.pth' % (
    #     (epoch + 1), (total_triple_loss + total_CE_loss) / (epoch_size + 1), test_acc))

    # model_name = 'logs/Epoch%d-Total_Loss%.4f-Test_acc%.4f.pth' % ((epoch + 1),(total_triple_loss + total_CE_loss) / (epoch_size + 1),test_acc)
    # return (val_total_triple_loss + val_total_CE_loss) / (val_epoch_size + 1)
    return (total_triple_loss + total_CE_loss) / (epoch_size + 1) ,if_early_stopping #cfy



if __name__ == "__main__":
    log_dir = "./logs/"
    # annotation_path = "./cls_train.txt"
    # annotation_path = "F:\cfy_mistake_trigger/txt\constrastiveLearn/images_label_2n.txt"
    # testImagePath = "F:\cfy_mistake_trigger/txt\constrastiveLearn/images_test_label_2n.txt"
    trainImagePath = "F:\cfy_mistake_trigger/txt\constrastiveLearn/images_label_2n_train.txt"
    testImagePath = "F:\cfy_mistake_trigger/txt\constrastiveLearn/images_label_2n_test.txt"
    # num_classes = get_num_classes(annotation_path) #16
    num_classes = 25 #相机数量
    # num_classes_test = 9 #相机数量
    # facenet_threhold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alpha = 0.2
    facenet_threhold = 0.5
    curr_time = datetime.datetime.now()
    # time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    time_str = datetime.datetime.strftime(curr_time, '%m_%d_%H_%M_%S')
    # test_save_path = os.path.join('.data/test/', str(time_str))
    # self.acc = []
    # self.losses = []
    # self.val_loss = []

    # os.makedirs(test_save_path)


    # --------------------------------------#
    #   输入图片大小
    #   可选112,112,3
    # --------------------------------------#
    # input_shape = [112,112,3]
    input_shape = [448, 448, 3]
    # --------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet
    #   inception_resnetv1
    # --------------------------------------#
    # backbone = "mobilenet"
    # backbone = "inception_resnetv1"
    # backbone = "efficientNet-b0"
    # backbone = "ResNet50" #如果用resnet或resnest时，在定义模型时已经加载了预训练模型，下面不需要加载模型
    backbone = "ResNeSt50"
    # --------------------------------------#
    #   Cuda的使用
    # --------------------------------------#
    Cuda = True

    # model = Facenet(num_classes=num_classes, backbone=backbone)
    model = Facenet(num_classes=2*num_classes, backbone=backbone) #cfy,num_classes*2为所有标签数量，计算CEloss时使用
    weights_init(model)
    # -------------------------------------------#
    #   权值文件的下载请看README
    #   权值和主干特征提取网络一定要对应
    # -------------------------------------------#

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # 加快模型训练的效率  #如果用resnet或resnest时，在定义模型时已经加载了预训练模型，这里不需要加载模型
    # print('Loading weights into state dict...')
    # # 加载权重文件
    # # model_path = "model_pkl/facenet_inception_resnetv1.pth"
    # model_path = "models_pkl/facenet_mobilenet.pth"
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path, map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    # # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict) #如果用resnet或resnest时，在定义模型时已经加载了预训练模型，这里不需要加载模型

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    loss = triplet_loss(alpha=alpha)
    loss_history = LossHistory(log_dir)

    # LFW_loader = torch.utils.data.DataLoader(
    #     LFWDataset(dir="lfw/", pairs_path="model_data/lfw_pair.txt", image_size=input_shape), batch_size=32,
    #     shuffle=False)

    # -------------------------------------------------------#
    #   0.05用于验证，0.95用于训练
    # -------------------------------------------------------#
    # val_split = 0.05
    with open(trainImagePath, "r") as f:
        lines = f.readlines()
    # with open(annotation_path_test, "r") as f_test:
    #     lines_test = f_test.readlines()
    # np.random.seed(10101)
    # np.random.shuffle(lines)
    # np.random.seed(None)
    # num_val = int(len(lines) * val_split)
    # num_train = len(lines) - num_val
    num_train = len(lines)
    # num_test = len(lines_test)

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        lr = 1e-3
        # Batch_size = 64
        Batch_size = 4
        Init_Epoch = 0
        Interval_Epoch = 50

        optimizer = optim.Adam(net.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4, verbose=True)

        train_dataset = FacenetDataset(input_shape, lines[:num_train], num_train, num_classes)
        # val_dataset = FacenetDataset(input_shape, lines[num_train:], num_val, num_classes)

        # gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
        #                  drop_last=True, collate_fn=dataset_collate)
        # gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
        #                      drop_last=True, collate_fn=dataset_collate)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate)
        # gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,
        #                      drop_last=True, collate_fn=dataset_collate)

        epoch_size = max(1, num_train // Batch_size)
        # val_epoch_size = max(1, num_val // Batch_size)

        for param in model.backbone.parameters():
            param.requires_grad = False
            # param.requires_grad = True

        patience = 20
        model_name = './logs/checkpoint'+str(facenet_threhold)+'train1_'+backbone+"_"+str(time_str)+'.pth'
        early_stopping = EarlyStopping(patience=patience, verbose=True,model_name = model_name)
        Interval_Epoch_temp = Interval_Epoch
        for epoch in range(Init_Epoch, Interval_Epoch_temp):
            Interval_Epoch = epoch+1
            # _loss = fit_ont_epoch(model, loss, epoch, epoch_size, gen, val_epoch_size, gen_val, Interval_Epoch,
            #                       LFW_loader, Cuda)
            _loss,if_early_stopping = fit_ont_epoch(model,loss, epoch, epoch_size,trainImagePath,testImagePath,facenet_threhold, gen,  Interval_Epoch_temp,
                                Cuda)
            lr_scheduler.step(_loss)

            if if_early_stopping:
                print("Early stopping:",epoch+1)
                break
        model.load_state_dict(torch.load(model_name))


    if True:
        lr = 1e-4
        # Batch_size = 32
        Batch_size = 1
        # Interval_Epoch = 0 #若要测试batch_size  ,令Interval_Epoch=0
        Epoch = Interval_Epoch+50
        # Epoch = 100

        optimizer = optim.Adam(net.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4, verbose=True)

        train_dataset = FacenetDataset(input_shape, lines[:num_train], num_train, num_classes)
        # val_dataset = FacenetDataset(input_shape, lines[num_train:], num_val, num_classes)

        # gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
        #                  drop_last=True, collate_fn=dataset_collate)
        # gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
        #                      drop_last=True, collate_fn=dataset_collate)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate)
        # gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=0, pin_memory=True,
        #                      drop_last=True, collate_fn=dataset_collate)
        epoch_size = max(1, num_train // Batch_size)
        # val_epoch_size = max(1, num_val // Batch_size)

        for param in model.backbone.parameters():
            param.requires_grad = True

        patience = 20
        model_name = './logs/checkpoint' + str(facenet_threhold) +'train1_'+backbone+"_"+str(time_str)+ '.pth'
        early_stopping = EarlyStopping(patience=patience, verbose=True, model_name=model_name)
        for epoch in range(Interval_Epoch, Epoch):
            # _loss = fit_ont_epoch(model, loss, epoch, epoch_size, gen, val_epoch_size, gen_val, Epoch, LFW_loader, Cuda)
            _loss,if_early_stopping = fit_ont_epoch(model,loss, epoch, epoch_size,trainImagePath,testImagePath,facenet_threhold, gen,  Epoch, Cuda)
            lr_scheduler.step(_loss)
            if if_early_stopping:
                print("Early stopping:",epoch+1)
                break
        # model.load_state_dict(torch.load(model_name))
