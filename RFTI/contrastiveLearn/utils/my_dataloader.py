import random

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


class FacenetDataset(Dataset):
    def __init__(self, input_shape, dataset_path, num_train, num_classes):
        super(FacenetDataset, self).__init__()

        self.dataset_path = dataset_path

        self.image_height = input_shape[0]
        self.image_width = input_shape[1]
        self.channel = input_shape[2]

        self.paths = []
        self.labels = []

        self.num_train = num_train

        self.num_classes = num_classes
        self.load_dataset()

    def __len__(self):
        return self.num_train

    def load_dataset(self):
        for path in self.dataset_path:
            # path_split = path.split(";")
            path_split = path.split(",,,")
            # self.paths.append(path_split[1].split()[0])
            self.paths.append(path_split[0])
            self.labels.append(int(path_split[1].split('\n')[0])) # cfy，根据我的路径，不要删掉空格
        self.paths = np.array(self.paths, dtype=np.object)
        self.labels = np.array(self.labels)

    def get_random_data(self, image, input_shape, jitter=.1, hue=.05, sat=1.3, val=1.3, flip_signal=True):
        image = image.convert("RGB")

        h, w = input_shape
        rand_jit1 = rand(1 - jitter, 1 + jitter)
        rand_jit2 = rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        scale = rand(0.9, 1.1)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        flip = rand() < .5
        if flip and flip_signal:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        rotate = rand() < .5
        if rotate:
            angle = np.random.randint(-5, 5)
            a, b = w / 2, h / 2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])

        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        if self.channel == 1:
            image_data = Image.fromarray(np.uint8(image_data)).convert("L")
        # cv2.imshow("TEST",np.uint8(cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)))
        # cv2.waitKey(0)
        return image_data

    def __getitem__(self, index):
        images = np.zeros((3, self.channel, self.image_height, self.image_width))
        labels = np.zeros((3))

        # ------------------------------#
        #   获得同一相机的两张背景
        #   用来作为anchor和positive
        # ------------------------------#
        # c_cfy = self.labels[index]
        # path_index = self.paths[index]
        #
        # image = Image.open(path_index)
        # image = self.get_random_data(image, [self.image_height, self.image_width])
        # image = np.transpose(np.asarray(image).astype(np.float64), [2, 0, 1]) / 255
        # if self.channel == 1:
        #     images[0, 0, :, :] = image
        # else:
        #     images[0, :, :, :] = image
        # labels[0] = c_cfy
        #
        # c_path_index = np.array(range(self.num_train))
        # c_path_index = c_path_index[self.labels[:] == c_cfy]
        # c_path_index = c_path_index[c_path_index != index]
        # image_indexes = np.random.choice(range(0, len(c_path_index)), 1)  # 从c_path_index随机选一张作为一个样本的positive
        # image = Image.open(self.paths[image_indexes[0]])

        # selected_path_cfy = self.paths[self.labels[:] == c_cfy]
        # selected_path_cfy = selected_path_cfy[selected_path_cfy[:] != path_index]
        # image_indexes = np.random.choice(range(0, len(selected_path_cfy)), 1)  # 从selected_path随机选一张作为一个样本的positive
        # image = Image.open(selected_path_cfy[image_indexes[0]])

        # image = self.get_random_data(image, [self.image_height, self.image_width])
        #         # image = np.transpose(np.asarray(image).astype(np.float64), [2, 0, 1]) / 255
        #         # if self.channel == 1:
        #         #     images[1, 0, :, :] = image
        #         # else:
        #         #     images[1, :, :, :] = image
        #         # labels[1] = c_cfy


        c = 2*random.randint(0, self.num_classes - 1)  # 先随机选一个类别,randint函数可以选到0或num_classes - 1
        selected_path = self.paths[self.labels[:] == c]  # 从训练集或验证机的所有path中取出类别为c的路径selected_path
        while len(selected_path) < 2:
            c = random.randint(0, self.num_classes - 1)
            selected_path = self.paths[self.labels[:] == c]

        # ------------------------------#
        #   随机选择两张
        # ------------------------------#
        image_indexes = np.random.choice(range(0, len(selected_path)), 2)  # 从selected_path随机选两张作为一个样本的anchor和positive
        image = Image.open(selected_path[image_indexes[0]])
        image = self.get_random_data(image, [self.image_height, self.image_width])
        image = np.transpose(np.asarray(image).astype(np.float64), [2, 0, 1]) / 255
        if self.channel == 1:
            images[0, 0, :, :] = image
        else:
            images[0, :, :, :] = image
        labels[0] = c

        image = Image.open(selected_path[image_indexes[1]])
        # plt.imshow(image)
        image = self.get_random_data(image, [self.image_height, self.image_width])
        # plt.imshow(image)
        image = np.transpose(np.asarray(image).astype(np.float64), [2, 0, 1]) / 255
        # plt.imshow(image)

        if self.channel == 1:
            images[1, 0, :, :] = image
        else:
            images[1, :, :, :] = image
        labels[1] = c

        # ------------------------------#
        #   取出另外一个人的人脸
        # ------------------------------#
        # different_c = list(range(self.num_classes))
        # different_c.pop(c)  # different_c此时有num_classes-1个数
        # different_c_index = np.random.choice(range(0, self.num_classes - 1),
        #                                      1)  # 从0到num_classes-2中选一个索引（共num_classes-1个数）
        # current_c = different_c[different_c_index[0]]
        current_c = c+1
        selected_path = self.paths[self.labels == current_c]
        while len(selected_path) < 1:
            print("有动物图像数量少于1:", c + 1)
            # different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
            # current_c = c+1
            current_c = random.randint(0, self.num_classes - 1) #若没有有动物图像，则从所有图像中随机选一个类别
            selected_path = self.paths[self.labels == current_c]


        # ------------------------------#
        #   随机选择一张同相机有动物的图像
        # ------------------------------#
        image_indexes = np.random.choice(range(0, len(selected_path)), 1)
        image = Image.open(selected_path[image_indexes[0]])
        image = self.get_random_data(image, [self.image_height, self.image_width])
        image = np.transpose(np.asarray(image).astype(np.float64), [2, 0, 1]) / 255
        if self.channel == 1:
            images[2, 0, :, :] = image
        else:
            images[2, :, :, :] = image
        labels[2] = current_c

        return images, labels  # 此为每个batch中的一个样本，包含三张图


class FacenetDataset_N2(Dataset):
    def __init__(self, input_shape, dataset_path, num_train, num_classes):
        super(FacenetDataset, self).__init__()

        self.dataset_path = dataset_path

        self.image_height = input_shape[0]
        self.image_width = input_shape[1]
        self.channel = input_shape[2]

        self.paths = []
        self.labels = []

        self.num_train = num_train

        self.num_classes = num_classes
        self.load_dataset()

    def __len__(self):
        return self.num_train

    def load_dataset(self):
        for path in self.dataset_path:
            # path_split = path.split(";")
            path_split = path.split(",,,")
            # self.paths.append(path_split[1].split()[0])
            self.paths.append(path_split[0])
            self.labels.append(int(path_split[1].split('\n')[0])) # cfy，根据我的路径，不要删掉空格
        self.paths = np.array(self.paths, dtype=np.object)
        self.labels = np.array(self.labels)

    def get_random_data(self, image, input_shape, jitter=.1, hue=.05, sat=1.3, val=1.3, flip_signal=True):
        image = image.convert("RGB")

        h, w = input_shape
        rand_jit1 = rand(1 - jitter, 1 + jitter)
        rand_jit2 = rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        scale = rand(0.9, 1.1)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        flip = rand() < .5
        if flip and flip_signal:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        rotate = rand() < .5
        if rotate:
            angle = np.random.randint(-5, 5)
            a, b = w / 2, h / 2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])

        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        if self.channel == 1:
            image_data = Image.fromarray(np.uint8(image_data)).convert("L")
        # cv2.imshow("TEST",np.uint8(cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)))
        # cv2.waitKey(0)
        return image_data

    def __getitem__(self, index):
        images = np.zeros((3, self.channel, self.image_height, self.image_width))
        labels = np.zeros((3))

        # ------------------------------#
        #   获得同一相机的两张背景
        #   用来作为anchor和positive
        # ------------------------------#
        # c_cfy = self.labels[index]
        # path_index = self.paths[index]
        #
        # image = Image.open(path_index)
        # image = self.get_random_data(image, [self.image_height, self.image_width])
        # image = np.transpose(np.asarray(image).astype(np.float64), [2, 0, 1]) / 255
        # if self.channel == 1:
        #     images[0, 0, :, :] = image
        # else:
        #     images[0, :, :, :] = image
        # labels[0] = c_cfy
        #
        # c_path_index = np.array(range(self.num_train))
        # c_path_index = c_path_index[self.labels[:] == c_cfy]
        # c_path_index = c_path_index[c_path_index != index]
        # image_indexes = np.random.choice(range(0, len(c_path_index)), 1)  # 从c_path_index随机选一张作为一个样本的positive
        # image = Image.open(self.paths[image_indexes[0]])

        # selected_path_cfy = self.paths[self.labels[:] == c_cfy]
        # selected_path_cfy = selected_path_cfy[selected_path_cfy[:] != path_index]
        # image_indexes = np.random.choice(range(0, len(selected_path_cfy)), 1)  # 从selected_path随机选一张作为一个样本的positive
        # image = Image.open(selected_path_cfy[image_indexes[0]])

        # image = self.get_random_data(image, [self.image_height, self.image_width])
        #         # image = np.transpose(np.asarray(image).astype(np.float64), [2, 0, 1]) / 255
        #         # if self.channel == 1:
        #         #     images[1, 0, :, :] = image
        #         # else:
        #         #     images[1, :, :, :] = image
        #         # labels[1] = c_cfy


        c = 2*random.randint(0, self.num_classes - 1)  # 先随机选一个类别,randint函数可以选到0或num_classes - 1
        selected_path = self.paths[self.labels[:] == c]  # 从训练集或验证机的所有path中取出类别为c的路径selected_path
        while len(selected_path) < 2:
            c = random.randint(0, self.num_classes - 1)
            selected_path = self.paths[self.labels[:] == c]

        # ------------------------------#
        #   随机选择两张
        # ------------------------------#
        image_indexes = np.random.choice(range(0, len(selected_path)), 2)  # 从selected_path随机选两张作为一个样本的anchor和positive
        image = Image.open(selected_path[image_indexes[0]])
        image = self.get_random_data(image, [self.image_height, self.image_width])
        image = np.transpose(np.asarray(image).astype(np.float64), [2, 0, 1]) / 255
        if self.channel == 1:
            images[0, 0, :, :] = image
        else:
            images[0, :, :, :] = image
        labels[0] = c

        image = Image.open(selected_path[image_indexes[1]])
        image = self.get_random_data(image, [self.image_height, self.image_width])
        image = np.transpose(np.asarray(image).astype(np.float64), [2, 0, 1]) / 255
        if self.channel == 1:
            images[1, 0, :, :] = image
        else:
            images[1, :, :, :] = image
        labels[1] = c

        # ------------------------------#
        #   取出另外一个人的人脸
        # ------------------------------#
        # different_c = list(range(self.num_classes))
        # different_c.pop(c)  # different_c此时有num_classes-1个数
        # different_c_index = np.random.choice(range(0, self.num_classes - 1),
        #                                      1)  # 从0到num_classes-2中选一个索引（共num_classes-1个数）
        # current_c = different_c[different_c_index[0]]
        current_c = c+1
        selected_path = self.paths[self.labels == current_c]
        while len(selected_path) < 1:
            print("有动物图像数量少于1:", c + 1)
            # different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
            # current_c = c+1
            current_c = random.randint(0, self.num_classes - 1) #若没有有动物图像，则从所有图像中随机选一个类别
            selected_path = self.paths[self.labels == current_c]


        # ------------------------------#
        #   随机选择一张同相机有动物的图像
        # ------------------------------#
        image_indexes = np.random.choice(range(0, len(selected_path)), 1)
        image = Image.open(selected_path[image_indexes[0]])
        image = self.get_random_data(image, [self.image_height, self.image_width])
        image = np.transpose(np.asarray(image).astype(np.float64), [2, 0, 1]) / 255
        if self.channel == 1:
            images[2, 0, :, :] = image
        else:
            images[2, :, :, :] = image
        labels[2] = current_c

        return images, labels  # 此为每个batch中的一个样本，包含三张图






# DataLoader中collate_fn使用
def dataset_collate(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)

    images1 = np.array(images)[:, 0, :, :, :]
    images2 = np.array(images)[:, 1, :, :, :]
    images3 = np.array(images)[:, 2, :, :, :]
    images = np.concatenate([images1, images2, images3], 0)

    labels1 = np.array(labels)[:, 0]
    labels2 = np.array(labels)[:, 1]
    labels3 = np.array(labels)[:, 2]
    labels = np.concatenate([labels1, labels2, labels3], 0)
    return images, labels
