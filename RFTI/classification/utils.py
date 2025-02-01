import torch
import shutil



def save_checkpoint(state, is_best, filename='./models_pkl/checkpoint_allCamera_2n_focal_50_0.5_1000.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './models_pkl/model_best_allCamera_2n_focal_50_0.5_1000.pth.tar')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n #val * n为本次的数量，sum为目前为止的总数
        self.count += n
        self.avg = self.sum / self.count #求目前为止的平均值





def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True) #_, ind分别为最大值和对应索引
    correct = ind.eq(targets.view(-1, 1).expand_as(ind)) #correct为一个batchsize中每个图像预测对错的结果（bool型张量列表）
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size),correct




