import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False,model_name='checkpoint.pth', delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.val_loss_min = np.Inf
        self.test_acc_max = 0
        self.model_name = model_name
        self.delta = delta

    def __call__(self, test_acc, model):

        # score = -val_loss
        score = test_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(test_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(test_acc, model)
            self.counter = 0

    def save_checkpoint(self, test_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print(f'Test acc increase ({self.test_acc_max:.6f} --> {test_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_name)	# 这里会存储迄今最优模型的参数
        self.test_acc_max = test_acc

