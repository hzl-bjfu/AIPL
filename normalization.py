import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
from net.resnet34MHA1 import ResNet34WithAttention

torch.manual_seed(2026)
np.random.seed(0)

with open('./data1/3channel_S3.pkl', 'rb') as f:
    data = pickle.load(f)

frames_train = data[0]
labels_train = data[1]


class BirdDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = torch.tensor(frames, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.frames[idx], self.labels[idx]

dataset_all = BirdDataset(frames_train, labels_train)
dataset_train, dataset_val, dataset_test = random_split(dataset_all, [0.6, 0.2, 0.2])

train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)



#  自适应归一化模块
class AdaptiveNorm(nn.Module):
    def __init__(self, mode='all'):
        super(AdaptiveNorm, self).__init__()
        self.mode = mode

        if mode == 'channel':
            self.gammas = nn.Parameter(torch.tensor([10.0, 0.0, 0.0]))
        elif mode == 'time':
            self.gammas = nn.Parameter(torch.tensor([0.0, 10.0, 0.0]))
        elif mode == 'freq':
            self.gammas = nn.Parameter(torch.tensor([0.0, 0.0, 10.0]))
        elif mode == 'all':
            self.gammas = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, x):
        B, C, T, F = x.shape

        norm_C = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        norm_T = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-6)
        norm_F = (x - x.mean(dim=3, keepdim=True)) / (x.std(dim=3, keepdim=True) + 1e-6)

        gamma_softmax = torch.softmax(self.gammas, dim=0)

        if   self.mode == 'channel': out = gamma_softmax[0] * norm_C
        elif self.mode == 'time':    out = gamma_softmax[1] * norm_T
        elif self.mode == 'freq':    out = gamma_softmax[2] * norm_F
        else:
            out = (
                gamma_softmax[0] * norm_C +
                gamma_softmax[1] * norm_T +
                gamma_softmax[2] * norm_F
            )

        return out



class BirdClassifier(nn.Module):
    def __init__(self, mode='all'):
        super(BirdClassifier, self).__init__()
        self.norm = AdaptiveNorm(mode=mode)
        self.backbone = ResNet34WithAttention(num_classes=5)

    def forward(self, x):
        x = self.norm(x)
        return self.backbone(x)



def train(model, train_loader, val_loader, epochs=100, lr=0.01, patience=5):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    wait = 0

    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {(correct/total):.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break



def evaluate(model, test_loader):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total, correct = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

    print(f"Test Accuracy: {correct / total:.4f}")



def main():
    mode = "all"  
    model = BirdClassifier(mode=mode)

    train(model, train_loader, val_loader, epochs=100, lr=0.0001)
    model.load_state_dict(torch.load("best_model.pth"))
    evaluate(model, test_loader)

if __name__ == "__main__":
    main()
