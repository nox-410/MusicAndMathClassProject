import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class S3TDataset():
    '''Spectrograms of 3 Temperations.'''

    def __init__(self, path='./dataset', sample_rate=8000, split=False):
        self._path = path
        self._sample_rate = sample_rate

        self._labels = []
        for i in os.listdir(path):
            d = os.path.join(path, i)
            if os.path.isdir(d) and not i.startswith('.'):
                self._labels.append(i)
        self._labels.sort()

        items = []
        for i, l in enumerate(self._labels):
            d = os.path.join(path, l)
            for n in os.listdir(d):
                if n.endswith('.npy') and not n.startswith('.'):
                    items.append((n, l, i))

        self._items = []
        for n, l, i in sorted(items):
            p = os.path.join(path, l, n)
            specgram = np.load(p)  # (channel, bin, frame)
            label = torch.tensor(i)
            if split:
                for j in range(specgram.shape[2]):
                    self._items.append((specgram[:, :, j:j + 1], label))
            else:
                self._items.append((specgram, label))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def data_loader(self, batch_size=1, shuffle=False):
        if device == 'cuda':
            num_workers = 4
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class D3TModel(nn.Module):
    '''Discriminator of 3 Temperations.'''

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2049, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.log_softmax(x, dim=2)
        return x


train_set = S3TDataset('./train_set', split=True)
print('train:', len(train_set))
train_loader = train_set.data_loader(BATCH_SIZE, True)

test_set = S3TDataset('./test_set')
print('test:', len(test_set))
test_loader = test_set.data_loader()

model = D3TModel()

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)


def train_epoch():
    model.train()
    for i, (data, label) in enumerate(train_loader):
        data = data.to(device)  # (batch, channel=1, bin, frame=1)
        label = label.to(device)

        output = model(data[:, :, :, 0])
        loss = F.nll_loss(output.squeeze(1), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('[%5d]\r' % i, end='', flush=True)


def test():
    model.eval()
    correct = 0
    for i, (data, label) in enumerate(test_loader):
        data = data.to(device)  # (batch=1, channel=1, bin, frame)
        label = label.to(device)
        pred_count = [0] * 3

        for j in range(data.size(-1)):
            output = model(data[:, :, :, j])
            pred = output.argmax(-1).item()
            pred_count[pred] += 1

        pred = pred_count.index(max(pred_count))
        if pred == label.item():
            correct += 1
    print('correct: %d/%d' % (correct, i + 1))


for i in range(5):
    print('epoch:', i + 1)
    train_epoch()
    test()
