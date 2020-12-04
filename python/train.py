import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DATASET_PATH = './dataset'
BATCH_SIZE = 16
TEST_RATIO = .2
SEED = 0
EPOCH = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True


class S3TDataset():
    '''Spectrograms of 3 Temperations.'''

    def __init__(self, path='./dataset'):
        self._path = path

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
            stereo = (specgram[:1], specgram[1:])
            specgram = np.concatenate(stereo, axis=2)
            self._items.append((specgram, i))

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

    def partial(self, ratio=.1):
        labels = [[] for _ in range(len(self._labels))]
        for s, l in self._items:
            labels[l].append(s)

        self._items = []
        partial_items = []

        for i, label in enumerate(labels):
            k = int(len(label) * ratio)
            samples = set(random.sample(range(len(label)), k))
            for j, s in enumerate(label):
                if j in samples:
                    partial_items.append((s, i))
                else:
                    self._items.append((s, i))

        class PartialDataset(S3TDataset):
            _path = self._path
            _labels = self._labels
            _items = partial_items

            def __init__(self):
                pass

        return PartialDataset()

    def split(self):
        splited_items = []
        for s, l in self._items:
            for i in range(s.shape[2]):
                splited_items.append((s[:, :, i:i + 1], l))
        self._items = splited_items


class D3TModel(nn.Module):
    '''Discriminator of 3 Temperations.'''

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, 3)

    def forward(self, x):
        # (batch, channel=1, bin=2048)
        x = self.fc1(x)
        # (batch, channel=1, class=3)
        return x


train_set = S3TDataset(DATASET_PATH)
test_set = train_set.partial(TEST_RATIO)
train_set.split()
print('train:', len(train_set))
print('test:', len(test_set))
train_loader = train_set.data_loader(BATCH_SIZE, shuffle=True)
test_loader = test_set.data_loader()

model = D3TModel()

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)


def train_epoch():
    model.train()
    for i, (data, label) in enumerate(train_loader):
        data = data.to(device)  # (batch, channel=1, bin, frame=1)
        label = label.to(device)

        output = model(data[:, :, :, 0])
        output = F.log_softmax(output, dim=2)
        loss = F.nll_loss(output.squeeze(1), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('[%5d]\r' % (i * BATCH_SIZE), end='', flush=True)


def test():
    model.eval()
    correct = 0
    for i, (data, label) in enumerate(test_loader):
        data = data.to(device)  # (batch=1, channel=1, bin, frame)
        label = label.to(device)
        pred_sum = torch.zeros(3)

        for j in range(data.size(-1)):
            with torch.no_grad():
                output = model(data[:, :, :, j])
            output = output.squeeze()
            output = F.softmax(output, dim=0)
            pred_sum += output

        pred = pred_sum.argmax(dim=-1, keepdim=True)
        if pred == label:
            correct += 1
    print('correct: %d/%d' % (correct, i + 1))


for i in range(EPOCH):
    print('epoch:', i + 1)
    train_epoch()
    test()
