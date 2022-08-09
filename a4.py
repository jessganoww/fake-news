import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from gensim.models import Word2Vec

def read_file(file_path):
    with open(file_path) as f:
        data = [line.rstrip('\n').lower().split(',') for line in f]
    return data

def create_embeddings(sentence):
    embeddings = []

    for word in sentence:
        try:
            # embeddings = [w2v.wv[word]]
            embeddings.append(w2v.wv[word])
        except KeyError:
            pass

    if not embeddings:
        return np.zeros(100)

    return np.sum(embeddings, axis=0)

def train_model(model, optimizer, criterion, train_loader, val_loader, n_epoch, kind):
    fmodel_accuracy = 0

    print('Training model ({})...'.format(kind))

    for e in range(n_epoch):
        train_accuracy = 0
        val_accuracy = 0
        fmodel_accuracy = 0

        model.train()

        for idx, train_data in enumerate(train_loader):
            x, y = train_data
            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)
            train_accuracy += (y_pred.argmax(axis=1, keepdim=True) == y).float().sum().item()

            loss.backward()
            optimizer.step()

        model.eval()
        train_accuracy /= len(train_loader.dataset)

        for idx, val_data in enumerate(val_loader):
            x, y = val_data

            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_accuracy += (y_pred.argmax(1, keepdim=True) == y).float().sum().item()

        val_accuracy /= len(val_loader.dataset)

        if val_accuracy > fmodel_accuracy:
            fmodel_accuracy = val_accuracy
            fmodel = model.state_dict()

        # print('\nEpoch {}/{}: --- Train: {} Validate: {}'.format(e, n_epoch, train_accuracy*100, val_accuracy*100))

    # fmodel_name = str(e) + '-' + kind + '-d05-L2'
    fmodel_name = 'nn_' + kind
    model.load_state_dict(fmodel)
    torch.save(model, OUT_FP + fmodel_name + '.model')

    return train_accuracy, val_accuracy, fmodel_name


def test_model(model, criterion, test_loader):
    test_accuracy = 0

    model.eval()
    for idx, test_data in enumerate(test_loader):
        x, y = test_data

        y_pred = model(x)
        loss = criterion(y_pred, y)
        test_accuracy += (y_pred.argmax(1, keepdim=True) == y).float().sum().item()

    return test_accuracy / len(test_loader.dataset)


class ReviewDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

parser = argparse.ArgumentParser()
parser.add_argument('datapath')
args = parser.parse_args()

TRAIN_FP = os.path.join(args.datapath, 'train.csv')
TRAIN_NS_FP = os.path.join(args.datapath, 'train_ns.csv')
TEST_FP = os.path.join(args.datapath, 'fake-news.csv')
TEST_NS_FP = os.path.join(args.datapath, 'test_ns.csv')
VAL_FP = os.path.join(args.datapath, 'val.csv')
VAL_NS_FP = os.path.join(args.datapath, 'val_ns.csv')

OUT_FP = os.path.join(os.getcwd(), 'data/')

train_txt = read_file(TRAIN_FP)
test_txt = read_file(TEST_FP)
val_txt = read_file(VAL_FP)

w2v = Word2Vec.load('../a3/data/w2v.model')

train_embeddings = torch.from_numpy(np.array([create_embeddings(s) for s in train_txt])).float()
test_embeddings = torch.from_numpy(np.array([create_embeddings(s) for s in test_txt])).float()
val_embeddings = torch.from_numpy(np.array([create_embeddings(s) for s in val_txt])).float()

train_label = np.array([1] * (len(train_txt) // 2) + [0] * (len(train_txt) // 2))
# train_label = np.zeros((temp.size, 2))
# train_label[np.arange(temp.size), temp] = 1

testval_label = np.array([1] * (len(test_txt) // 2) + [0] * (len(test_txt) // 2))
# testval_label = np.zeros((temp.size, 2))
# testval_label[np.arange(temp.size), temp] = 1

train = ReviewDataset(train_embeddings, train_label)
val = ReviewDataset(val_embeddings, testval_label)
test = ReviewDataset(test_embeddings, testval_label)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
val_loader = DataLoader(val, batch_size=32)
test_loader = DataLoader(test, batch_size=32)

FFNN_relu = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Dropout(p=0.25),
    nn.Linear(50, 2),
    nn.Softmax(dim=1)
)

FFNN_sigmoid = nn.Sequential(
    nn.Linear(100, 50),
    nn.Sigmoid(),
    nn.Dropout(p=0.25),
    nn.Linear(50, 2),
    nn.Softmax(dim=1)
)

FFNN_tanh = nn.Sequential(
    nn.Linear(100, 50),
    nn.Tanh(),
    nn.Dropout(p=0.25),
    nn.Linear(50, 2),
    nn.Softmax(dim=1)
)

loss_fn = nn.CrossEntropyLoss()

optimizer_relu = optim.Adam(FFNN_relu.parameters(), lr=1e-4, weight_decay=1e-5)
train_accuracy, val_accuracy, model_name = train_model(FFNN_relu, optimizer_relu, loss_fn, train_loader, val_loader, 5,
                                                       'relu')
model = torch.load(OUT_FP + model_name + '.model')
relu_accuracy = test_model(FFNN_relu, loss_fn, test_loader)

optimizer_sigmoid = optim.Adam(FFNN_sigmoid.parameters(), lr=1e-4, weight_decay=1e-5)
train_accuracy, val_accuracy, model_name = train_model(FFNN_sigmoid, optimizer_sigmoid, loss_fn, train_loader,
                                                       val_loader, 5, 'sigmoid')
model = torch.load(OUT_FP + model_name + '.model')
sigmoid_accuracy = test_model(FFNN_sigmoid, loss_fn, test_loader)

optimizer_tanh = optim.Adam(FFNN_tanh.parameters(), lr=1e-4, weight_decay=1e-5)
train_accuracy, val_accuracy, model_name = train_model(FFNN_tanh, optimizer_tanh, loss_fn, train_loader, val_loader, 5,
                                                       'tanh')
model = torch.load(OUT_FP + model_name + '.model')
tanh_accuracy = test_model(FFNN_tanh, loss_fn, test_loader)

