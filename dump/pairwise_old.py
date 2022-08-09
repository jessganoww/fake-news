import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange


class BlockA(nn.Module):  # without dropout
    def __init__(self, in_features, out_features):
        super(BlockA, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class BlockB(nn.Module):  # with dropout
    def __init__(self, in_features, out_features, dropout_rate):
        super(BlockB, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.d = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.d(x)
        return x


class PairwiseBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(PairwiseBlock, self).__init__()
        self.b1 = BlockA(in_features, out_features)
        self.b2 = BlockA(in_features, out_features)

    def forward(self, headline, body):
        h_out = self.b1(headline) + body
        b_out = self.b2(body) + headline
        return h_out, b_out


class Net3(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate, num_blocks=3):
        super(Net3, self).__init__()
        self.pblocks = nn.ModuleList([PairwiseBlock(in_features, out_features) for _ in num_blocks])

        # self.p1 = PairwiseBlock(in_features, out_features)
        # self.p2 = PairwiseBlock(in_features, out_features)
        # self.p3 = PairwiseBlock(in_features, out_features)

        self.b1 = BlockB(768, 384, dropout_rate)
        self.b2 = BlockB(384, 192, dropout_rate)
        self.b3 = BlockB(192, 96, dropout_rate)

        self.fc1 = nn.Linear(96, 4)

    def forward(self, headline, body):
        h, b = headline, body
        for pblock in self.pblocks:
            h, b = pblock(h, b)

        # h, b = self.p1(headline, body)
        # h, b = self.p2(h, b)
        # h, b = self.p3(h, b)

        x = self.b1(torch.cat((h, b), dim=1))
        x = self.b2(x)
        x = self.b3(x)
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x


class Net5(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate, num_blocks=5):
        super(Net5, self).__init__()
        self.pblocks = nn.ModuleList([PairwiseBlock(in_features, out_features) for _ in num_blocks])

        # self.p1 = PairwiseBlock(in_features, out_features)
        # self.p2 = PairwiseBlock(in_features, out_features)
        # self.p3 = PairwiseBlock(in_features, out_features)
        # self.p4 = PairwiseBlock(in_features, out_features)
        # self.p5 = PairwiseBlock(in_features, out_features)

        # MiniLM
        self.b1 = BlockB(768, 384, dropout_rate)
        self.b2 = BlockB(384, 192, dropout_rate)
        self.b3 = BlockB(192, 96, dropout_rate)

        # mpnet
        # self.b1 = BlockB(1536, 768, dropout_rate)
        # self.b2 = BlockB(768, 384, dropout_rate)
        # self.b3 = BlockB(384, 192, dropout_rate)
        # self.b4 = BlockB(192, 96, dropout_rate)

        self.fc1 = nn.Linear(96, 4)

    def forward(self, headline, body):
        h, b = headline, body
        for pblock in self.pblocks:
            h, b = pblock(h, b)

        # h, b = self.p1(headline, body)
        # h, b = self.p2(h, b)
        # h, b = self.p3(h, b)
        # h, b = self.p4(h, b)
        # h, b = self.p5(h, b)

        x = self.b1(torch.cat((h, b), dim=1))
        x = self.b2(x)
        x = self.b3(x)
        # x = self.b4(x)  # mpnet only
        x = self.fc1(x)
        x = F.softmax(x, dim=1)

        return x


#########################################################################

def train_model(model, optim, loss_fn, epoch, train_dl, val_dl, model_id):
    print('Training model ({})...'.format(model_id))
    device = torch.device('mps' if torch.has_mps else 'cpu')

    train_accuracy_list = []
    val_accuracy_list = []
    train_loss_list = []
    val_loss_list = []

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    model.train()
    pbar = trange(epoch)

    for e in pbar:
        train_accuracy = 0
        val_accuracy = 0
        train_loss = 0
        val_loss = 0
        best_accuracy = 0

        for idx, data in enumerate(train_dl):
            x1, x2, y = data  # headline, body, stance

            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            optim.zero_grad()

            y_pred = model(x1, x2)
            loss = loss_fn(y_pred, y)
            train_accuracy += (y_pred.argmax(axis=1) == y).float().sum().item()
            train_loss += loss.item()

            loss.backward()
            optim.step()

        train_accuracy /= len(train_dl.dataset)
        train_loss /= len(train_dl.dataset)
        train_accuracy_list.append(train_accuracy)
        train_loss_list.append(train_loss)

        for idx, data in enumerate(val_dl):
            x1, x2, y = data

            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            y_pred = model(x1, x2)
            val_accuracy += (y_pred.argmax(axis=1) == y).float().sum().item()
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()

        val_accuracy /= len(val_dl.dataset)
        val_loss /= len(val_dl.dataset)
        val_accuracy_list.append(val_accuracy)
        val_loss_list.append(val_loss)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model.state_dict()

        model.load_state_dict(best_model)
        torch.save(model, 'models/' + model_id + '.model')

        pbar.set_description('Epoch {}: Accuracy: {} | Loss: {}'.format(e, val_accuracy, val_loss))

    model.eval()

    return train_accuracy_list, train_loss_list, val_accuracy_list, val_loss_list


def test_model(test_dl, model_id):
    print('Testing model ({})...'.format(model_id))
    device = torch.device('mps' if torch.has_mps else 'cpu')

    model = torch.load('models/' + model_id + '.model')
    model = model.to(device)
    model.eval()

    pred = []

    for idx, data in enumerate(test_dl):
        x1, x2 = data

        x1 = x1.to(device)
        x2 = x2.to(device)

        y_pred = model(x1, x2)
        pred.extend(y_pred.argmax(axis=1).tolist())

    return pred


# input = torch.rand((5, 384))
# input2 = torch.rand((5, 384))
#
# b = Net3(384, 384, 0.2)
# output = b(input, input2)
