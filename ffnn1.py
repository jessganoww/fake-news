import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNN1(nn.Module):
    def __init__(self):
        super(FFNN1, self).__init__()
        self.fc1 = nn.Linear(384, 192)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 96)
        self.fc4 = nn.Linear(192, 96)
        self.fc5 = nn.Linear(192, 96) # concat
        self.fc6 = nn.Linear(96, 4)

    def forward(self, headline, body):
        h = self.fc1(headline)
        b = self.fc2(body)
        h = F.relu(h)
        b = F.relu(b)
        h = self.fc3(h)
        b = self.fc4(b)
        x = torch.cat((h, b), dim=1)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.softmax(x, dim=1)

        return x


class FFNN2(nn.Module):
    def __init__(self):
        super(FFNN2, self).__init__()
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 96)
        self.fc4 = nn.Linear(96, 4)

    def forward(self, headline, body):
        # x = torch.cat((headline, body), dim=1)
        x = self.fc1(torch.cat((headline, body), dim=1))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.softmax(x, dim=1)

        return x
    

class FFNN3(nn.Module):
    def __init__(self):
        super(FFNN3, self).__init__()
        self.fc1 = nn.Linear(384, 192)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(1200, 600)
        self.fc4 = nn.Linear(1200, 600)

        self.fc5 = nn.Linear(192, 96)
        self.fc6 = nn.Linear(192, 96)
        self.fc7 = nn.Linear(600, 300)
        self.fc8 = nn.Linear(600, 300)

        self.fc9 = nn.Linear(792, 396)
        self.fc10 = nn.Linear(396, 4)
        # self.fc11 = nn.Linear(384, 4)

    def forward(self, headline_embeddings, body_embeddings, headline_tfidf, body_tfidf):
        he = self.fc1(headline_embeddings)
        be = self.fc2(body_embeddings)
        ht = self.fc3(headline_tfidf)
        bt = self.fc4(body_tfidf)
        he = F.relu(he)
        be = F.relu(be)
        ht = F.relu(ht)
        bt = F.relu(bt)

        he = self.fc5(he)
        be = self.fc6(be)
        ht = self.fc7(ht)
        bt = self.fc8(bt)
        he = F.relu(he)
        be = F.relu(be)
        ht = F.relu(ht)
        bt = F.relu(bt)

        x = torch.cat((he, be, ht, bt), dim=1)
        x = F.relu(x)

        x = self.fc9(x)
        x = F.relu(x)

        x = self.fc10(x)
        x = F.softmax(x, dim=1)

        # x = self.fc11(x)
        # x = F.softmax(x)

        return x
