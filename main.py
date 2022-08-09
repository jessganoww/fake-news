import matplotlib.pyplot as plt
import os
import pickle
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import *
from pairwise import NetA, NetB, train_model, test_model
from score import score_submission, print_confusion_matrix

device = torch.device('mps')

SBERT_MODEL = 'all-mpnet-base-v2'
# SBERT_MODEL = 'all-MiniLM-L6-v2'
BATCH_SIZE = 32  # dataloader
EPOCHS = 2
LR = 0.001
DIM = 768
DROPOUT = 0.2

# word embeddings
# net3d02-mini >>> 7120.25 --------
# net3d04-mini >>> 7080.75 --------
# net3d02-mpnet >>> 6902.5 --------
# net3d04-mpnet >>> 6995.5 --------

# net5d02-mini >>> 7225.5 --------
# net5d04-mini >>> 6824.5 --------
# net5d02-mpnet >>> 7328.75 --------
# net5d04-mpnet >>> 7154.75 --------

# net7d02-mini >>> 7127.25 --------
# net7d04-mini >>> 7372 --------
# net7d02-mpnet >>> 7457.25 --------
# net7d04-mpnet >>> 7478.75 --------

# word embeddings + score
# net3d02-mini-score >>> 7433.75 <<<
# net5d02-mini-score >>> 7369
# net7d02-mini-score >>> 7240.25

# lower dropout better
# deeper network better

models_dict = {
    'net3A': NetA(DIM, DIM, DROPOUT, 3),
    'net3B': NetB(DIM, DIM, DROPOUT, 3),
    'net5A': NetA(DIM, DIM, DROPOUT, 5),
    'net5B': NetB(DIM, DIM, DROPOUT, 5),
    'net7A': NetA(DIM, DIM, DROPOUT, 7),
    'net7B': NetB(DIM, DIM, DROPOUT, 7),
}

experiments_dict = {
    'net3d02-mpnet-score': 'net3B',
    'net5d02-mpnet-score': 'net5B',
    'net7d02-mpnet-score': 'net7B'
    # 'net3d02-mini-score': 'net3B',
    # 'net5d02-mini-score': 'net5B',
    # 'net7d02-mini-score': 'net7B'
    # 'net3d02-mini': 'net3',
    # 'net5d02-mini': 'net5',
    # 'net7d02-mini': 'net7'
    # 'net10d02lre3-mpnet': 'net10'
    # 'net3d02lre4-mpnet': 'net3',
    # 'net5d02lre4-mpnet': 'net5',
    # 'net7d02lre4-mpnet': 'net7'
    # 'net3d04lre3-mpnet': 'net3',
    # 'net5d04lre3-mpnet': 'net5',
    # 'net7d04lre3-mpnet': 'net7'
    # 'net3d02lre3-mpnet': 'net3',
    # 'net5d02lre3-mpnet': 'net5',
    # 'net7d02lre3-mpnet': 'net7',
    # 'net3d04lre3-mini': 'net3',
    # 'net5d04lre3-mini': 'net5',
    # 'net7d04lre3-mini': 'net7',
    # 'net3d02lre3-mini': 'net3',
    # 'net5d02lre3-mini': 'net5',
    # 'net7d02lre3-mini': 'net7'
}

df_train_headline = pd.read_csv("data/train_stances.csv")
df_train_body = pd.read_csv("data/train_bodies.csv")
df_test_headline = pd.read_csv("data/competition_test_stances_unlabeled.csv")
df_test_body = pd.read_csv("data/competition_test_bodies.csv")

test_labels = pd.read_csv("data/competition_test_stances.csv")['Stance']

train_headline_dict = create_headline_dictionary(df_train_headline)
test_headline_dict = create_headline_dictionary(df_test_headline)

# create_embeddings(list(train_headline_dict.values()), df_train_body['articleBody'], 'train', SBERT_MODEL)
# create_embeddings(list(test_headline_dict.values()), df_test_body['articleBody'], 'test', SBERT_MODEL)

# experiment
train_headline_embedding = np.load('embeddings/' + SBERT_MODEL + '/train/headline_embedding.npy')
train_body_embedding = np.load('embeddings/' + SBERT_MODEL + '/train/body_embedding.npy')
test_headline_embedding = np.load('embeddings/' + SBERT_MODEL + '/test/headline_embedding.npy')
test_body_embedding = np.load('embeddings/' + SBERT_MODEL + '/test/body_embedding.npy')

df_train = generate_feature_matrix(df_train_headline, df_train_body, train_headline_dict, 'train')
df_test = generate_feature_matrix(df_test_headline, df_test_body, test_headline_dict, 'test')

# score experiments only
df_train['Score'] = compute_cos_score(df_train, train_headline_embedding, train_body_embedding)
df_test['Score'] = compute_cos_score(df_test, test_headline_embedding, test_body_embedding)

df_train, df_val = split_train_val(df_train)

class_weights = 1. / np.unique(df_train['Stance'], return_counts=True)[1]
samples_weights = torch.from_numpy(np.array([class_weights[s] for s in df_train['Stance']]))

# print('Creating (embedding) dataset...')
# train_ds = EmbeddingTrainDataset(df_train, train_headline_embedding, train_body_embedding)
# val_ds = EmbeddingTrainDataset(df_val, train_headline_embedding, train_body_embedding)
# test_ds = EmbeddingTestDataset(df_test, test_headline_embedding, test_body_embedding)

print('Creating (embedding + score) dataset...')
train_ds = EmbeddingScoreTrainDataset(df_train, train_headline_embedding, train_body_embedding)
val_ds = EmbeddingScoreTrainDataset(df_val, train_headline_embedding, train_body_embedding)
test_ds = EmbeddingScoreTestDataset(df_test, test_headline_embedding, test_body_embedding)

sampler = WeightedRandomSampler(samples_weights.type('torch.DoubleTensor'), len(samples_weights))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

for exp_id in experiments_dict:
    exp_dir = os.path.join('experiments', exp_id)

    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

    model = models_dict[experiments_dict[exp_id]]
    optim_adam = torch.optim.Adam(model.parameters(), lr=LR)
    loss_CE = nn.CrossEntropyLoss()

    print('Training model ({})...'.format(exp_id))
    train_accuracy_list, train_loss_list, val_accuracy_list, val_loss_list = train_model(model, optim_adam, loss_CE,
                                                                                         EPOCHS, train_dl, val_dl,
                                                                                         exp_id)

    print('Testing model ({})'.format(exp_id))
    pred = test_model(test_dl, exp_id)
    pred = [*map(predictions_dict.get, pred)]
    test_accuracy = (pred == test_labels).sum() / len(pred)

    score, cm = score_submission(test_labels, pred)
    print('{} score: {}'.format(exp_id, score))
    print_confusion_matrix(cm)

    print('Generating plots...')
    # train-validation losses
    plt.figure(figsize=(7, 5))
    loss_plt = plt.gca()
    loss_plt.plot(range(1, EPOCHS + 1), train_loss_list, label='train loss')
    loss_plt.plot(range(1, EPOCHS + 1), val_loss_list, label='validation loss')
    loss_plt.set_xlabel('epoch')
    loss_plt.set_ylabel('loss')
    loss_plt.set_ylim(min(train_loss_list)-0.01, max(val_loss_list)+0.01)
    loss_plt.legend(loc=1, fontsize='x-small')
    loss_plt.tick_params(labelsize=8)
    plt.savefig(os.path.join(exp_dir, 'loss_plt.png'), bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()


    # actual / true positive
    plt.figure(figsize=(8, 5))
    x = ['agree', 'disagree', 'discuss', 'unrelated']
    y1 = np.unique(test_labels, return_counts=True)[1]
    y2 = [cm[i][i] for i in range(4)]
    pred_plt = plt.gca()
    pred_plt.bar(x, y1, color='lightgray', alpha=0.65, label='actual')
    pred_plt.bar(x, y2, color='tab:blue', label='true positive')
    pred_plt.set_xlabel('stance')
    pred_plt.set_ylabel('count')
    pred_plt.legend(loc=2, fontsize='x-small')
    pred_plt.tick_params(labelsize=8)
    plt.savefig(os.path.join(exp_dir, 'prediction.png'), bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()


    log = {
        'model': exp_id,
        'blocks': len(model.pblocks),
        'score': score,
        'train_accuracy': train_accuracy_list,
        'train_loss': train_loss_list,
        'val_accuracy': val_accuracy_list,
        'val_loss': val_loss_list,
        'dropout': DROPOUT
        # 'learning_rate': LR,
    }

    print('Saving {} logs...'.format(exp_id))
    log_fp = os.path.join(exp_dir, 'log.pkl')
    with open(log_fp, 'wb') as f:
        pickle.dump(log, f)

    score_fp = os.path.join(exp_dir, 'score.txt')
    with open(score_fp, 'w') as f:
        f.write('{}: {}'.format(exp_id, score))

    print('Exporting csv...')
    answer = pd.DataFrame(data={'Headline': df_test_headline['Headline'], 'Body ID': df_test_headline['Body ID'], 'Stance': pred})
    answer.to_csv(os.path.join(exp_dir, 'answer.csv'), index=False, encoding='utf-8')

