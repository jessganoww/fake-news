import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, headline_embedding, body_embedding):
        self.df = df
        self.headline_embedding = headline_embedding
        self.body_embedding = body_embedding

        self.headline_id = list(df['Headline ID'])
        self.body_id = list(df['Mapped Body ID'])
        self.stance = list(df['Stance'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.headline_embedding[self.headline_id[item]], self.body_embedding[self.body_id[item]], \
               self.stance[item]


class EmbeddingTestDataset(torch.utils.data.Dataset):
    def __init__(self, df, headline_embedding, body_embedding):
        self.df = df
        self.headline_embedding = headline_embedding
        self.body_embedding = body_embedding

        self.headline_id = list(df['Headline ID'])
        self.body_id = list(df['Mapped Body ID'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.headline_embedding[self.headline_id[item]], self.body_embedding[self.body_id[item]]


class EmbeddingScoreTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, headline_embedding, body_embedding):
        self.df = df
        self.headline_embedding = headline_embedding
        self.body_embedding = body_embedding

        self.headline_id = list(df['Headline ID'])
        self.body_id = list(df['Mapped Body ID'])
        self.score = df['Score'].values
        self.stance = list(df['Stance'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.headline_embedding[self.headline_id[item]], self.body_embedding[self.body_id[item]], \
               self.score[item], self.stance[item]


class EmbeddingScoreTestDataset(torch.utils.data.Dataset):
    def __init__(self, df, headline_embedding, body_embedding):
        self.df = df
        self.headline_embedding = headline_embedding
        self.body_embedding = body_embedding

        self.headline_id = list(df['Headline ID'])
        self.body_id = list(df['Mapped Body ID'])
        self.score = df['Score'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.headline_embedding[self.headline_id[item]], self.body_embedding[self.body_id[item]], \
               self.score[item]


class TfidfTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, headline_embedding, body_embedding, headline_tfidf, body_tfidf):
        self.df = df
        self.headline_embedding = headline_embedding
        self.body_embedding = body_embedding
        self.headline_tfidf = headline_tfidf
        self.body_tfidf = body_tfidf

        self.headline_id = list(df['Headline ID'])
        self.body_id = list(df['Body ID'])
        self.stance = list(df['Stance'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.headline_embedding[self.headline_id[item]], self.body_embedding[self.body_id[item]], \
               self.headline_tfidf[self.headline_id[item]], self.body_tfidf[self.body_id[item]], self.stance[item]


class TfidfTestDataset(torch.utils.data.Dataset):
    def __init__(self, df, headline_embedding, body_embedding, headline_tfidf, body_tfidf):
        self.df = df
        self.headline_embedding = headline_embedding
        self.body_embedding = body_embedding
        self.headline_tfidf = headline_tfidf
        self.body_tfidf = body_tfidf

        self.headline_id = list(df['Headline ID'])
        self.body_id = list(df['Body ID'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.headline_embedding[self.headline_id[item]], self.body_embedding[self.body_id[item]], \
               self.headline_tfidf[self.headline_id[item]], self.body_tfidf[self.body_id[item]]


def create_headline_dictionary(df):
    dictionary = {}

    for h in df['Headline']:
        if h not in dictionary.values():
            key = len(dictionary)
            dictionary[key] = h

    return dictionary


def create_embeddings(headline_list, body_list, ds_type, model):
    sbert = SentenceTransformer(model)

    print('Creating {} headline embeddings...'.format(ds_type))
    headline_embedding = sbert.encode(headline_list, batch_size=16, show_progress_bar=True)

    print('Creating {} body embeddings...'.format(ds_type))
    body_embedding = sbert.encode(body_list, batch_size=16, show_progress_bar=True)

    np.save('embeddings/' + str(model) + '/' + str(ds_type) + '/headline_embedding.npy', headline_embedding)
    np.save('embeddings/' + str(model) + '/' + str(ds_type) + '/body_embedding.npy', body_embedding)


def generate_feature_matrix(df_headline, df_body, headline_dict, df_type):
    df_body['Mapped Body ID'] = range(len(df_body))
    df_headline['Mapped Body ID'] = df_headline['Body ID'].map(dict(zip(df_body['Body ID'], df_body['Mapped Body ID'])))
    df_headline['Headline ID'] = df_headline['Headline'].map(
        dict(zip(list(headline_dict.values()), headline_dict.keys())))

    if df_type == 'train':
        df_combined = df_headline[['Headline ID', 'Body ID', 'Mapped Body ID', 'Stance']].copy()
        df_combined['Stance'] = df_combined['Stance'].apply(stance_dict.get)
    else:
        df_combined = df_headline[['Headline ID', 'Body ID', 'Mapped Body ID']].copy()

    return df_combined


def split_train_val(df):
    train_split = 0.8

    # unrelated - 36545
    # agree - 3678
    # discuss - 8909
    # disagree - 840

    class_count = np.unique(df['Stance'], return_counts=True)[1].astype(np.int32)
    train_count = (class_count * train_split).astype(int)

    df_unrelated = df.loc[df['Stance'] == 0]
    df_agree = df.loc[df['Stance'] == 1]
    df_discuss = df.loc[df['Stance'] == 2]
    df_disagree = df.loc[df['Stance'] == 3]

    df_train = pd.concat(
        [df_unrelated.iloc[:train_count[0]], df_agree.iloc[:train_count[1]], df_discuss.iloc[:train_count[2]],
         df_disagree.iloc[:train_count[3]]], ignore_index=True)

    df_val = pd.concat(
        [df_unrelated.iloc[train_count[0]:], df_agree.iloc[train_count[1]:], df_discuss.iloc[train_count[2]:],
         df_disagree.iloc[train_count[3]:]], ignore_index=True)

    return df_train, df_val


def compute_cos_score(df, headline_embedding, body_embedding):
    score = []

    for idx, row in df.iterrows():
        score.append(util.cos_sim(headline_embedding[row['Headline ID']], body_embedding[[row['Mapped Body ID']]])[0][0].item())

    return np.float32(score)


stance_dict = {
    'unrelated': 0,
    'agree': 1,
    'discuss': 2,
    'disagree': 3
}

predictions_dict = {v: k for k, v in stance_dict.items()}

# sample = ['This is text', 'This is not text', 'How does this work', 'AAAHHHHH']
