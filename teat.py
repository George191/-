from sklearn.cluster import k_means_
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import numpy as np

k_means_cls = 10
agg_cls = 700


def k_means(sparse_data, k=k_means_cls):
    def euc_dist(x, y=None, Y_norm_squared=None, squared=False):
        return cosine_similarity(x, y)

    k_means_.euclidean_distances = euc_dist

    ss = StandardScaler(with_mean=False)
    sparse_data = ss.fit_transform(sparse_data)
    model = k_means_.KMeans(n_clusters=k, n_jobs=20, random_state=3425)
    _ = model.fit(sparse_data)
    return model.labels_


def agg_(sparse_data, k=agg_cls):
    model = AgglomerativeClustering(k)
    model.fit(sparse_data)
    return model.labels_


def get_vocab(path, file):
    sheet = pd.read_excel(os.path.join(path, file))
    sentences = [str(sentence).split() for sentence in sheet['Term']]
    words = {word for sentence in sentences for word in sentence}
    return sentences, words


def get_vec(path, file, words):
    w2i = {}
    with open(os.path.join(path, file), 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split()
            if line[0] in words:
                w2i[line[0]] = [float(i) for i in line[1:]]
    return w2i





def average_embed(path, file):
    sentences, words = get_vocab(PATH, 'word2vec/testdata.xlsx')
    word2ids = get_vec(PATH, 'wiki-news-300d-1M.vec', words)
    sentences = [[word for word in sentence if word in word2ids] for sentence in sentences]
    embedding = []
    for sentence in sentences:
        embed = []
        for i in range(0, 300):
            embed.append(0)
        embed = np.array(embed)  # 转化为array便于加减乘除
        count = 0
        for word in sentence:
            try:
                count = count + 1
                embed = word2ids[word] + embed  # 向量对应值相加
            except:
                continue

        if count != 0:
            embed = embed / count

        embed = embed.tolist()  # 变回List容易增加项 和 SVM分类
        embedding.append(embed)
    return sentences, embedding


PATH = '/Users/george/Downloads'


