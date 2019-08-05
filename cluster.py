from sklearn.cluster import k_means_
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import numpy as np


def k_means(sparse_data, k):
    def euc_dist(x, y=None, Y_norm_squared=None, squared=False):
        return cosine_similarity(x, y)

    k_means_.euclidean_distances = euc_dist

    ss = StandardScaler(with_mean=False)
    sparse_data = ss.fit_transform(sparse_data)
    model = k_means_.KMeans(n_clusters=k, n_jobs=20, random_state=3425)
    _ = model.fit(sparse_data)
    return model.labels_


def agg_(sparse_data, k):
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


def average_embed(path):

    sentences, words = get_vocab(path, 'word2vec/testdata.xlsx')
    word2ids = get_vec(path, 'wiki-news-300d-1M.vec', words)
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

        embed = embed.tolist()
        embedding.append(embed)
    return sentences, embedding


if __name__ == '__main__':

    PATH = '/Users/george/Downloads'
    k_means_cls = 10
    agg_cls = 1000
    wfile = open(os.path.join('D:/桌面', 'cls.txt'), 'w', encoding='utf-8', newline='\n')
    words = {word.strip() for word in open(os.path.join('E:/data/sources', 'vocabulary.txt'), encoding='utf-8')}
    word2vec = get_vec('E:/data/工作文档/embeddings', 'normal_cbow.tsv', words)
    ids = [vec for _, vec in word2vec.items()]
    k_means_model = KMeans(k_means_cls)
    df = pd.DataFrame()
    df['cls'] = k_means_model.fit(ids).labels_
    df['vec'] = ids
    for label in df.groupby('cls'):
        print(f"当前：{label[0]} 批次， 共 {k_means_cls} 批次")
        agg_label = agg_([i for i in label[-1]['vec']], agg_cls)
        label[-1]['cls2'] = agg_label
        for label2 in label[-1].groupby('cls2'):
            wfile.writelines(f"{label[0]}\t{label2[0]}\t{' '.join([[k for k, v in word2vec.items() if v == i][0] for i in label2[-1]['vec']])}\n")
    wfile.close()
    # for kmeans_cls in df.groupby('vector'):
    #     for index in range(len(agg_(kmeans_cls[-1]['vector'], agg_cls))):
    #         wfile.writelines()

    # sentences, embedding = average_embed(PATH)
    # for k_means_label in k_means(embedding, k_means_cls):
    #     print(k_means_label)


