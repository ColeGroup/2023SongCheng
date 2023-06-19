from sklearn.cluster import Birch,MiniBatchKMeans
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import torch.nn.functional as F
import datetime
import pandas as pd
import copy

# 读取句向量存放在数组中
def read_vector(filepath):
    list_sim = []
    for line in open(filepath, "r", encoding="UTF-8"):
        line = line.strip('\n')
        line = line.replace("[", "")
        line = line.replace("]", "")
        line = line.split(",")
        list_sim.append(np.array(line).astype(float))
    # return np.array(list_sim).astype(float)
    return list_sim
# 归一化数据
def transform_data(data):
    data_nomal = F.normalize(torch.Tensor(data), p=2, dim=1)
    return data_nomal

# 获取聚类类别对应的索引向量集合，生成一个类别对应向量集合
def get_birch_dict(filepath,n_clusters):
    random_normal = read_vector(filepath)
    data_nomal = transform_data(random_normal)
    # 0.5-->0.6
    # CSC--设置
    brc = Birch(n_clusters=n_clusters, threshold=0.5, branching_factor=50).fit(data_nomal)
    # print(brc.labels_[:50])
    # nlpcc--10万条设置
    # brc = Birch(n_clusters=None, threshold=0.9, branching_factor=200).fit(data_nomal)
    category = len(brc.subcluster_labels_)
    # category = n_clusters
    print("category", category)
    dict_birch = {}
    for i in range(category):
        dict_birch[i] = []
    for j, value in enumerate(brc.labels_):
        dict_birch[value].append(data_nomal[j].numpy())
    return dict_birch



# 将类别对应向量集合，存放在csv中
def get_category_csv(dict_birch, filepath):
    category = []
    for i in range(len(dict_birch)):
        category.append(i)
    list_category = []
    # print(dict_birch[0][0])
    for i in dict_birch:
        list_category.append(dict_birch[i])
    # print(list_category[0][0])
    # print(list_category[0])
    # path = "data_category/CSC_5wei_17000_category.csv"

    df = pd.DataFrame(index=category, data=list_category)
    df.T.to_csv(filepath, index=False)


def get_birch_category(filepath_vector, filepath_birch, n_clusters):
    print("单个向量文件路径", filepath_vector)
    dict_birch = get_birch_dict(filepath_vector, n_clusters)
    # 改----不设置聚类个数，有些类别数量太少
    for i in range(len(dict_birch)):
        if len(dict_birch[i]) < 100:
            for vector in dict_birch[i]:
                if i > 1:
                    dict_birch[i-1].append(vector)
                else:
                    dict_birch[i+1].append(vector)
    for i in range(len(dict_birch)):
        if len(dict_birch[i]) < 100:
            dict_birch.pop(i)
    # 改-------
    print("聚类类别：", len(dict_birch))
    # print(len(dict_birch[0]))
    # print(len(dict_birch[1]))
    for i in dict_birch:
        print("类别中的数量", len(dict_birch[i]))
    get_category_csv(dict_birch, filepath_birch)


"""
    实现数据聚类，分类别存放句向量---尝试先聚类，再降维计算最小支持度
"""
if __name__ == "__main__":
    # n_clusters = 30
    # filepath_vector = "data_vector/CGED_correct_all_eda_30_vector_pooler.txt"
    # filepath_birth = "data_category_birch/CGED_correct_all_eda_30_vector_pooler_birch.csv"
    # get_birch_category(filepath_vector, filepath_birth, n_clusters)
    n_clusters = 10
    filepath_vector = "2023040538e1312e.txt"
    filepath_birth = "data_category_birch/test_birch.csv"
    get_birch_category(filepath_vector, filepath_birth, n_clusters)
    # ---------测试聚类并存放向量
    # filepath = "data_vector/CSC_5wei_17000_vector.txt"
    # dict_birch = get_birch_dict(filepath)
    # print(len(dict_birch))
    # print(len(dict_birch[0]))
    # print(len(dict_birch[1]))
    # get_category_csv(dict_birch)
