from sklearn.cluster import Birch,MiniBatchKMeans
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import torch.nn.functional as F
import datetime
import pandas as pd
import re
from tqdm import tqdm

# 归一化数据
def transform_data(data):
    data_nomal = F.normalize(torch.Tensor(data), p=2, dim=1)
    return data_nomal

# 获取聚类类别对应的索引向量集合，生成一个类别对应向量集合
def get_birch_dict(random_normal,n_clusters):
    data_nomal = transform_data(random_normal)
    # 0.5-->0.6
    brc = Birch(n_clusters=n_clusters, threshold=0.5, branching_factor=50).fit(data_nomal)
    # print(brc.labels_[:50])
    # brc = Birch(n_clusters=None, threshold=0.5, branching_factor=100).fit(data_nomal)
    # category = len(brc.subcluster_labels_)
    category = n_clusters
    dict_birch = {}
    for i in range(category):
        dict_birch[i] = []
    for j, value in enumerate(brc.labels_):
        dict_birch[value].append(data_nomal[j].numpy())
    return dict_birch

#  读取csv文件---聚类结果文件，包含类别和类别下的向量
def get_category_vector(filepath,index):
    # filepath = "data_category/7wei_200000_category_1.csv"
    # index = "0"
    df = pd.read_csv(filepath, usecols=[index])
    list_df = []
    for i in range(len(df)):
        if pd.isna(df[index][i]):
            break
        else:
            line = df[index][i]
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace("\n", "")
            line = line.strip(" ")
            line = re.sub(' +', ' ', line)
            line = line.split(" ")
            line = [float(x) for x in line]
            list_df.append(line)
    # print("0标签长度:", len(list_df))
    # print(list_df)
    return list_df

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

# 调用此方法实现分类别的降维
def to_5wei_category_to_csv(filepath_in, filepath_out, n_clusters):
    # path_1 = "data_category/CSC_5wei_17000_category.csv"
    df2 = pd.read_csv(filepath_in, nrows=0)
    # 获取不同聚类的标签['0', '1', '2', '3', '4',,,,]
    list_index = [x for x in df2.columns]
    print("5wei重新聚类类别：", list_index)
    list_vector = []
    for i in tqdm(list_index):
        list_df = get_category_vector(filepath_in, i)
        for k in list_df:
            list_vector.append(k)

    dict_birch = get_birch_dict(list_vector, n_clusters)
    get_category_csv(dict_birch, filepath_out)

# 将5维数据合并再聚类
if __name__ == "__main__":
    n_clusters = 30
    filepath_in_csv = "data_category_ISOMAP/CGED_correct_all_eda_30_vector_pooler_isomap.csv"
    filepath_out_csv = "data_category_ISOMAP/CGED_correct_all_eda_30_vector_pooler_isomap_to_5.csv"
    to_5wei_category_to_csv(filepath_in_csv, filepath_out_csv, n_clusters)