import pandas as pd
from sklearn.cluster import Birch
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import torch.nn.functional as F
import datetime
import re
from decimal import Decimal
from tqdm import tqdm
from sklearn.manifold import MDS,Isomap
from time import strftime, localtime

#  读取csv文件---聚类结果文件，包含类别和类别下的向量
def get_category_vector(filepath,index):
    # filepath = "data_category/7wei_200000_category_1.csv"
    # index = "0"
    df = pd.read_csv(filepath, usecols=[index])
    list_df = []
    print("1")
    for i in range(len(df)):
        print("2")
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
            # print(line)
            # for i in line:
            #     print(i)
            #     # print(float(i))
            # line = [float(x) for x in line]
            list_df.append(np.array(line).astype(float))
    # print("0标签长度:", len(list_df))
    # print(list_df)
    return list_df


# 归一化数据
def transform_data(data):
    data_nomal = F.normalize(torch.Tensor(data), p=2, dim=1)
    return data_nomal

# 通过MDS对每一个类别进行降维
def get_dict_MDS(list_df):
    distances_cosine = pairwise_distances(transform_data(list_df), metric="euclidean")
    # clf2 = MDS(n_components=5, dissimilarity="precomputed")
    # result_MDS_vector = clf2.fit_transform(distances_cosine)
    # isomap = Isomap(n_components=5, metric="precomputed", n_neighbors=100)
    isomap = Isomap(n_components=5, metric="precomputed", n_neighbors=80)
    new_X_isomap = isomap.fit_transform(distances_cosine)
    return new_X_isomap

# 将类别对应向量集合，存放在csv中
def get_category_csv(dict_birch, filepath):
    category = []
    for i in range(len(dict_birch)):
        category.append(i)
    list_category = []
    for i in dict_birch:
        list_category.append(dict_birch[i])
    df = pd.DataFrame(index=category, data=list_category)
    df.T.to_csv(filepath, index=False)

# 调用此方法实现分类别的降维
def category_to_csv(filepath_category, filepath_category_MDS):
    # path_1 = "data_category/CSC_5wei_17000_category.csv"
    df2 = pd.read_csv(filepath_category, nrows=0)
    # 获取不同聚类的标签['0', '1', '2', '3', '4',,,,]
    list_index = [x for x in df2.columns]
    # print("类别：", list_index)
    dict_MDS = {}
    for j in range(len(list_index)):
        dict_MDS[str(j)] = []
    for i in tqdm(list_index):
        list_df = get_category_vector(filepath_category, i)
        print(len(list_df))
        result_MDS_vector = get_dict_MDS(list_df)
        for k in result_MDS_vector:
            dict_MDS[i].append(k)
    get_category_csv(dict_MDS, filepath_category_MDS)



"""
    分类别进行降维
"""

if __name__ == "__main__":
    # print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    # filepath_category = "data_category_birch/CGED_correct_all_eda_30_vector_pooler_birch.csv"
    # filepath_MDS = "data_category_ISOMAP/CGED_correct_all_eda_30_vector_pooler_isomap.csv"
    # category_to_csv(filepath_category, filepath_MDS)
    # print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    get_category_vector("2023040538e1312e_one_vector_birch.csv","0")

