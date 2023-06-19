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


# 遍历0到0.5之间，距离间隔0.02，使用贪心算法获取最优的集合覆盖个数
def greedy_cover(stations:dict):
    """
    使用贪心算法解决集合覆盖问题：选择最少的广播台，让所有的地区都可以接收到信号
    :param stations:
    :return:
    """
    # 创建一个set存放需要覆盖但还未覆盖的地区
    not_cover = set()
    for v in stations.values():
        for s in v:
            not_cover.add(s)
    selects = []  # 存放我们选择的电台
    while True:
        # 首先，选择覆盖了最多未覆盖地区的电台
        max_key = ''
        max_num = 0
        for k in stations.keys():
            intersection = not_cover.intersection(stations[k])
            if len(intersection) > max_num:
                max_key = k
                max_num = len(intersection)
        selects.append(max_key)
        # 然后，将选择电台覆盖的地区从not_cover中移除
        for e in stations[max_key]:
            if e in not_cover:
                not_cover.remove(e)
        # 如果，not_cover未空即所有地区已覆盖，则可以结束算法
        if len(not_cover) == 0:
            break
    return selects

# 获取当前聚类中句子的cosine_list
def get_cosine_list(length_gap,distances_cosine):
    for i in range(len(distances_cosine)):
        similary_index = distances_cosine[i]
        # 降序排列的索引
        indies = np.argsort(similary_index)
        for j in indies:
            # print("0.004长度:", len(length_gap[0.004]))
            # print(similary_index[j])
            if similary_index[j] > 0.11:
                break
            if similary_index[j] <= 0.112:
                 length_gap[0.112][i].append(j)
            if similary_index[j] <= 0.108:
                 length_gap[0.108][i].append(j)
            if similary_index[j] <= 0.104:
                 length_gap[0.104][i].append(j)
            if similary_index[j] <= 0.100:
                 length_gap[0.100][i].append(j)
            if similary_index[j] <= 0.096:
                 length_gap[0.096][i].append(j)
            if similary_index[j] <= 0.092:
                 length_gap[0.092][i].append(j)
            if similary_index[j] <= 0.088:
                 length_gap[0.088][i].append(j)
            if similary_index[j] <= 0.084:
                length_gap[0.084][i].append(j)
            if similary_index[j] <= 0.080:
                 length_gap[0.080][i].append(j)
            if similary_index[j] <= 0.076:
                length_gap[0.076][i].append(j)
            if similary_index[j] <= 0.072:
                 length_gap[0.072][i].append(j)
            if similary_index[j] <= 0.068:
                 length_gap[0.068][i].append(j)
            if similary_index[j] <= 0.064:
                 length_gap[0.064][i].append(j)
            if similary_index[j] <= 0.060:
                 length_gap[0.060][i].append(j)
            if similary_index[j] <= 0.056:
                 length_gap[0.056][i].append(j)
            if similary_index[j] <= 0.052:
                 length_gap[0.052][i].append(j)
            if similary_index[j] <= 0.048:
                 length_gap[0.048][i].append(j)
            if similary_index[j] <= 0.044:
                 length_gap[0.044][i].append(j)
            if similary_index[j] <= 0.040:
                 length_gap[0.040][i].append(j)
            if similary_index[j] <= 0.036:
                 length_gap[0.036][i].append(j)
            if similary_index[j] <= 0.032:
                 length_gap[0.032][i].append(j)
            if similary_index[j] <= 0.028:
                 length_gap[0.028][i].append(j)
            if similary_index[j] <= 0.024:
                 length_gap[0.024][i].append(j)
            if similary_index[j] <= 0.020:
                 length_gap[0.020][i].append(j)
            if similary_index[j] <= 0.016:
                 length_gap[0.016][i].append(j)
            if similary_index[j] <= 0.012:
                 length_gap[0.012][i].append(j)
            if similary_index[j] <= 0.008:
                 length_gap[0.008][i].append(j)
            if similary_index[j] <= 0.004:
                 length_gap[0.004][i].append(j)
    # print("0.004长度:", len(length_gap[0.004]))
    # print("dd", length_gap[0.012])
    return length_gap

# 获取贪心集合覆盖计算结果的返回值
def get_dict(list_df, index):
    distances_cosine = pairwise_distances(list_df, metric="cosine")
    # 生成length_gap的字典----第一层0.004:{0:[],1:[]...}
    length_gap = {}
    # 字典外层
    i = 0
    while i < 0.11:
        i = Decimal(i) + Decimal(0.004)
        i = Decimal(i).quantize(Decimal("0.000"))
        # 字典内层
        dict_vector = {}
        for j in range(len(list_df)):
            dict_vector[j] = []
        length_gap[float(i)] = dict_vector
    # print(length_gap)
    # length_gap[0.008][0].append(2)
    # print("11", length_gap[0.004])
    # print("12", length_gap[0.008])
    # 给每个句子索引添加集合
    gap_result = get_cosine_list(length_gap, distances_cosine)

    dict_category = {}
    dict_category[index] = []
    for i in gap_result:
        # 贪心算法的集合覆盖
        selects = greedy_cover(gap_result[i])
        dict_category[index].append(len(selects))
    return dict_category

# 将集合覆盖结果存放到csv文件中
def to_greedy_category_to_csv(filepath_category, filepath_greedy):

    list_gap = []
    a = 0
    while a < 0.11:
        a = Decimal(a) + Decimal(0.004)
        a = Decimal(a).quantize(Decimal("0.000"))
        list_gap.append(a)

    # path_1 = "data_category/CSC_5wei_17000_category.csv"
    df2 = pd.read_csv(filepath_category, nrows=0)
    # 获取不同聚类的标签['0', '1', '2', '3', '4',,,,]
    list_index = [x for x in df2.columns]
    list_length = []
    # 之前写的有问题,list_length是对类别的循环添加值,应该是对间隔的循环添加值
    for j in range(len(list_gap)):
        list_length.append(0)
    # print(list_length)
    for i in tqdm(list_index):
        list_df = get_category_vector(filepath_category, i)
        dict_category = get_dict(list_df, i)
        # print(dict_category[i])
        # 不同类的集合覆盖结果相加--验证过
        list_length = list(map(lambda x: x[0]+x[1], zip(list_length, dict_category[i])))
        # print("相加:", list_length)
        # print(len(list_length))

    length_gap = []
    c = 0
    while c < 0.11:
        c = Decimal(c) + Decimal(0.004)
        c = Decimal(c).quantize(Decimal("0.000"))
        length_gap.append(c)
    # path = "data_cosine_list_greedy/greedy_CSC_5wei_17000_category.csv"
    df = pd.DataFrame(columns=length_gap)
    df.to_csv(filepath_greedy, index=False)
    df.loc[len(df)] = list_length
    df.to_csv(filepath_greedy, mode='a', header=False, index=False)


if __name__ == "__main__":
    filepath_category = "data_category_ISOMAP/CGED_correct_all_eda_30_vector_pooler_isomap_to_5.csv"
    # filepath_category = "data_reduce_vector/CSC_all_vector_pooler_reduce_MDS.txt"
    filepath_greedy = "data_cosine_list_greedy/CGED_correct_all_eda_30_vector_pooler_isomap_to_5_greedy.csv"
    to_greedy_category_to_csv(filepath_category, filepath_greedy)

