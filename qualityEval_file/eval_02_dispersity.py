
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
import matplotlib.pylab as plt
from time import strftime, localtime

# 图表中显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']

def IsNum(str):
    s=str.split('.')
    if len(s)>2:
        return False
    else:
        for si in s:
            if not si.isnumeric():
                return False
        return True

# 读取句向量存放在数组中
def read_vector(filepath):
    list_sim = []
    for line in tqdm(open(filepath, "r", encoding="UTF-8")):
        line = line.strip('\n')
        line = line.replace("[", "")
        line = line.replace("]", "")
        line = line.split(",")
        list_sim.append(np.array(line).astype(float))
    return np.array(list_sim).astype(float)
    # return list_sim


"""
    实现最大间隙的指标
"""
def get_max_gap(filepath):
    list_vector = read_vector(filepath)
    distances_cosine = pairwise_distances(list_vector, metric="cosine")
    pro = np.amax(distances_cosine, axis=1)
    max_gap = 0
    for i in tqdm(pro):
        max_gap = max_gap + (1 - i)
    max_gap = -(max_gap/len(pro))
    print("总分散度：", max_gap)
    return max_gap

"""
    实现最大间隙的指标 2.0 
    读取每一个向量去计算和全部向量的相似度，再将总分散度进行相加，不用一次性计算所有的
"""
def get_max_gap_large_data(filepath):
    list_vector = read_vector(filepath)
    max_gap = 0
    num_start = 0
    num_end = 10000
    pro_one = []
    while num_end <= len(list_vector):
        print(num_end)
        # vector = []
        # vector.append(list_vector[num_start:num_end])
        vector = list_vector[num_start:num_end]
        distances_cosine_one = pairwise_distances(X=vector, Y=list_vector, metric="cosine")
        pro_one = np.append(pro_one, np.amax(distances_cosine_one, axis=1))
        num_start = num_end
        num_end = num_end + 10000
        if num_end > len(list_vector):
            # vector_end = []
            # vector_end.append(list_vector[num_end-10000:])
            vector_end = list_vector[num_end - 10000:]
            distances_cosine_end = pairwise_distances(X=vector_end, Y=list_vector, metric="cosine")
            pro_one = np.append(pro_one, np.amax(distances_cosine_end, axis=1))
    for i in tqdm(pro_one):
        max_gap = max_gap + (1 - i)
    max_gap = -(max_gap / len(pro_one))
    print("大小：", len(pro_one))
    print("总分散度：", max_gap)
    return max_gap

"""
    实现最大间隙的指标 3.0 
    读取每一个向量去计算和全部向量的相似度，再将总分散度进行相加，不用一次性计算所有的
"""
def get_max_gap_large_data_01(filepath):
    list_vector = read_vector(filepath)
    print("开始！！！")
    max_gap = 0
    num_start = 0
    num_end = 19
    pro_one = []
    while num_start < len(list_vector):
        # print(num_end)
        vector = list_vector[num_start:num_end]
        start = 0
        end = 19
        pro_one_temp = []
        while start < len(list_vector):
            result = list_vector[start:end]
            distances_cosine_one = pairwise_distances(X=vector, Y=result, metric="cosine")
            if len(pro_one_temp) == 0:
                pro_one_temp = np.amax(distances_cosine_one, axis=1)
            else:
                temp = np.amax(distances_cosine_one, axis=1)
                for i in range(len(pro_one_temp)):
                    if pro_one_temp[i] < temp[i]:
                        pro_one_temp[i] = temp[i]
            start = end
            if end + 19 < len(list_vector):
                end = end + 19
            else:
                end = len(list_vector)
        pro_one = np.append(pro_one, pro_one_temp)
        num_start = num_end
        if num_end + 19 < len(list_vector):
            num_end = num_end + 19
        else:
            num_end = len(list_vector)
    for i in tqdm(pro_one):
        max_gap = max_gap + (1 - i)
    max_gap = -(max_gap / len(pro_one))
    print("大小：", len(pro_one))
    print("总分散度：", max_gap)
    return max_gap
"""
    实现方式：数据合成生成1个错误句4万条,2个错误句6万条；
            使用sklearn的pairwise_distances来计算余弦距离矩阵
"""

if __name__ == "__main__":
    # file = "nlpcc_sample_100000_test_2000"
    # file = "nlpcc_sample_100000_eda_10_four"
    # file = "CSC_13_14_15_correct_ocr_asr_01"
    # filepath = "data_vector_one/"+file+"_one_vector.txt"
    file = "40000"
    filepath = "20221210_quality/data_vector_one/nlpcc_new_"+file+"_one_vector.txt"
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    # get_max_gap(filepath)
    # get_max_gap_large_data(filepath)
    get_max_gap_large_data_01(filepath)
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    # get_cosine_distance_distribute(filepath)
    #----------------------0.实现max_gap
    # list_vector = read_vector("data_vector/CSC_13_14_15_all_rule_10_no_re_vector_pooler.txt")
    # # # cosine euclidean
    # distances_cosine = pairwise_distances(list_vector[:3], metric="cosine")
    # print(distances_cosine)
    # # axis=0 代表列 , axis=1 代表行
    # idx = np.argmax(distances_cosine, axis=1)
    # print(idx)
    # pro = np.amax(distances_cosine, axis=1)
    # print(pro)
    # print(type(pro))
    # print(len(pro))
    # print(pro[0])
    # # ---------------------1. 验证余弦距离和余弦相似度---正确
    # x = np.array([list_vector[0]])
    # y = np.array([list_vector[1]])
    # # 余弦相似度
    # simi = cosine_similarity(x, y)
    # print('cosine similarity:', simi)
    # # 余弦距离 = 1 - 余弦相似度
    # dist = paired_distances(x, y, metric='cosine')
    # print('cosine distance:', dist)
    # print('cosine similarity:', 1 - dist)
    # first = list_vector[0]
    # second = list_vector[1]
    # d1 = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    # print('cosine similarity:', d1)
    # 测试pairwise_distances,计算成对的余弦相似度
    # a = [[1, 3, 2], [2, 2, 1], [2, 2, 1], [2, 2, 1]]
    # b = [[1, 3, 2], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1], [2, 2, 1]]
    # distances_cosine = pairwise_distances(b, metric="cosine")
    # # print(distances_cosine)
    # distances_cosine_a = pairwise_distances(X=a[0:2], Y=b, metric="cosine")
    # print(a[0:2])
    # # print(distances_cosine_a)
    # pro = np.amax(distances_cosine_a, axis=1)
    # print(pro)