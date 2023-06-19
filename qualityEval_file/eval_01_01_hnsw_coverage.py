# coding:utf-8

import numpy as np
import nmslib
import datetime
from functools import wraps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from time import strftime, localtime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def func_execute_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        res = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        duration_time = (end_time - start_time).microseconds // 1000
        # print("execute function %s, elapse time %.4f ms" % (func.__name__, duration_time))
        return res
    return wrapper

# ef_search=300 50000
def load_indexer(index_path, ef_search=300):
    """
    加载构建好的向量索引文件
    :param index_path: 索引文件地址
    :param ef_search: 查询结果参数
    :return:
    """
    indexer = nmslib.init(method="hnsw", space="cosinesimil")
    indexer.loadIndex(index_path)
    #  setQueryTimeParams--设置 knnQuery 中使用的参数。
    indexer.setQueryTimeParams({"efSearch": ef_search})
    return indexer


@func_execute_time
def search_vec_top_n(indexer, vecs, top_n=7, threads=2):
    """
    使用构建好的索引文件完成向量查询
    :param indexer: 索引
    :param vecs: 待查询向量
    :param top_n: 返回前top_n个查询结果
    :param threads:
    :return:
    """
    # knnQueryBatch--对索引执行多个查询，通过线程池分配工作
    neighbours = indexer.knnQueryBatch(vecs, k=top_n, num_threads=threads)
    # print(neighbours)
    return neighbours


# 读取句向量存放在数组中
def read_vector(filepath):
    list_sim = []
    for line in tqdm(open(filepath, "r", encoding="UTF-8")):
        line = line.strip('\n')
        line = line.replace("[", "")
        line = line.replace("]", "")
        # print(line.split(","))
        list_sim.append(line.split(","))
    return np.array(list_sim).astype(float)



"""
    1.0 调用此方法获取互覆盖度--A中每一个元素去计算B
"""
def get_hnsw_generate(filepath_vector):
    list_vector = read_vector(filepath_vector)
    # 经过微调的句向量生成的hnsw图
    # indexer = load_indexer("nlpcc2018_lang8_CSC_no_repeat_vector_1000.hnsw")
    indexer = load_indexer("../nlpcc2018_lang8_CGED_all_no_repeat_vector_1000.hnsw")
    list_sim = []
    for v in list_vector:
        list_sim.append(v)
    list_result = search_vec_top_n(indexer, list_sim, top_n=5)
    # print(list_result)
    sum_res = 0
    for i in tqdm(list_result):
        sum_res = sum_res + i[1][0]
    coverage = sum_res/len(list_vector)
    print("互覆盖度：", coverage)
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))


"""
    2.0 调用此方法获取互覆盖度---B中每一个元素去计算A,A覆盖了B
"""
def get_coverage(file_hnsw, task):
    indexer = load_indexer(file_hnsw)
    sum_res = 0
    count = 0
    file_task = ""
    current_path = os.path.dirname(__file__)
    if task is "语法纠错":
        file_task = current_path + "/data_vector_all/nlpcc_sample_100000_test_2000_all_vector.txt"
    else:
        file_task = current_path + "/data_vector_all/CSC_test15_correct_error_all_vector.txt"
    with open(file_task, "r", encoding="UTF-8") as file:
        for line in tqdm(file):
            list_sim = []
            line = line.strip('\n')
            line = line.replace("[", "")
            line = line.replace("]", "")
            list_sim.append(line.split(","))
            list_sim = np.array(list_sim).astype(float)
            list_result = search_vec_top_n(indexer, list_sim, top_n=1)
            res = [x[1] for x in list_result]
            sum_res = sum_res + (1 - res[0][0])
            count = count + 1
    coverage = sum_res/count
    return coverage



"""
    计算覆盖度测试
"""
if __name__ == '__main__':
    # file = "nlpcc_sample_100000_eda_10_four"
    file = "CSC_13_14_15_correct_ocr_asr_01"
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    file_hnsw = "hnsw_generate/"+file+"_all_vector_300.hnsw"
    # get_coverage(file_hnsw)
