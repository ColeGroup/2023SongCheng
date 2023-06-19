# coding:utf-8


import nmslib
import datetime
from functools import wraps
import numpy as np
from tqdm import tqdm

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def func_execute_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        res = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        duration_time = (end_time - start_time).microseconds // 1000
        print("execute function %s, elapse time %.4f ms" % (func.__name__, duration_time))
        return res

    return wrapper


def create_indexer(filepath, file_hnsw,index_thread, m, ef):
    """
    基于数据向量构建索引
    :param vec: 原始数据向量
    :param index_thread: 线程数
    :param m: 自定义的邻居数
    :param ef: 动态索引长度，一般大于自定义的邻居数，小于原始数据向量个数
    :return:
    """
    print("开始")
    index = nmslib.init(method="hnsw", space="cosinesimil")
    print("结束")
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.split(",")
            line = np.array(line).astype(float)
            # print(count)
            index.addDataPoint(count, line)
            count = count + 1
    # addDataPointBatch--将多个数据点添加到索引
    # index.addDataPointBatch(vec)
    # 线程数、最大邻居数、构图过程中动态索引长度、构图后处理的数量和类型：0表示不处理，1和2（2意味着更多的后处理）
    INDEX_TIME_PARAMS = {
        "indexThreadQty": index_thread,
        "M": m,
        "efConstruction": ef,
        "post": 2
    }
    #  createIndex--创建索引    INDEX_TIME_PARAMS--用于索引的可选参数列表   print_progress--创建索引时是否显示进度条
    index.createIndex(INDEX_TIME_PARAMS, print_progress=True)
    #  saveIndex--加载索引，传入索引名称
    index.saveIndex(file_hnsw)


# ef_search=300
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
def search_vec_top_n(indexer, vecs, top_n=7, threads=4):
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
    # fopen = open(filepath, 'r', encoding="UTF-8")  # 返回一个文件对象
    # lines = fopen.readlines()  # 调用文件的 readline()方法
    # print(len(lines))
    list_sim = []
    # for line in lines:
    for line in tqdm(open(filepath, "r", encoding="UTF-8")):
        line = line.strip('\n')
        line = line.replace("[", "")
        line = line.replace("]", "")
        # print(line.split(","))
        list_sim.append(line.split(","))
    return np.array(list_sim).astype(float)


def get_hnsw(file_CGED,file_hnsw):
    create_indexer(file_CGED,file_hnsw, 10, 50, 300)
    # create_indexer(filepath_CGED, 10, 200, 50000)
    # 经过微调的句向量生成的hnsw图
    print("生成结束！！！")


if __name__ == '__main__':
    # file = "nlpcc_sample_100000_rule_30_one"
    file = "CSC_13_14_15_correct_ocr_asr_01"
    file_hnsw = "hnsw_generate/"+file+"_all_vector_300.hnsw"
    filepath_CGED = "data_vector_all/"+file+"_all_vector.txt"

    # list_CGED = read_vector(filepath_CGED)

    # create_indexer(filepath_CGED, 10, 5, 300)
    create_indexer(filepath_CGED, 10, 50, 300)
    # create_indexer(filepath_CGED, 10, 200, 50000)
    # 经过微调的句向量生成的hnsw图
    indexer = load_indexer(file_hnsw)

    print("生成结束！！！")

    # list_sim = []
    # list_sim.append(list_CGED[0])
    # # get_CGED_sentence(2)
    # # 查询是一个二维列表，里面存放需要查询的句子向量
    # list_result = search_vec_top_n(indexer, list_sim, top_n=10)
    # # 返回索引和距离
    # # print(np.shape(list_result))
    # print(list_result)
    # res = [x[1] for x in list_result]
    # print(len(res[0]))
    # print(float(res[0][1]))
