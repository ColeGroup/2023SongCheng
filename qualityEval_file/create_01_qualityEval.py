from qualityEval_file.eval_01_00_hnsw_generate import get_hnsw
from qualityEval_file.eval_01_01_hnsw_coverage import get_coverage
from qualityEval_file.eval_02_dispersity import get_max_gap_large_data_01
from qualityEval_file.eval_03_00_min_support import get_min_support
from qualityEval_file.eval_03_01_get_support_file import get_support_file
from time import strftime, localtime
import os
from tqdm import tqdm
import numpy as np
import math

# 读取句向量存放在数组中
def read_vector(filepath):
    list_sim = []
    for line in tqdm(open(filepath, "r", encoding="UTF-8")):
        line = line.strip('\n')
        line = line.replace("[", "")
        line = line.replace("]", "")
        line = line.split(",")
        list_sim.append(np.array(line).astype(float))
    print(np.shape(list_sim))
    return np.array(list_sim).astype(float)
    # return list_sim

"""
    互覆盖度计算--->传入向量文件名称，纠错任务名称
"""
def get_result_coverage(fileName_vector,task):
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    file_hnsw = fileName_vector + ".hnsw"
    """
        本地向量文件路径
    """
    # file_CGED = "D:/java_project/data_vector/" + fileName_vector
    file_CGED = "/songcheng_file/data_vector/" + fileName_vector
    get_hnsw(file_CGED, file_hnsw)
    coverage = get_coverage(file_hnsw,task)
    os.remove(file_hnsw)
    print("互覆盖度：", coverage)
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return coverage

"""
    总分散度计算--->传入向量文件
"""
def get_result_dispersity(fileName_vector):
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    # file_CGED = "D:/java_project/data_vector/" + fileName_vector
    file_CGED = "/songcheng_file/data_vector/" + fileName_vector
    list_sim = read_vector(file_CGED)
    current_path = "/songcheng_file/data_vector"
    fileName_vector_new = current_path + "/data_one_vector/" + fileName_vector
    # list_test = read_vector("test20230401_all_vector.txt")
    # 在当前目录下创建向量文件，将拼接向量转换为单个向量存储
    for vector in list_sim:
        file_one = open(fileName_vector_new, 'a', encoding="UTF-8")
        file_one.write(str(vector[:768].tolist()) + '\n')
        file_one.write(str(vector[768:].tolist()) + '\n')
        file_one.close()
    dispersity = get_max_gap_large_data_01(fileName_vector_new)
    print("总分散度：", dispersity)
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return dispersity

"""
    自支撑度计算--->传入向量文件
"""
def get_result_support(fileName_vector):
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    file = fileName_vector.replace(".txt", "")
    get_support_file(file)
    support = get_min_support(file)
    current_path = "/songcheng_file/data_vector"
    file_768_vector_birth_category = current_path + "/data_category_birch/" + file + "_one_vector_birch.csv"
    file_5_vector_ISOMAP = current_path + "/data_category_ISOMAP/" + file + "_one_vector_isomap.csv"
    file_5_vector_birth_category = current_path + "/data_category_ISOMAP/" + file + "_one_vector_isomap_to_5.csv"
    file_5_vector_greedy = current_path + "/data_cosine_list_greedy/" + file + "_one_vector_isomap_to_5_greedy.csv"
    os.remove(file_768_vector_birth_category)
    os.remove(file_5_vector_ISOMAP)
    os.remove(file_5_vector_birth_category)
    os.remove(file_5_vector_greedy)
    print("自支撑度：", support)
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return support

"""
    质量融合计算--->传入向量文件和纠错任务
"""
def get_result_quality(fileName_vector, task):
    current_path = "/songcheng_file/data_vector"
    list_quality = {}
    coverage = get_result_coverage(fileName_vector,task)
    dispersity = get_result_dispersity(fileName_vector)
    support = get_result_support(fileName_vector)
    if task == "拼写纠错":
        new_coverage = math.pow(coverage, 5)
        print("1")
        print(task)
    else:
        new_coverage = math.pow(coverage, 2)
        print("2")
        print(task)
    quality = new_coverage*dispersity*support
    # 删除当前目录下由拼接向量转换的单个向量文件
    os.remove(current_path + "/data_one_vector/" + fileName_vector)
    file_CGED = "/songcheng_file/data_vector/" + fileName_vector
    os.remove(file_CGED)
    print("质量融合：", quality)
    list_quality['coverage'] = coverage
    list_quality['dispersity'] = dispersity
    list_quality['support'] = support
    list_quality['quality'] = quality
    return list_quality

if __name__ == "__main__":
    fileName_vector = "test20230402_all_vector.txt"
    # fileName_vector = "test20230402_one_vector.txt"
    task = "拼写纠错"
    # 1. 互覆盖度计算
    # get_result_coverage(fileName_vector, task)
    # 2. 总分散度计算
    # get_result_dispersity(fileName_vector)
    # 3. 自支撑度计算
    get_result_support(fileName_vector)
    # 4. 质量融合计算
    # get_result_quality(fileName_vector, task)



