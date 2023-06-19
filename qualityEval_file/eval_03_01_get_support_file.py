
from qualityEval_file.eval_03_02_birch_category_no_reduce import get_birch_category
from qualityEval_file.eval_03_03_birch_dimensionality_ISOMAP import category_to_csv
from qualityEval_file.eval_03_04_5wei_to_5wei_birch import to_5wei_category_to_csv
from qualityEval_file.eval_03_05_birch_greedy import to_greedy_category_to_csv

# from eval_03_02_birch_category_no_reduce import get_birch_category
# from eval_03_03_birch_dimensionality_ISOMAP import category_to_csv
# from eval_03_04_5wei_to_5wei_birch import to_5wei_category_to_csv
# from eval_03_05_birch_greedy import to_greedy_category_to_csv

from time import strftime, localtime
import os

def get_support_file(file):
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    current_path = "/songcheng_file/data_vector"
    # file = "CSC_13_14_15_correct_ocr_asr_01"
    file_768_vector = current_path + "/data_one_vector/" + file + ".txt"
    file_768_vector_birth_category = current_path + "/data_category_birch/" + file + "_one_vector_birch.csv"
    file_5_vector_ISOMAP = current_path + "/data_category_ISOMAP/" + file + "_one_vector_isomap.csv"
    file_5_vector_birth_category = current_path + "/data_category_ISOMAP/" + file + "_one_vector_isomap_to_5.csv"
    file_5_vector_greedy = current_path + "/data_cosine_list_greedy/" + file + "_one_vector_isomap_to_5_greedy.csv"
    # 设定聚类的数量，用于isomap中，进行划分   30 60
    n_clusters = 30
    # 1. 将768维的向量通过聚类分类别
    print("1.将768维的向量通过聚类分类别:", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    get_birch_category(file_768_vector, file_768_vector_birth_category, n_clusters)
    # 2. 将每一个类别的768维向量降维到5维---在降维中因为设置的isomap的邻近点是200，所有每一个类别下的数量都要大于200
    print("2.将每一个类别的768维向量降维到5维:", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    category_to_csv(file_768_vector_birth_category, file_5_vector_ISOMAP)
    # 3. 合并所有5维向量的类别，并重新进行聚类
    print("3.合并所有5维向量的类别，并重新进行聚类:", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    to_5wei_category_to_csv(file_5_vector_ISOMAP, file_5_vector_birth_category, n_clusters)
    # 4. 对每一个类别的5维向量计算最小支持度
    print("4.对每一个类别的5维向量计算最小支持度:", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    to_greedy_category_to_csv(file_5_vector_birth_category, file_5_vector_greedy)
"""
    生成最小支持度所需要的文件
"""
if __name__ == "__main__":
    # file = "30_three"
    # file_768_vector = "data_vector/CGED_correct_all_rule_" + file + "_no_re_vector_pooler.txt"
    # file_768_vector_birth_category = "data_category_birch/CGED_correct_all_rule_" + file + "_no_re_vector_pooler_birch.csv"
    # file_5_vector_ISOMAP = "data_category_ISOMAP/CGED_correct_all_rule_" + file + "_no_re_vector_pooler_isomap.csv"
    # file_5_vector_birth_category = "data_category_ISOMAP/CGED_correct_all_rule_" + file + "_no_re_vector_pooler_isomap_to_5.csv"
    # file_5_vector_greedy = "data_cosine_list_greedy/CGED_correct_all_rule_" + file + "_no_re_vector_pooler_isomap_to_5_greedy.csv"
    # file = "nlpcc_sample_100000_eda_10_four"
    file = "CSC_13_14_15_correct_ocr_asr_01"
    file_768_vector = "data_vector_one/" + file + "_one_vector.txt"
    file_768_vector_birth_category = "data_category_birch/" + file + "_one_vector_birch.csv"
    file_5_vector_ISOMAP = "data_category_ISOMAP/" + file + "_one_vector_isomap.csv"
    file_5_vector_birth_category = "data_category_ISOMAP/" + file + "_one_vector_isomap_to_5.csv"
    file_5_vector_greedy = "data_cosine_list_greedy/" + file + "_one_vector_isomap_to_5_greedy.csv"
    # 设定聚类的数量，用于isomap中，进行划分   30 60
    n_clusters = 30
    # 1. 将768维的向量通过聚类分类别
    print("1.将768维的向量通过聚类分类别:", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    get_birch_category(file_768_vector, file_768_vector_birth_category, n_clusters)
    # 2. 将每一个类别的768维向量降维到5维---在降维中因为设置的isomap的邻近点是200，所有每一个类别下的数量都要大于200
    print("2.将每一个类别的768维向量降维到5维:", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    category_to_csv(file_768_vector_birth_category, file_5_vector_ISOMAP)
    # 3. 合并所有5维向量的类别，并重新进行聚类
    print("3.合并所有5维向量的类别，并重新进行聚类:", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    to_5wei_category_to_csv(file_5_vector_ISOMAP, file_5_vector_birth_category, n_clusters)
    # 4. 对每一个类别的5维向量计算最小支持度
    print("4.对每一个类别的5维向量计算最小支持度:", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    to_greedy_category_to_csv(file_5_vector_birth_category, file_5_vector_greedy)