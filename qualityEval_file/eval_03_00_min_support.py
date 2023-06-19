import pandas as pd
from decimal import Decimal
import matplotlib.pylab as plt
import numpy as np
from time import strftime, localtime
import os
# 图表中显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 对CSC和CGED两个验证数据集，获取最小支持度的折线图

def get_line_chart(csv_5wei_b_20,standard_subtract_5wei_CSC_1):

    x_5w_b_20 = []
    y_5w_b_20 = []

    x_5w_sub_csc_1 = []
    y_5w_sub_csc_1 = []



    for i in csv_5wei_b_20:
        x_5w_b_20.append(i)
        y_5w_b_20.append(csv_5wei_b_20[i])
    for i in standard_subtract_5wei_CSC_1:
        x_5w_sub_csc_1.append(i)
        y_5w_sub_csc_1.append(standard_subtract_5wei_CSC_1[i])


    plt.plot(x_5w_b_20, y_5w_b_20, label="5维-400万条")
    plt.plot(x_5w_b_20, y_5w_b_20, 'o')

    plt.plot(x_5w_sub_csc_1, y_5w_sub_csc_1, label="5维-真实数据")
    plt.plot(x_5w_sub_csc_1, y_5w_sub_csc_1, 'o')


    plt.legend()
    # plt.show()


def read_csv_data(filepath):
    # 忽略0.008之前的间隔
    i = 0.004
    # i = 0
    dict_01 = {}
    while i < 0.11:
        i = Decimal(i) + Decimal(0.004)
        i = Decimal(i).quantize(Decimal("0.000"))
        datas = pd.read_csv(filepath, converters={str(i): eval}, usecols=[str(i)])
        for j in datas[str(i)]:
            dict_01[float(i)] = j
    return dict_01


# 标准支持点数量对位减去真实支持度数量
def standard_subtract(dict_standard,dict_real):
    # 忽略0.008之前的间隔
    i = 0.004
    # i = 0
    dict_01 = {}
    while i < 0.11:
        i = Decimal(i) + Decimal(0.004)
        i = Decimal(i).quantize(Decimal("0.000"))
        standard = dict_standard[float(i)]
        real = dict_real[float(i)]
        dict_01[float(i)] = standard - real
    return dict_01


def get_min_support(file):
    current_path = "/songcheng_file/data_vector"
    filepath_5wei_b_20 = current_path +"/data_cosine_list_greedy/greedy_birch_5wei_4000000_1.csv"
    csv_5wei_b_20 = read_csv_data(filepath_5wei_b_20)
    # file = "nlpcc_sample_100000_eda_10_three"
    # file = "CSC_13_14_15_correct_ocr_asr_01"
    filepath_5wei_CSC_1 = current_path +"/data_cosine_list_greedy/"+file+"_one_vector_isomap_to_5_greedy.csv"
    csv_5wei_CSC_1 = read_csv_data(filepath_5wei_CSC_1)

    # standard_subtract_5wei_CSC_1 = standard_subtract(csv_5wei_b_20, csv_5wei_CSC_1)

    # get_line_chart(csv_5wei_b_20, standard_subtract_5wei_CSC_1)
    get_line_chart(csv_5wei_b_20, csv_5wei_CSC_1)
    # 计算散点图下方的面积
    i = 0.004
    # i = 0
    moni_false = []
    csc_true = []
    while i < 0.11:
        i = Decimal(i) + Decimal(0.004)
        i = Decimal(i).quantize(Decimal("0.000"))
        moni_false.append(csv_5wei_b_20[float(i)])
        # csc_true.append(standard_subtract_5wei_CSC_1[float(i)])
        csc_true.append(csv_5wei_CSC_1[float(i)])
    # print("模拟曲线面积:", np.trapz(moni_false))
    # print("真实曲线面积:", np.trapz(csc_true))
    # print("面积差值:", np.trapz(moni_false)-np.trapz(csc_true))
    # print("面积差值比值:", np.trapz(csc_true)/np.trapz(moni_false))
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return np.trapz(csc_true)/np.trapz(moni_false)

if __name__ == "__main__":
    filepath_5wei_b_20 = "data_cosine_list_greedy/greedy_birch_5wei_4000000_1.csv"
    csv_5wei_b_20 = read_csv_data(filepath_5wei_b_20)
    file = "nlpcc_sample_100000_eda_10_three"
    # file = "CSC_13_14_15_correct_ocr_asr_01"
    filepath_5wei_CSC_1 = "data_cosine_list_greedy/"+file+"_one_vector_isomap_to_5_greedy.csv"
    csv_5wei_CSC_1 = read_csv_data(filepath_5wei_CSC_1)

    # standard_subtract_5wei_CSC_1 = standard_subtract(csv_5wei_b_20, csv_5wei_CSC_1)

    # get_line_chart(csv_5wei_b_20, standard_subtract_5wei_CSC_1)
    get_line_chart(csv_5wei_b_20, csv_5wei_CSC_1)
    # 计算散点图下方的面积
    i = 0.004
    # i = 0
    moni_false = []
    csc_true = []
    while i < 0.11:
        i = Decimal(i) + Decimal(0.004)
        i = Decimal(i).quantize(Decimal("0.000"))
        moni_false.append(csv_5wei_b_20[float(i)])
        # csc_true.append(standard_subtract_5wei_CSC_1[float(i)])
        csc_true.append(csv_5wei_CSC_1[float(i)])
    print("模拟曲线面积:", np.trapz(moni_false))
    print("真实曲线面积:", np.trapz(csc_true))
    print("面积差值:", np.trapz(moni_false)-np.trapz(csc_true))
    print("面积差值比值:", np.trapz(csc_true)/np.trapz(moni_false))







