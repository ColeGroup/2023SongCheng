# !/user/bin/env python3
# -*- coding: utf-8 -*-
import flask
import json
from flask import jsonify
from data_enhancement import create_01_rule_based, create_02_EDA, create_03_asr, create_04_ocr
from file_enhancement import create_01_rule_based_file, create_02_EDA_file, create_04_ocr_asr_file
from data_enhancement.BackTranslation import create_05_backTranslate
from file_enhancement import create_05_backTranslate_file
from qualityEval_file import create_01_qualityEval
from data_correctDemo import create01_correctDemo


server = flask.Flask(__name__)
server.config["JSON_AS_ASCII"] = False




# 1.数据增强方法
# 1.1 基于腐化语料的数据增强
@server.route('/rule_based', methods=['post', 'get'])
def rule_based():
    """
    input:获取传入的字符串
    :return:
    本地：
    http://10.50.50.6:5000/rule_based?line=少先队员因该为老人让坐&E_rate=30&num=3
    服务器：
    http://124.221.174.224:5002/rule_based?line=少先队员因该为老人让坐&E_rate=30&num=3
    """
    line = flask.request.values.get('line')
    E_rate = flask.request.values.get('E_rate')
    num = flask.request.values.get('num')
    list_result = create_01_rule_based.get_rule_based_num(line, int(E_rate), int(num))
    # print(list_result)
    return jsonify({'list_sentence_rule_based': list_result}) # 防止出现乱码；json.dumps()函数是将字典转化为字符串


# 1.2 EDA
@server.route('/EDA', methods=['post', 'get'])
def EDA():
    """
    input:获取传入的字符串
    :return:
    本地：
    http://10.50.50.6:5000/EDA?line=少先队员因该为老人让坐&alpha=0.1&num_age=3
    服务器：
    http://124.221.174.224:5002/EDA?line=少先队员因该为老人让坐&alpha=0.1&num_age=3
    """
    line = flask.request.values.get('line')
    alpha = flask.request.values.get('alpha')
    num_age = flask.request.values.get('num_age')
    list_result = create_02_EDA.get_EDA_num(line, int(num_age), float(alpha))
    # print(list_result)
    return jsonify(
            {'list_sentence_EDA': list_result})
    # return jsonify({'code': 0,"msg":list_result})# 防止出现乱码；json.dumps()函数是将字典转化为字符串

# 1.3 asr
@server.route('/asr', methods=['post', 'get'])
def asr():
    """
    input:获取传入的字符串
    :return:
    本地：
    http://10.50.50.6:5000/asr?line=少先队员因该为老人让坐
    服务器：
    http://124.221.174.224:5002/asr?line=少先队员因该为老人让坐
    """
    line = flask.request.values.get('line')
    list_result = create_03_asr.get_asr(line)
    # print(list_result)
    return jsonify(
        {'list_sentence_asr': list_result})  # 防止出现乱码；json.dumps()函数是将字典转化为字符串

# 1.4 ocr
@server.route('/ocr', methods=['post', 'get'])
def ocr():
    """
    input:获取传入的字符串
    :return:
    本地：
    http://10.50.50.6:5000/ocr?line=少先队员因该为老人让坐
    服务器：
    http://124.221.174.224:5002/ocr?line=少先队员因该为老人让坐
    """
    line = flask.request.values.get('line')
    list_result = create_04_ocr.get_ocr(line)
    print(list_result)
    return jsonify(
        {'list_sentence_ocr': list_result})  # 防止出现乱码；json.dumps()函数是将字典转化为字符串

# 1.5 backTranslation
@server.route('/backTranslation', methods=['post', 'get'])
def backTranslation():
    """
    input:获取传入的字符串
    :return:
    本地：
    http://10.50.50.6:5000/backTranslation?line=少先队员因该为老人让坐
    服务器：
    http://124.221.174.224:5002/backTranslation?line=少先队员因该为老人让坐
    """
    line = flask.request.values.get('line')
    list_result = create_05_backTranslate.get_backTranslation(line)
    # print(list_result)
    return jsonify(
        {'list_sentence_back': list_result})  # 防止出现乱码；json.dumps()函数是将字典转化为字符串


# 2.数据增强方法--文件
# 2.1 基于腐化语料的数据增强
@server.route('/rule_based_file', methods=['post', 'get'])
def rule_based_file():
    """
    input:获取传入的字符串
    :return:
    本地：
    http://10.50.50.6:5000/rule_based_file?fileInName=2023010377fesadf&fileOutName=2023_rule_based&E_rate=30&num=3
    服务器：
    http://124.221.174.224:5002/rule_based_file?fileInName=2023010377fesadf&fileOutName=2023_rule_based&E_rate=30&num=3
    """
    fileInName = flask.request.values.get('fileInName')
    # fileOutName = flask.request.values.get('fileOutName')
    print(fileInName)
    # InName = "/home/ubuntu/songcheng/upload/enhance_file/" + fileInName + ".txt"
    InName = "/songcheng_file/upload/enhance_file/" + fileInName
    print(InName)
    # OutName = "/home/ubuntu/songcheng/download/enhance_file/" + fileOutName + ".txt"
    OutName = "/songcheng_file/download/enhance_file/" + fileInName
    E_rate = flask.request.values.get('E_rate')
    num = flask.request.values.get('num')
    code = create_01_rule_based_file.get_rule_based_two(InName,OutName,int(E_rate), int(num))
    # print(list_result)
    return jsonify(
        {'code': code})  # code==0 表明生成成功！！！

# 2.2 EDA
@server.route('/EDA_file', methods=['post', 'get'])
def EDA_file():
    """
    input:获取传入的字符串
    :return:
    本地：
    http://10.50.50.6:5000/EDA_file?fileInName=2023010377fesadf&fileOutName=2023_EDA&alpha=0.1&num_age=3
    服务器：
    http://124.221.174.224:5002/EDA_file?fileInName=2023010377fesadf&fileOutName=2023_EDA&alpha=0.1&num_age=3
    """
    fileInName = flask.request.values.get('fileInName')
    # fileOutName = flask.request.values.get('fileOutName')
    # InName = "/home/ubuntu/songcheng/upload/enhance_file/" + fileInName + ".txt"
    # OutName = "/home/ubuntu/songcheng/download/enhance_file/" + fileOutName + ".txt"
    InName = "/songcheng_file/upload/enhance_file/" + fileInName
    OutName = "/songcheng_file/download/enhance_file/" + fileInName
    alpha = flask.request.values.get('alpha')
    num_age = flask.request.values.get('num_age')
    code = create_02_EDA_file.get_EDA(InName, OutName, int(num_age), float(alpha))
    return jsonify({'code': code})


# 2.3 ocr_asr
@server.route('/ocr_asr_file', methods=['post', 'get'])
def ocr_asr_file():
    """
    input:获取传入的字符串
    :return:
    本地：
    http://10.50.50.6:5000/ocr_asr_file?fileInName=2023010377fesadf&fileOutName=2023_ocr_asr
    服务器：
    http://124.221.174.224:5002/ocr_asr_file?fileInName=2023010377fesadf&fileOutName=2023_ocr_asr
    """
    fileInName = flask.request.values.get('fileInName')
    # fileOutName = flask.request.values.get('fileOutName')
    # InName = "/home/ubuntu/songcheng/upload/enhance_file/" + fileInName + ".txt"
    # OutName = "/home/ubuntu/songcheng/download/enhance_file/" + fileOutName + ".txt"
    InName = "/songcheng_file/upload/enhance_file/" + fileInName
    OutName = "/songcheng_file/download/enhance_file/" + fileInName
    code = create_04_ocr_asr_file.get_ocr_asr_file(InName, OutName)
    return jsonify(
        {'code': code})  # 防止出现乱码；json.dumps()函数是将字典转化为字符串

# 2.4 backTranslation_file
@server.route('/backTranslation_file', methods=['post', 'get'])
def backTranslation_file():
    """
    input:获取传入的字符串
    :return:
    本地：
    http://10.50.50.6:5000/backTranslation_file?fileInName=2023010377fesadf&fileOutName=2023_backTranslation_file
    服务器：
    http://124.221.174.224:5002/backTranslation_file?fileInName=2023010377fesadf&fileOutName=2023_backTranslation_file
    """
    fileInName = flask.request.values.get('fileInName')
    # fileOutName = flask.request.values.get('fileOutName')
    # InName = "/home/ubuntu/songcheng/upload/enhance_file/" + fileInName + ".txt"
    # OutName = "/home/ubuntu/songcheng/download/enhance_file/" + fileOutName + ".txt"
    InName = "/songcheng_file/upload/enhance_file/" + fileInName
    OutName = "/songcheng_file/download/enhance_file/" + fileInName
    code = create_05_backTranslate_file.get_backTranslation(InName, OutName)
    return jsonify(
        {'code': code})  # 防止出现乱码；json.dumps()函数是将字典转化为字符串

# 3.1 qualityEval_file
@server.route('/qualityEval_file', methods=['post', 'get'])
def qualityEval_file():
    """
    input:获取传入的字符串
    :return:
    本地：
    http://10.50.50.6:5000/qualityEval_file?fileInName=2023010377fesadf&task=拼写纠错
    服务器：
    http://124.221.174.224:5002/qualityEval_file?fileInName=2023010377fesadf&task=拼写纠错
    """
    fileInName = flask.request.values.get('fileInName')
    task = flask.request.values.get('task')
    print("中文纠错任务名：", task)
    print(task == "拼写纠错")
    list_quality = create_01_qualityEval.get_result_quality(fileInName, str(task))
    print(list_quality)
    return jsonify(
        {'list_quality': list_quality})  # 防止出现乱码；json.dumps()函数是将字典转化为字符串


# 4.1 中文纠错
@server.route('/correct_demo', methods=['post', 'get'])
def correct_demo():
    """
    input:获取传入的字符串
    :return:
    本地：
    http://10.50.50.6:5000/correct_demo?line=少先队员因该为老人让坐&model=RNN&task=拼写纠错
    服务器：
    http://124.221.174.224:5002/correct_demo?line=少先队员因该为老人让坐&model=RNN&task=拼写纠错
    """
    sentence = flask.request.values.get('line')
    model = flask.request.values.get('model')
    task = flask.request.values.get('task')
    correct = create01_correctDemo.get_correct_result(task, sentence)
    return jsonify({'correct': correct}) # 防止出现乱码；json.dumps()函数是将字典转化为字符串

"""
    1.数据增强的方案的api调用
"""
if __name__ == '__main__':
    server.run(host='0.0.0.0', port=5002)