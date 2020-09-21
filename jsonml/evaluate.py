import numpy as np
from sklearn import metrics
import math
from pandas import DataFrame
import logging

logger = logging.getLogger('pylaxin')


def top_analyse(label_arr, pred_prob_arr, top_percent=0.2):
    '''
    top分析
    :param label_arr: numpy.ndarray    真实标签
    :param pred_prob_arr: numpy.ndarray   预测出来的概率
    :return: 概率前20%中的召回, 人数前20%的召回
    '''
    top_prob_recall_num, top_prob_num = 0, 0  # 概率前20%中的召回（即概率>=0.8的召回）
    top_uidnum_recall_num, top_uidnum_num = 0, 0  # 人数前20%的召回（即按照概率从大到小排序的前20%的人的召回）
    prob_index = np.argsort(pred_prob_arr)[::-1]  # 预测出来概率从大到小的值的下标
    all_num = len(pred_prob_arr)  # 总人数
    all_pos_num = sum(label_arr)  # 总的正样本数

    for one_i, one_idx in enumerate(prob_index):
        if pred_prob_arr[one_idx] >= (1 - top_percent):
            top_prob_recall_num += label_arr[one_idx]
            top_prob_num += 1
        if one_i <= all_num * top_percent:
            top_uidnum_recall_num += label_arr[one_idx]
            top_uidnum_num += 1
    logger.info('top_prob_recall_num:{0}, top_prob_num:{1}, top_uidnum_recall_num:{2}, top_uidnum_num:{3}'
                  .format(top_prob_recall_num, top_prob_num, top_uidnum_recall_num, top_uidnum_num))

    if top_prob_num > 0 and top_uidnum_num > 0:
        return top_prob_recall_num * 1.0 /top_prob_num, top_uidnum_recall_num * 1.0 / top_uidnum_num
    else:
        return -1, -1


def segment_analyse(label_arr, pred_prob_arr, ana_top):
    '''
    分段分析
    :param label_arr: numpy.ndarray    真实标签
    :param pred_prob_arr: numpy.ndarray   预测出来的概率
    :param ana_top:  float 分段比例
    :return: 各段的召回情况
    '''
    ana_segment = ana_top  # 每一段的比例，top开始算起
    ana_segment_recall_num = np.zeros([int(math.ceil(1.0 / ana_segment))])  # 各段召回个数
    ana_segment_num = np.zeros([int(math.ceil(1.0 / ana_segment))])  # 各段人数
    prob_index = pred_prob_arr.argsort()[::-1]  # 预测出来概率从大到小的值的下标
    all_num = len(pred_prob_arr)  # 总人数
    all_pos_num = sum(label_arr)  # 总的正样本数
    for one_idx in prob_index:
        adj_pred_prob = pred_prob_arr[one_idx]
        if 0.00001 > pred_prob_arr[one_idx]:
            adj_pred_prob = 0.00001
        elif 0.9999 < pred_prob_arr[one_idx]:
            adj_pred_prob = 0.9999
        ana_segment_idx = int((1 - adj_pred_prob) / ana_segment)
        ana_segment_num[ana_segment_idx] += 1
        if 1 == label_arr[one_idx]:
            ana_segment_recall_num[ana_segment_idx] += 1

    ana_segment_recall_rate = [float('{0:.3f}'.format(one_recall)) for one_recall in
                               ana_segment_recall_num / all_pos_num]
    ana_segment_acc_rate = []
    for index, value in enumerate(ana_segment_recall_num):
        one_acc = -1 if ana_segment_num[index] == 0 else value / ana_segment_num[index]
        ana_segment_acc_rate.append(float('{0:.3f}'.format(one_acc)))
    # ana_segment_acc_rate = [float('{0:.3f}'.format(one_acc)) for one_acc in ana_segment_recall_num / ana_segment_num]
    segment_columns = ['seg_top{idx}'.format(idx=one_idx) for one_idx in range(int(math.ceil(1.0 / ana_segment)))]
    ana_index = [u'人数', u'召回个数', u'召回率', u'准确率']
    ana_segment_values = [ana_segment_num,
                          ana_segment_recall_num,
                          ana_segment_recall_rate,
                          ana_segment_acc_rate]

    ana_segment_df = DataFrame(ana_segment_values, index=ana_index, columns=segment_columns)
    return ana_segment_df


def get_confusion_matrix(label_arr, pred_arr):
    '''
    计算混淆矩阵
    PARA:       label_arr = numpy.ndarray   真实标签
                pred_arr = numpy.ndarray   预测结果
    RETURN:     p2p, p2n, n2p, n2n = int
    '''
    p2p, p2n, n2p, n2n = 0, 0, 0, 0
    for one_idx, one_label in enumerate(label_arr):
        one_pred = pred_arr[one_idx]
        if 1 == one_label and 1 == one_pred:
            p2p += 1
        elif 1 == one_label and 0 == one_pred:
            p2n += 1
        elif 0 == one_label and 1 == one_pred:
            n2p += 1
        elif 0 == one_label and 0 == one_pred:
            n2n += 1
        else:
            pass
    return p2p, p2n, n2p, n2n


def analyse_model_result(pred_prob, pred, test_y, ana_top=0.2, valid_call_rate=0.2):
    '''
    分析模型结果
    Args:       pred_prob = numpy.ndarray
                test_x = DataFrame
                test_y = DataFrame
                ana_top = float
                valid_call_rate = float
    Returns:    ana_segment_df = DataFrame
    '''
    # S1: 使用metrics模块计算模型整体效果
    logger.info('[模型整体效果]')
    pos_neg_scale = "{pos_num}/{neg_num}".format(pos_num=sum(test_y.values),
                                                 neg_num=len(test_y.values) - sum(test_y.values))
    logger.info('正负样本比例, 正/负={scale}'.format(scale=pos_neg_scale))
    accuracy = metrics.accuracy_score(test_y.values, pred)
    logger.info('准确度:{0:.4f}'.format(accuracy))
    precision = metrics.precision_score(test_y.values, pred, average='weighted')
    logger.info('精确度:{0:.4f}'.format(precision))
    recall = metrics.recall_score(test_y.values, pred, average='weighted')
    logger.info('召回:{0:0.4f}'.format(recall))
    f1_score = metrics.f1_score(test_y.values, pred, average='weighted')
    logger.info('f1-score:{0:.4f}'.format(f1_score))
    auc = metrics.roc_auc_score(test_y.values, pred_prob)
    logger.info('auc:{0:.4f}'.format(auc))

    # S2: 统计出混淆
    logger.info('[模型混淆矩阵]')
    p2p, p2n, n2p, n2n = get_confusion_matrix(test_y['label'].values, pred)
    logger.info('p2p:{0}, p2n:{1}, n2p:{2}, n2n:{3}'.format(p2p, p2n, n2p, n2n))


    # S3: TMK相关的top分析
    logger.info('[TMK top分析]')
    top_analyse(test_y['label'].values, pred_prob)

    ## 分析各段的召回
    logger.info("[分段分析](会将概率剪切在(0, 1)之间)")
    ana_segment_df = segment_analyse(test_y['label'].values, pred_prob, ana_top)

    return ana_segment_df



