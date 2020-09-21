
from sklearn import svm
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import numpy as np
from pandas import DataFrame
import logging

from collections import defaultdict
from sklearn.metrics import roc_auc_score

import xgboost as xgb

from laxin import evaluate


# dtrain = xgb.DMatrix('../data/agaricus.txt.train#dtrain.cache')


class ModelProcess(object):
    '''
    模型预测相关类, 构造函数：

    '''
    def __init__(self, model_type="", config_info=None):
        self.config_info = config_info
        self.model_type = model_type
        self.model = None

    def train_model(self, train_x, train_y, test_x=None, test_y=None, is_gridsearch=-1):
        '''
        训练模型
        Args:       train_x = DataFrame
                    train_y = DataFrame
        Returns:    model = obj
        '''
        # 配置参数并训练模型
        if 'gradient_boost' == self.model_type:
            cfg_max_features = self.config_info["max_features"]
            cfg_n_estimators = self.config_info["n_estimators"]
            cfg_random_state = self.config_info["random_state"]
            cfg_learning_rate = self.config_info["learning_rate"]
            cfg_min_samples_split = self.config_info["min_samples_split"]
            cfg_min_samples_leaf = self.config_info["min_samples_leaf"]
            cfg_max_depth = self.config_info["max_depth"]
            cfg_subsample = self.config_info["subsample"]
            param = {'n_estimators':range(20 ,81 ,10)}
            cfg_scoring = self.config_info["scoring"]
            cfg_cv = self.config_info["cv"]
            model = GradientBoostingClassifier(max_features=cfg_max_features,
                                               n_estimators=cfg_n_estimators,
                                               random_state=cfg_random_state,
                                               learning_rate=cfg_learning_rate,
                                               min_samples_split=cfg_min_samples_split,
                                               min_samples_leaf=cfg_min_samples_leaf,
                                               max_depth=cfg_max_depth,
                                               subsample=cfg_subsample)
            model.fit(train_x, train_y)
        elif "xgboost" == self.model_type:
            cfg_nthread = self.config_info["nthread"] if -1 != self.config_info["nthread"] else -1
            cfg_scale_pos_weight = self.config_info["scale_pos_weight"] if -1 != self.config_info["scale_pos_weight"] else 1
            cfg_n_estimators = self.config_info["n_estimators"] if -1 != self.config_info["n_estimators"] else 8
            cfg_min_child_weight = self.config_info["min_child_weight"] if -1 != self.config_info["min_child_weight"] else 1
            cfg_max_depth = self.config_info["max_depth"] if -1 != self.config_info["max_depth"] else 6
            cfg_subsample = self.config_info["subsample"] if -1 != self.config_info["subsample"] else 1
            cfg_colsample_bytree = self.config_info["colsample_bytree"] if -1 != self.config_info["colsample_bytree"] else 1
            cfg_learning_rate = self.config_info["learning_rate"] if -1 != self.config_info["learning_rate"] else 0.3
            cfg_gamma = self.config_info["gamma"] if -1 != self.config_info["gamma"] else 0.0
            cfg_alpha = self.config_info["alpha"] if -1 != self.config_info["alpha"] else 1
            cfg_objective = self.config_info["objective"] if "" != self.config_info["objective"] else "reg:logistic"
            cfg_eval_metric = self.config_info["eval_metric"] if "" != self.config_info["eval_metric"] else "rmse"
            model = XGBClassifier(nthread=cfg_nthread,
                                  scale_pos_weight=cfg_scale_pos_weight,
                                  n_estimators=cfg_n_estimators,
                                  min_child_weight=cfg_min_child_weight,
                                  max_depth=cfg_max_depth,
                                  subsample=cfg_subsample,
                                  colsample_bytree=cfg_colsample_bytree,
                                  learning_rate=cfg_learning_rate,
                                  gamma=cfg_gamma,
                                  objective = cfg_objective,
                                  eval_metric = cfg_eval_metric,
                                  alpha=cfg_alpha, silent=True)

            early_stopping_rounds = None if "early_stopping_rounds" not in self.config_info else self.config_info["early_stopping_rounds"]
            logging.error("early_stopping_rounds = " + str(early_stopping_rounds))
            model.fit(train_x, train_y, early_stopping_rounds=early_stopping_rounds, eval_set=[(test_x, test_y)])
        if "svm" == self.model_type:
            if 1 == is_gridsearch:
                # rbf核函数，设置数据权重
                svc = svm.SVC(kernel='linear', class_weight='balanced' ,)
                c_range = np.logspace(-5, 15, 5, base=2)
                gamma_range = np.logspace(-9, 3, 6, base=2)
                # 网格搜索交叉验证的参数范围，cv=3,3折交叉
                param_grid = [{'kernel': ['poly'], 'C': c_range, 'gamma': gamma_range}]
                grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=5)
                # 训练模型
                clf = grid.fit(train_x, train_y)
                # 计算测试集精度
                score = grid.score(test_x, test_y)
                return
            else:
                cfg_C = self.config_info["C"] if -1 != self.config_info["C"] else 1.0
                cfg_kernel = self.config_info["kernel"] if -1 != self.config_info["kernel"] else "rbf"
                cfg_degree = self.config_info["degree"] if -1 != self.config_info["degree"] else 3
                cfg_gamma = self.config_info["gamma"] if -1 != self.config_info["gamma"] else 'auto'
                cfg_coef0 = self.config_info["coef0"] if -1 != self.config_info["coef0"] else 9.0
                cfg_cache = self.config_info["cache"] if -1 != self.config_info["cache"] else 200
                cfg_max_iter = self.config_info["max_iter"] if -1 != self.config_info["max_iter"] else -1
                model = svm.SVC(kernel=cfg_kernel, C=cfg_C, degree=cfg_degree, gamma=cfg_gamma, coef0=cfg_coef0, max_iter=cfg_max_iter, probability=True)
                model.fit(train_x ,train_y)
        self.model = model

    def save_model(self, model_path):
        '''
        保存模型
        Args:       model_path = str
        Returns:    None
        '''
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        '''
        加载模型
        Args:       model_path = str
        '''
        self.model = joblib.load(model_path)

    def predict(self, feature_df):
        '''
        模型预测
        Args:       model = model_obj
                    feature_df = DataFrame
        Retruns:    pred_df = DataFrame
                        columns = ["pred_prob","pred_label"]
        '''
        columns = ["pred_prob" ,"pred_label"]
        pred_prob = self.model.predict_proba(feature_df)[:, 1].reshape([-1, 1])
        pred = self.model.predict(feature_df).reshape([-1, 1])
        pred_concat = np.concatenate((pred_prob, pred) ,axis=1)
        pred_df = DataFrame(pred_concat, index=feature_df.index, columns=columns)
        return pred_df

    def evaluate_model(self, test_x, test_y, ana_top=0.2, threshold=0.5, show_xgb_importance=-1):
        '''
        评估模型
        Args:       model = model_obj
                    test_x = DataFrame
                    test_y = DataFrame
        Returns:    DataFrame
        '''
        # 预测结果并分析
        pred_prob = self.model.predict_proba(test_x)[:, 1]
        pred = np.array(list(map(lambda x: 1 if x >= threshold else 0, pred_prob)))
        ana_segment_df = evaluate.analyse_model_result(pred_prob, pred, test_y, ana_top=ana_top)

        # 展现xgboost中特征的重要程度
        # if self.model_type in ('xgboost') and 1 == show_xgb_importance:
        #     fig, ax = plt.subplots(figsize=(15, 15))
        #     plot_importance(self.model,
        #                     height=0.5,
        #                     ax=ax,
        #                     max_num_features=64)
        #     plt.show()
        columns = ["pred_prob", "pred_label"]
        pred_concat = np.concatenate((pred_prob.reshape([-1, 1]), pred.reshape([-1, 1])), axis=1)
        return DataFrame(pred_concat, index=test_x.index, columns=columns)

    def feature_importance(self, importance_type='gain'):
        return self.model.get_booster().get_score(importance_type=importance_type)

    def features(self):
        return self.model.get_booster().feature_names


def cal_group_auc(labels, preds, user_id_list):
    """Calculate group auc"""

    print('*' * 50)
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0

    detail_auc = {}
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
            detail_auc[user_id] = auc
        else:
            detail_auc[user_id] = 1
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    return group_auc, detail_auc

