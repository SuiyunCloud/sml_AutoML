"""
对输入的模型自动调参
输入：模型（xgboost、rf），参数（经验范围，上下限）
输出：  1.调完参数的模型
        2.以log文件方式展示调参迭代过程
"""

import os
import time
import logging
from math import ceil
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV


def logger_set():
    """
    创建一个logger
    :rtype : loger
    """

    logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='autoTuning.log',
                filemode='w')
    #################################################################################################
    #定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    global logger
    logger = logging

# logger.setLevel(logging.DEBUG)  # Log等级总开关

# # 创建一个handler，用于写入日志文件
# rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
# log_path = os.path.dirname(os.getcwd()) + '/Logs/'
# if not os.path.exists(log_path):
#     os.makedirs(log_path)
# # log_name = log_path + rq + '.log'
# log_name = log_path + 'Logs.log'
# # log_name = 'D:/log.log'
# logfile = log_name
# fh = logging.FileHandler(logfile, mode='a')
# fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

# # 定义handler的输出格式
# formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
# fh.setFormatter(formatter)
# logger.addHandler(fh)
# logger.info("new try")
# #return logger


def calculater(base_model, params, train_data, target):
    # assert isinstance(train_data, object)
    train_xy, val = train_test_split(train_data, test_size=0.3, random_state=1)
    y = train_xy.Survived
    X = train_xy.drop([target], axis=1)
    val_y = val.Survived
    val_X = val.drop([target], axis=1)

    gsearch = GridSearchCV(estimator=base_model, param_grid=params, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    # Fit the algorithm on the data
    gsearch.fit(X, y, eval_metric='auc')
    # gsearch.fit(X,y)

    # Predict training set:
    dtrain_predictions = gsearch.predict(val_X)
    dtrain_predprob = gsearch.predict_proba(val_X)[:, 1]

    # save result
    accuracy = metrics.accuracy_score(val_y, dtrain_predictions)
    auc = metrics.roc_auc_score(val_y, dtrain_predprob)
    res = {'params': params, 'best_params': gsearch.best_params_, 'best_scores': gsearch.best_score_,
           'accuracy': accuracy, 'AUC': auc, 'gridScore': gsearch.grid_scores_}

    # Print model report:
    print("\nModel Report")
    print("\nParams:", params)
    print("\nbest_params:", gsearch.best_params_)
    print("\nbest_scores:", gsearch.best_score_)
    print("Accuracy : %.4g" % accuracy)
    print("AUC Score (Train): %f" % auc)
    return res


def createSamples(input_params, samples):
    # 字符类型参数值用list，数值型用tuple给出最小、最大

    param_new = {}
    for (name, val) in input_params.items():
        if isinstance(val, list):
            param_new[name] = val
        elif isinstance(val, tuple):
            min_val, max_val = val
            if max_val < min_val:
                (min_val, max_val) = (max_val, min_val)
            if min_val >= 1:
                res_temp, canAdjust = intRange(min_val, max_val, samples)
            elif min_val == 0:
                logger.debug('min_val is {}'.format(min_val))
                logger.debug("max_val is {}".format(max_val))
                step = ceil((max_val / (samples-1))*100)/100
                res_temp = [i*step for i in range(samples)]
                canAdjust = True

            else:
                # 较小值的小数位数应比较大值多
                logger.debug('min_val is {}'.format(min_val))
                min_str = str(min_val)
                min_str = min_str.split('.')
                num = len(min_str[1])
                min_new = int(min_val * pow(10, num))
                # max_new = ceil(max_val/min_val)
                max_new = int(max_val * pow(10, num))

                res_temp, canAdjust = intRange(min_new, max_new, samples)
                res_temp = [i / pow(10, num) for i in res_temp]

            # print(res_temp)
            if res_temp[-1] < max_val:
                res_temp.append(max_val)
            elif res_temp[-1] > max_val:
                res_temp[-1] = max_val

            param_new[name] = res_temp

        else:
            return input_params, False

    return param_new, canAdjust


def intRange(min, max, samples):
    if (max - min) > samples:
        step = ceil((max - min) / samples)
        can_adjust = True
    else:
        step = 1
        samples = max - min
        can_adjust = False
    return [i * step + min for i in range(samples)], can_adjust


def is_converge(res, expand=10, converge_threshold=0.001):
    meanScores = []
    itera_params = []
    for row in res['gridScore']:
        meanScores.append(row[1])
        itera_params.append(row[0])
    data = pd.DataFrame({'meanScore': [i for i in meanScores], 'itera_params': [i for i in itera_params]})
    maxIndex = data['meanScore'].idxmax(axis=1)  # 最大值索引号
    itera_params = data['itera_params']

    if maxIndex > 0 & maxIndex < len(meanScores) - 1:
        smaller = meanScores[maxIndex - 1]
        bigger = meanScores[maxIndex + 1]
        avg = (smaller + bigger) / 2
        max = meanScores[maxIndex]
        if max - avg < converge_threshold:
            if max - avg == 0:
                if meanScores[0] != meanScores[len(meanScores) - 1]:
                    return True, {}
                else:
                    # pass #所有值都一样的情况下，两边各扩大
                    return False, expand_both(itera_params, expand)
            else:
                return True, {}
        else:
            return False, combine_dict(itera_params[maxIndex - 1], itera_params[maxIndex + 1])
    else:
        # pass #重新分区域，最值在两端点的情况3
        #
        if maxIndex == 0:
            tem_index = 0
            for i, v in enumerate(meanScores):
                if v < meanScores[0]:
                    tem_index = i
                    break
            if tem_index == len(meanScores) - 1:
                return False, expand_both(itera_params, expand)
            else:
                for k, v in itera_params[0].items():
                    itera_params[0][k] = (v / expand, itera_params[tem_index][k])
                return False, itera_params[0]
        else:
            tem_index = 0
            for i, v in enumerate(meanScores):
                if v < meanScores[len(meanScores) - i]:
                    tem_index = len(meanScores) - i
                    break
            if tem_index == 0:
                return False, expand_both(itera_params, expand)
            else:
                count = len(meanScores) - 1
                for k, v in itera_params[count].items():
                    itera_params[0][k] = (itera_params[tem_index][k], v * expand)
                return False, itera_params[0]


def expand_both(data, expand):
    for k, v in data[0].items():
        data[0][k] = (v / expand, data[0][len(data) - 1] * expand)
    return data[0]

def combine_dict(x, y):
    for k, v in x.items():
        y[k] = (v, y[k])
    return y

# def compare_boundary(param,label,min_fixed,max_fixed):
def auto_tuning(model, init_param, params, data, target='Survived', expand=10,samples=10):
    """
    params包含参数的字典，每个参数的值由四维tuple构成，前两位是初始最小、最大，后两位是取值上下极限边界,若上下极限边界值为-1表示不做上下极限限制
    如：learning_rate:(0.05,0.3,0,1)表示learning_rate初始范围为（0.05,0.3），当在给范围找不到最优结果时，会扩大到（0,1）内
    :param model: 模型
    :param init_param:模型的初始值
    :param params: 参数寻优范围列表
    :param data: 训练数据
    :param target:训练数据中的目标
    :param expand:当初始寻优范围内寻优失败时，扩大边界所乘或除的系数，下边界除以expand，上边界乘expand
    :param samples: 每次迭代寻优时，参数点数
    :return:训练好的模型
    """
    logger_set()
    #logger = logger_set()

    label = -1  # 标识参数没有上界或下界限
    rests = []
    param = {}
    num = 0
    estimator = model(**init_param)
    for k, v in params.items():
        param.clear()
        param[k] = (v[0], v[1])
        min_fixed = v[2]
        max_fixed = v[3]
        is_done = False

        res = []
        while not is_done:
            param_new, canAdjust = createSamples(param, samples)
            res = calculater(estimator, param_new, data, target)
            num += 1
            # logger.info('hello')
            logger.info("Current calculation step is {}".format(num))
            logger.info(res)
            logger.info("\n")
            rests.append(res)
            #logger.debug('hello')
            if canAdjust:
                is_done, param = is_converge(res,expand)
                if is_done:
                    break
                # deal with fixed boundary
                temp = [i for i in param.values()]
                key = [i for i in param.keys()]
                min_val_temp, max_val_temp = temp[0]
                if min_fixed == label:
                    # only upper boundary
                    if max_fixed != label:
                        if max_fixed < min_val_temp:
                            is_done = True
                            logger.warning(
                                u"调整后的边界超出设定的上限范围，上限位{}，当前调整范围为：({},{})".format(max_fixed, min_val_temp, max_val_temp))

                        elif max_fixed < max_val_temp:
                            param[key[0]] = (min_val_temp, max_fixed)
                elif max_fixed == label:
                    if max_fixed > max_val_temp:
                        is_done = True
                        logger.warning(
                            u"调整后的边界超出设定的下限范围，下限位{}，当前调整范围为：({},{})".format(min_fixed, min_val_temp, max_val_temp))
                    elif min_fixed < min_val_temp:
                        param[key[0]] = (min_fixed, max_val_temp)
                else:
                    if max_fixed >= max_val_temp:
                        if min_fixed > min_val_temp:
                            param[key[0]] = (min_fixed, max_val_temp)
                    else:
                        if min_fixed <= min_val_temp:
                            param[key[0]] = (min_val_temp, max_fixed)
                        else:
                            is_done = True
                            logger.warning(u"调整后的边界超出设定的上下限范围，上下限位：({},{})，当前调整范围为：({},{})".format(min_fixed, max_fixed,
                                                                                                   min_val_temp,
                                                                                                   max_val_temp))
            else:
                is_done = True
        # 更新参数
        # print(res)
        init_param.update(res['best_params'])
        estimator = model(**init_param)
    return estimator






from xgboost import XGBClassifier

__author__ = 'suiyun.yang'
# coding: utf-8
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn  # ignore annoying warning (from sklearn and seaborn)
from scipy import stats
from scipy.stats import norm, skew  # for some statistics
from speedml import Speedml

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))  # Limiting floats

sml = Speedml("D:/train.csv", "D:/test.csv", target='Survived', uid='PassengerId')
train = sml.train
test = sml.test
ntrain = train.shape[0]
ntest = test.shape[0]
data = train.append(test)

sml.eda()
sml.feature.outliers('Fare', upper=99)
sml.feature.outliers('Parch', upper=99)
sml.feature.density(['Age', 'Ticket'])
sml.train[['Age', 'Age_density']].head()
sml.feature.fillna(a='Cabin', new='Z')
sml.feature.extract(new='Deck', a='Cabin', regex='([A-Z]){1}')
sml.feature.mapping('Sex', {'male': 0, 'female': 1})
sml.feature.sum(new='FamilySize', a='Parch', b='SibSp')
sml.feature.add('FamilySize', 1)
sml.feature.drop(['Parch', 'SibSp'])
sml.feature.impute()
sml.feature.extract(new='Title', a='Name', regex='([A-Za-z]+)\.')
sml.feature.replace(a='Title',
                    match=['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', "Major", 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                    new='Rare')
sml.feature.replace('Title', 'Mlle', 'Miss')
sml.feature.replace("Title", 'Mme', 'Mrs')
sml.feature.drop(['Name', 'Cabin', 'Ticket'])
sml.feature.labels(['Title', 'Embarked', 'Deck'])
sml.eda()
sml.feature.drop(['Ticket_density', 'Age_density'])
train = sml.train
test = sml.test
y_train = train.Survived
train_avg = train.drop('Survived', axis=1)
test_avg = test
train_avg.shape
from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings('ignore')
import scipy.stats as sns


from collections import defaultdict, OrderedDict
from sklearn.ensemble import GradientBoostingClassifier
param_test3 = OrderedDict()
param_test3['n_estimators']=(15, 200, 2, 10000)
param_test3['max_depth']=(3, 10, 2, 1000)
param_test3['max_leaf_nodes']=(2, 1000, -1, -1)
param_test3['min_samples_split']=(2, 1000, -1, -1)
param_test3['min_samples_leaf']=(2,1000, 1, 100000)
param_test3['min_weight_fraction_leaf']=(0, 0.5,0, 0.5)
#param_test3['max_features']=(0.1, 10, 0.001, 100)
from speedml.tuning import auto_tuning
estimator = GradientBoostingClassifier
init_param ={}
##init_param = { 'n_estimators': 15,
#              'max_depth': 5, 'max_leaf_nodes': 0.9, 'min_samples_split': 0,
#              'min_samples_leaf': 0.8, 'min_weight_fraction_leaf': 0.8,
#              'min_weight_fraction_leaf': 'binary:logistic',
#              'max_features': 4}
#if __name__ == "__main__":
#    gbdt = auto_tuning(estimator, init_param, param_test3, train)

import lightgbm as lgb
from collections import defaultdict, OrderedDict
param_test5 = OrderedDict()
#param_test5['bagging_freq']=(1, 50, 1, -1)
#param_test5['max_depth'] = (3, 8, 2, 1000)
#param_test5['num_leaves'] = (20, 200, 2, 10000)
#param_test5['max_bin'] = (10, 255,2, -1)
#param_test5['min_data_in_leaf'] = (2, 100, -1, -1)
#param_test5['feature_fraction'] = (0.01, 1, 0.0001, 1)
#param_test5['bagging_fraction'] = (0.01, 1, 0.0001, 1)
#param_test5['lambda_l1']=(0.0, 1, 0, 1)
#param_test5['lambda_l2']=(0.0, 1, 0, 1)

param_test5['max_depth']=(3, 8, 2, 1000)
param_test5['num_leaves']=(20, 200, 2, 10000)
param_test5['max_bin']=(10, 255,2, -1)
param_test5['min_data_in_leaf']=(2, 100, 2, -1)
param_test5['feature_fraction']=(0.01, 1, 0.0001, 1)
param_test5['bagging_fraction']=(0.01, 1, 0.0001, 1)
param_test5['bagging_freq']=(2, 50,2, -1)
param_test5['lambda_l1']=(0.0, 1, 0, 1)
param_test5['lambda_l2']=(0.0, 1, 0, 1)


estimator = lgb.LGBMClassifier
init_param = {}
if __name__ == "__main__":
   # print('hello')
   # xgb = auto_tuning(estimator, init_param, param_test5, train)
   automl = autosklearn.classification.