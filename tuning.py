"""
对输入的模型自动调参
输入：模型（xgboost、rf），参数（经验范围，上下限）
输出：  1.调完参数的模型
        2.以log文件方式展示调参迭代过程
"""

import time
import logging
from math import ceil
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV
def method_name():
    import lightgbm as lgb
    return lgb

lgb = method_name()


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

def calculater(base_model, params, train_data, target, scorer):
    result = {}
    # assert isinstance(train_data, object)
    train_xy, val = train_test_split(train_data, test_size=0.3, random_state=1)
    y = train_xy[target]
    X = train_xy.drop([target], axis=1)
    val_y = val[target]
    val_X = val.drop([target], axis=1)

    logger.debug('Parameters for calculation are {}'.format(params))
    logger.debug('model is {}'.format(base_model))
    #gsearch = GridSearchCV(estimator=base_model, param_grid=params, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch = GridSearchCV(estimator=base_model, param_grid=params, scoring=scorer, n_jobs=4, iid=False, cv=5)
    logger.debug('gsearch is {}'.format(gsearch))
    # Fit the algorithm on the data
    try:
        gsearch.fit(X, y)
    except:
        logger.exception("Exception of GridSearchCV when cv.fit()")
        return False,False
    # gsearch.fit(X,y)

    # Predict training set:
    #dtrain_predictions = gsearch.predict(val_X)
    #if scorer == "roc_auc":
    #    dtrain_predprob = gsearch.predict_proba(val_X)[:, 1]

    # save result
    #accuracy = metrics.accuracy_score(val_y, dtrain_predictions)
    #if scorer == "roc_auc":
    #    auc = metrics.roc_auc_score(val_y, dtrain_predprob)
    #res = {'params': params, 'best_params': gsearch.best_params_, 'best_scores': gsearch.best_score_,
    #       'accuracy': accuracy, 'AUC': auc, 'gridScore': gsearch.grid_scores_}
    result = {'params': params, 'best_params': gsearch.best_params_, 'best_scores': gsearch.best_score_,'gridScore':gsearch.grid_scores_}

    # Print model report:
    print("\nModel Report")
    print("\nParams:", params)
    print("\nbest_params:", gsearch.best_params_)
    print("\nbest_scores:", gsearch.best_score_)
    #print("Accuracy : %.4g" % accuracy)
    #print("AUC Score (Train): %f" % auc)
    logger.debug('Before calculation return,res is {}'.format(result))
    return result,gsearch


def createSamples(input_params, samples):
    # 字符类型参数值用list，数值型用tuple给出最小、最大

    param_new = {}
    for (name, val) in input_params.items():
        if isinstance(val, list):
            param_new[name] = val
        elif isinstance(val, tuple):
            min_val, max_val = val
            logger.debug("Original scope is {}".format(val))
            if max_val < min_val:
                (min_val, max_val) = (max_val, min_val)
            if min_val >= 1:
                res_temp, canAdjust = intRange(min_val, max_val, samples)
            elif min_val == 0:
                if max_val <= 1:
                    logger.debug('min_val is {}'.format(min_val))
                    logger.debug("max_val is {}".format(max_val))
                    temp = str(max_val*1.0)
                    temp = temp.split('.')
                    #num = len(temp[1].rstrip("0"))
                    num = len(temp[1])
                    max_new = int(max_val * pow(10, num))
                    res_temp, canAdjust = intRange(0, max_new, samples)
                    res_temp = [i / pow(10, num) for i in res_temp]
                else:
                    res_temp, canAdjust = intRange(0, int(max_val), samples)
                    

            else:
                # 较小值的小数位数应比较大值多
                logger.debug('min_val is {}'.format(min_val))
                num = get_decimal_place(min_val)
                min_new = int(min_val * pow(10, num))
                # max_new = ceil(max_val/min_val)
                max_new = int(max_val * pow(10, num))

                res_temp, canAdjust = intRange(min_new, max_new, samples)
                res_temp = [i / pow(10, num) for i in res_temp]
                logger.debug('optimization scope is {}'.format(res_temp))

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


def is_converge(res, min_fixed,expand, converge_threshold):
    meanScores = []
    itera_params = []
    for row in res['gridScore']:
        meanScores.append(row[1])
        itera_params.append(row[0])
    data = pd.DataFrame({'meanScore': [i for i in meanScores], 'itera_params': [i for i in itera_params]})
    maxIndex = data['meanScore'].idxmax(axis=1)  # 最大值索引号
    itera_params = data['itera_params']

    if (maxIndex > 0) & (maxIndex < len(meanScores) - 1):
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
            if tem_index == 0:
                return False, expand_both(itera_params, expand)
            else:
                #最优值在下端点
                for k, v in itera_params[0].items():
                    if min_fixed == 0:
                        itera_params[0][k] = (int(v / expand), itera_params[tem_index][k])
                    if get_decimal_place(min_fixed) == 0 & min_fixed > 0:
                        itera_params[0][k] = (math.ceil(v / expand) , itera_params[tem_index][k])
                    else:
                        itera_params[0][k] = (v / expand, itera_params[tem_index][k])
                return False, itera_params[0]
        else:
            tem_index = len(meanScores) -1
            for i in range(len(meanScores)):
                if meanScores[len(meanScores)  -i -1] < meanScores[len(meanScores)  -1]:
                    tem_index = len(meanScores)  -i -1
                    break
            if tem_index == len(meanScores) -1:
                return False, expand_both(itera_params, expand)
            else:
                count = len(meanScores) - 1
                for k, v in itera_params[count].items():
                    itera_params[0][k] = (itera_params[tem_index][k], v * expand)
                return False, itera_params[0]


def expand_both(data, expand):
    for k, v in data[0].items():
        data[0][k] = (v / expand, data[len(data) - 1][k] * expand)
    return data[0]

def combine_dict(x, y):
    for k, v in x.items():
        y[k] = (v, y[k])
    return y

'''
获取小数的位数，输入的小数可能是浮点型也可能是科学计数法表示,当整数部分很大时失效如1000000000.01，不会识别出小数位Python之间把小数位去掉了
'''
def get_decimal_place(decimal):
    str_temp = str(decimal*1.0)
    if 'e' in str_temp:
        str_temp = str(str_temp).split('e')
        try:
            num = len(str_temp[0].split('.')[1])
        except:
            num = 0
        num1 = int(str_temp[1])
        num += (num1*(-1) if '-' in str_temp[1] else num1 )
    else:
        num = len(str_temp.rstrip('0').split('.')[1])
    return num

# def compare_boundary(param,label,min_fixed,max_fixed):
def auto_tuning(model, init_param, params, data, target='Survived',scorer='roc_auc',expand=10,samples=10,min_positive_num=0.001,converge_threshold=0.001):
    """
    params包含参数的字典，每个参数的值由四维tuple构成，前两位是初始最小、最大，后两位是取值上下极限边界,若上下极限边界值为-1表示不做上下极限限制
    如：learning_rate:(0.05,0.3,0,1)表示learning_rate初始范围为（0.05,0.3），当在给范围找不到最优结果时，会扩大到（0,1）内

    注意：模型参数要么在0-1间，要么在1-无穷，其中1-无穷的参数，自动生成的参数均为整数

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

    #参数检查，模型参数要么在0-1间，要么在1-无穷

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

        #res = []
        while not is_done:
            res = {}
            logger.info("Current calculation step is {}".format(num))
            param_new, canAdjust = createSamples(param, samples)
            #logger.debug('Before calculation,res is {}'.format(res))
            res,end_model = calculater(estimator, param_new, data, target,scorer=scorer)
            #logger.debug('After calculation,res is {}'.format(res))
            if not res:
                return False

            #it has got the end reslut when best parameter is in fix boundaries
            best_params = res['best_params']
            best_params_val = [i for i in best_params.values()][0]
            if best_params_val in (min_fixed,max_fixed):
                is_done = True
            
            #if best parameter is smaller than min_positive_num, it should end
            if best_params_val <= min_positive_num:
                is_done = True

            num += 1
            # logger.info('hello')
            
            logger.info(res)
            logger.info("\n")
            rests.append(res)
            #logger.debug('hello')
            if canAdjust and (not is_done):
                is_done, param = is_converge(res,min_fixed,expand,converge_threshold)
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
                    if min_fixed > max_val_temp:
                        is_done = True
                        logger.warning(
                            u"调整后的边界超出设定的下限范围，下限位{}，当前调整范围为：({},{})".format(min_fixed, min_val_temp, max_val_temp))
                    elif min_fixed > min_val_temp:
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
                            logger.warning(u"调整后的边界超出设定的上下限范围，上下限位：({},{})，当前调整范围为：({},{})".format(min_fixed, max_fixed,min_val_temp,max_val_temp))
                
                temp = [i for i in param.values()]
                key = [i for i in param.keys()]
                min_val_temp, max_val_temp = temp[0]
                #当最小极限大于等于1，则参数必须是正整数
                if min_fixed >= 1 and (min_val_temp <min_fixed):
                    min_val_temp = min_fixed
                #当去除新调整的范围没有小数
                if get_decimal_place(min_val_temp) == 0:
                    min_val_temp = int(min_val_temp)
                if get_decimal_place(max_val_temp) == 0:
                    max_val_temp = int(max_val_temp)
                param[key[0]] = (min_val_temp, max_val_temp)   
               
            else:
                is_done = True

        # 更新参数
        logger.debug("best_params is {}".format(best_params))
        init_param.update(best_params)
        estimator = model(**init_param)
    return end_model
