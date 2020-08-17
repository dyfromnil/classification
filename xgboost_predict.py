from utils import computeTime
from utils import readDataFromMysql

from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import xgboost as xgb
import os
os.environ["PATH"] += os.pathsep + 'D://Graphviz2.38/bin/'

# test_size
ts = 0.1

######## 训练数据 ###########
# data_df：VIP正负样本
data_df = pd.read_csv('./dataDump/vip_data.csv')

##############################
pre_sample = pd.read_csv('./dataDump/pre_data.csv')

old_supplier = pre_sample.query('years>0.25').reset_index(drop=True)
new_supplier = pre_sample.query('years<=0.25').reset_index(drop=True)

########
# 此后都是针对old_supplier进行预测
# new_supplier单独进行推广
########

# name_province：前两列是供应商的名字和地区，保存留到后面使用
name_province = old_supplier[['name', 'province']]

# pre_data:分离name_province后的剩余列数据,用于预测的列
pre_data = old_supplier.loc[:, 'attach_count':]
pre_data = pre_data.astype('float32')
pre_data = np.array(pre_data)

# 标准化预处理
# scaler = preprocessing.StandardScaler().fit(pre_data)
# pre_data = scaler.transform(pre_data)

########## train ############

labels = np.array(data_df.loc[:, 'label'])
data = np.array(data_df.loc[:, 'attach_count':])
# data = scaler.transform(data)

# 负样本数/正样本数
scale_pos_weight = len(labels[labels == 0])/len(labels[labels == 1])

dataset_train, dataset_test, label_train, label_test = train_test_split(
    data, labels, test_size=ts)

dtrain = xgb.DMatrix(dataset_train, label=label_train)
dtest = xgb.DMatrix(dataset_test, label=label_test)

# data_size = data_df.shape[0]
# train_size = dataset_train.shape[0]
# test_size = dataset_test.shape[0]

params = {
    'booster': 'gbtree',
    # 'objective': 'multi:softmax',  # 多分类的问题
    # 'num_class': 10,               # 类别数，与 multisoftmax 并用
    'objective': 'binary:logistic',
    'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 20,               # 构建树的深度，越大越容易过拟合
    'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,              # 随机采样训练样本
    'colsample_bytree': 0.7,       # 生成树时进行的列采样
    'min_child_weight': 3,
    # 'silent': 0,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.2,                  # 如同学习率
    'seed': 1000,
    'nthread': 4,                  # cpu 线程数
    'scale_pos_weight': scale_pos_weight
}

params['eval_metric'] = ['error', 'auc']

evallist = [(dtrain, 'train'), (dtest, 'eval')]

num_round = 30
bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=5)

bst.save_model('./model/xgb1.model')
# 转存模型
bst.dump_model('./model/dump.raw.txt')
# 转储模型和特征映射
bst.dump_model('./model/dump.raw.txt', './model/featmap.txt')

xgb.plot_importance(bst)
# xgb.plot_tree(bst, num_trees=5)
# xgb.to_graphviz(bst, num_trees=10)

# ------------- 老供应商 ----------------
pre_data_DMatrix = xgb.DMatrix(pre_data)
probability = bst.predict(pre_data_DMatrix, ntree_limit=bst.best_ntree_limit)
probability = pd.DataFrame(probability)
probability.columns = ['probability']
pre_result = pd.concat([name_province, probability], axis=1)

# 概率大于0.5的供应商认为会充值VIP
pre_positive = pre_result[pre_result['probability'] > 0.5]
pre_positive_sort = pre_positive.sort_values(
    'probability', ascending=False).reset_index(drop=True)

# ------------- 新供应商 ----------------
new = new_supplier.query(
    'years<=0.0192 or (years>=0.0685 and years<=0.0959) or (years>=0.1507 and years<=0.1781)').reset_index(drop=True)

name_province = new[['name', 'province']]

pre_data_new = new.loc[:, 'attach_count':]
pre_data_new = pre_data_new.astype('float32')
pre_data_new = np.array(pre_data_new)

pre_data_new_DMatrix = xgb.DMatrix(pre_data_new)
probability_new = bst.predict(
    pre_data_new_DMatrix, ntree_limit=bst.best_ntree_limit)
probability_new = pd.DataFrame(probability_new)
probability_new.columns = ['probability']
pre_result_new = pd.concat([name_province, probability_new], axis=1)

# 概率大于0.5的供应商认为会充值VIP
pre_positive_new = pre_result_new[pre_result_new['probability'] > 0.5]
pre_positive_new_sort = pre_positive_new.sort_values(
    'probability', ascending=False).reset_index(drop=True)

############## 划分下发数据 #############
result = pd.concat(
    [pre_positive_sort[:170], pre_positive_new_sort[:30]]).reset_index(drop=True)
result.to_csv('./dataResult/total.csv', index=False)

# 长江三角洲地区
Yangtze_River_Delta = set({'上海市', '江苏省', '浙江省', '安徽省'})

# 其余按照地区分为长江三角洲和其他
Yangtze = result[result['province'].isin(Yangtze_River_Delta)]
Yangtze.to_csv('./dataResult/长三角地区.csv', index=False)
other_region = result[~result['province'].isin(
    Yangtze_River_Delta)]
other_region.to_csv('./dataResult/其余地区.csv', index=False)


result = pd.read_csv('./dataResult/total.csv')
tmp = pd.read_csv('./tmp.csv')
result=result[~result['name'].isin(tmp['name'])]
