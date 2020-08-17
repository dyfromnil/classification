from utils import computeTime
from utils import readDataFromMysql

from datetime import datetime
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt

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

# dataset_train, dataset_test, dataset_train, label_test = train_test_split(
#     data, labels, test_size=ts)

dataset_train, label_train = data, labels

# 特征名
feature_names = data_df.columns.to_list()
feature_names = feature_names[feature_names.index('attach_count'):]

# 网格搜索参数
cv_params = {
    'n_estimators': range(10, 50, 4),
    'gamma': np.linspace(0.1, 2, 20),     # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': range(10, 50, 2),               # 构建树的深度，越大越容易过拟合
    # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'lambda': np.linspace(0.1, 4, 40),
    'subsample': np.linspace(0.72, 0.9, 10),              # 随机采样训练样本
    'colsample_bytree': np.linspace(0.73, 1, 10),       # 生成树时进行的列采样
    'min_child_weight': range(1, 5),
    # 'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'learning_rate': np.linspace(0.01, 1, 100),                  # 如同学习率
}

# 固定参数
other_params = {
    'seed': 1000,
    'scale_pos_weight': scale_pos_weight,
    'early_stopping_rounds': 5,
    'objective': 'binary:logistic',
}

# xgboost训练
model = xgb.XGBClassifier(**other_params)

scoring = {
    'acc': make_scorer(accuracy_score),
    'roc_auc': make_scorer(roc_auc_score)
}
# grid = GridSearchCV(model, cv_params, cv=5, refit='acc',
#                     scoring=scoring, n_jobs=-1)

grid = RandomizedSearchCV(model, cv_params, cv=5, refit='acc', n_iter=300,
                          scoring=scoring, n_jobs=-1)

grid.fit(dataset_train, label_train)
best_estimator = grid.best_estimator_
# grid.cv_results_

# 水平柱状图
importance_dict = {
    'features': feature_names,
    'importance': best_estimator.feature_importances_
}
importance = pd.DataFrame(importance_dict)
importance = importance.sort_values(
    'importance').reset_index(drop=True)
importance = importance[~(importance['importance'] == 0)]

fig, ax = plt.subplots()
ax.barh(range(len(importance)),
        importance['importance'], tick_label=importance['features'])
ax.set_xlabel('importance', color='k')
ax.set_ylabel('features', color='k')
plt.title('features importance', loc='center', fontsize='15', color='red')
for x, y in enumerate(importance['importance']):
    ax.text(y+0.005, x, round(y, 4), va='center')
plt.show()

# xgb.plot_tree(bst, num_trees=5)
# xgb.to_graphviz(best_estimator, num_trees=10)

# 编码为onehot向量
enc = OneHotEncoder()
enc.fit(best_estimator.apply(dataset_train))

oh_train = enc.transform(best_estimator.apply(dataset_train)).toarray()
oh_pre = enc.transform(best_estimator.apply(pre_data)).toarray()

# 所有特征:onehot向量+原特征向量 --> lr模型训练使用
lr_train = np.hstack([oh_train, dataset_train])
lr_pre = np.hstack([oh_pre, pre_data])

# lr固定参数
lr_other_params = {
    'penalty': 'l2',
    'class_weight': 'balanced',
    'n_jobs': -1
}
LR = LogisticRegression(**lr_other_params)

# lr超参搜索
lr_cv_params = {
    'C': np.linspace(0.1, 6, 60),
    'max_iter': np.linspace(50, 1000, 20),
}

grid_lr = RandomizedSearchCV(LR, lr_cv_params, cv=5, refit='acc', n_iter=300,
                             scoring=scoring, n_jobs=-1)
grid_lr.fit(lr_train, label_train)

lr_best_estimator = grid_lr.best_estimator_

# 预测类别为1的概率
probability = lr_best_estimator.predict_proba(lr_pre)[:, 1]

# ------------- 老供应商 ----------------
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
result = result[~result['name'].isin(tmp['name'])]
