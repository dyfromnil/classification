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
ts = 0.3

######## 训练数据 ###########
# data_df：VIP正负样本
data_df = pd.read_csv('./dataDump/vip_data.csv')

##############################

sql = 'select zjc_supplier.name,zjc_supplier.province,zjc_supplier_param.attach_count,zjc_supplier.type,zjc_supplier.fund,zjc_supplier_param.service_level_average,zjc_supplier_param.instock_service_average,zjc_supplier_param.instock_product_average,zjc_supplier_param.instock_deliverspeed_average,zjc_supplier_param.iswin_count,zjc_supplier_param.price_level_average,zjc_supplier_param.tender_count,zjc_supplier_param.semester_tender_count,zjc_supplier_param.bidconcern_count,zjc_supplier_param.semester_bidconcern_count,zjc_supplier_param.login_days,zjc_supplier_param.semester_login_days,zjc_supplier_param.integrity_count,zjc_supplier_param.contract_rate,zjc_supplier_param.instock_honesty_average,zjc_supplier.regchecktime,zjc_supplier.createtime from zjc_supplier INNER JOIN zjc_supplier_param on zjc_supplier.id=zjc_supplier_param.supplier_id where zjc_supplier.state=2;'
# all_supplier:所有供应商，包括训练用的正负样本
all_supplier = readDataFromMysql(sql)
all_supplier.columns = ['name', 'province', 'attach_count', 'type', 'fund', 'service_average', 'instock_service', 'instock_product', 'instock_deliverspeed', 'win', 'price',
                        'tender', 'se_tender', 'bidconcern', 'se_bidconcern', 'login', 'se_login', 'integrity', 'contract_rate', 'instock_honesty', 'regchecktime', 'createtime']
# 去除训练用的正负样本
all_supplier = all_supplier[~all_supplier['name'].isin(
    data_df['name'])].reset_index(drop=True)
# 去除战略供应商
strategy = pd.read_csv('./data/strategy.csv')
all_supplier = all_supplier[~all_supplier['name'].isin(
    strategy['name'])].reset_index(drop=True)
# 除去因各种其他原因不开通的供应商
other = pd.read_csv('./data/other.csv')
all_supplier = all_supplier[~all_supplier['name'].isin(
    other['name'])].reset_index(drop=True)

# 最后两列是注册和审核时间，用于计算已经成为平台用户的时长
time_all = computeTime(all_supplier.iloc[:, -2:])
time_all.columns = ['years']

# 将注册和审核时间替换为时长
all_supplier = pd.concat([all_supplier.iloc[:, :-2], time_all], axis=1)

# years>0.25的old_supplier,years<=0.25的new_supplier
old_supplier = all_supplier.query('years>0.25').reset_index(drop=True)
new_supplier = all_supplier.query('years<=0.25').reset_index(drop=True)
new_supplier.iloc[:, :2].to_csv('./dataResult/新入必推.csv', index=False)

########
# 此后都是针对old_supplier进行预测
# new_supplier单独进行推广
########

# name_province：前两列是供应商的名字和地区，保存留到后面使用
name_province = old_supplier[['name', 'province']]

# pre_data:分离name_province后的剩余列数据,用于预测的列
pre_data = old_supplier.iloc[:, 2:]
pre_data = pre_data.astype('float32')
pre_data = np.array(pre_data)

# 标准化预处理
# scaler = preprocessing.StandardScaler().fit(pre_data)
# pre_data = scaler.transform(pre_data)

########## train ############

labels = np.array(data_df.iloc[:, 1])
data = np.array(data_df.iloc[:, 2:])
# data = scaler.transform(data)

# 负样本数/正样本数
scale_pos_weight = len(labels[labels == 0])/len(labels[labels == 1])

dataset_train, dataset_test, label_train, label_test = train_test_split(
    data, labels, test_size=ts)

dtrain = xgb.DMatrix(dataset_train, label=label_train)
dtest = xgb.DMatrix(dataset_test, label=label_test)

data_size = data_df.shape[0]
train_size = dataset_train.shape[0]
test_size = dataset_test.shape[0]

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

pre_data_DMatrix = xgb.DMatrix(pre_data)
probability = bst.predict(pre_data_DMatrix, ntree_limit=bst.best_ntree_limit)
probability = pd.DataFrame(probability)
probability.columns = ['probability']
pre_result = pd.concat([name_province, probability], axis=1)
xgb.plot_importance(bst)
# xgb.plot_tree(bst, num_trees=5)
# xgb.to_graphviz(bst, num_trees=10)

############## analyse #################

# 概率大于0.5的供应商认为会充值VIP
pre_positive = pre_result[pre_result['probability'] > 0.5]
pre_positive.to_csv('./dataResult/pre_positive.csv', index=False)
# 排序后的
pre_positive_sort = pre_positive.sort_values(
    'probability', ascending=False).reset_index(drop=True)
pre_positive_sort.to_csv('./dataResult/pre_positive_sort.csv', index=False)
# pre_positive_sort.iloc[:1000].to_csv('./dataResult/pre_positive_sort.csv', index=False)

# 相较于上次统计，新增的VIP供应商
# 训练用的正负样本
vip_0 = data_df.query('label==0')
vip_1 = data_df.query('label==1')
# new_vip 最新的VIP统计名单
new_vip = pd.read_csv('./data/new_vip_list.csv')
# new_add_vip 相较于上次的名单中新增的名单
# 除去之前的正样本
new_add_vip = new_vip[~new_vip['name'].isin(
    vip_1['name'])].reset_index(drop=True)
# 除去上次VIP名单中  开通了VIP的新供应商(因为新供应商不在训练数据中)
vip_new_supplier = pd.read_csv('./dataDump/vip_new_supplier.csv')
new_add_vip = new_add_vip[~new_add_vip['name'].isin(
    vip_new_supplier['name'])].reset_index(drop=True)
# 之前表示不充值，后来又充值的
reverse_vip = new_add_vip[new_add_vip['name'].isin(
    vip_0['name'])].reset_index(drop=True)
# 除去“之前表示不充值，后来又充值的”的这部分新充值的供应商
new_add_vip_delReverse = new_add_vip[~new_add_vip['name'].isin(
    vip_0['name'])].reset_index(drop=True)

# new_add_vip_delReverse中的新(years>0.25)供应商&老供应商(years<0.25)&不在数据库里的供应商
new_add_vip_delReverse_new = new_add_vip_delReverse[new_add_vip_delReverse['name'].isin(
    new_supplier['name'])].reset_index(drop=True)
new_add_vip_delReverse_old = new_add_vip_delReverse[new_add_vip_delReverse['name'].isin(
    old_supplier['name'])].reset_index(drop=True)
new_add_vip_delReverse_canNotFound = new_add_vip_delReverse[~new_add_vip_delReverse['name'].isin(
    all_supplier['name'])].reset_index(drop=True)

pre_right = new_add_vip_delReverse_old[new_add_vip_delReverse_old['name'].isin(
    pre_positive['name'])].reset_index(drop=True)

'''
# 长江三角洲地区
Yangtze_River_Delta = set({'上海市', '江苏省', '浙江省', '安徽省'})
pre_positive_del_newAdd = pre_positive[~pre_positive['name'].isin(
    new_vip['name'])].reset_index(drop=True)

pre_positive_del_newAdd = pre_positive_del_newAdd.sort_values(
    'probability', ascending=False).reset_index(drop=True)

# top50给产品
top50_for_product = pre_positive_del_newAdd.iloc[:50]
top50_for_product.iloc[:, :2].to_csv('./dataResult/产品.csv', index=False)

# 其余按照地区分为长江三角洲和其他
else_for_operation = pre_positive_del_newAdd.iloc[50:]
Yangtze_River_supplier = else_for_operation[else_for_operation['province'].isin(
    Yangtze_River_Delta)].reset_index(drop=True)
Yangtze_River_supplier.iloc[:, :2].to_csv(
    './dataResult/长三角地区.csv', index=False)
except_Yangtze_River_supplier = else_for_operation[~else_for_operation['province'].isin(
    Yangtze_River_Delta)].reset_index(drop=True)
except_Yangtze_River_supplier.iloc[:, :2].to_csv(
    './dataResult/其余地区.csv', index=False)
'''
