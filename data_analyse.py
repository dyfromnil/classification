import pandas as pd
from utils import computeTime
from utils import readDataFromMysql

# 下发名单7数据分析

# 上次的预测名单
pre5 = pd.read_csv('dataAnalyse/5/total.csv')
pre6 = pd.read_csv('dataAnalyse/6/total.csv')
pre_last = pd.concat([pre5, pre6]).reset_index(drop=True)
pre_last = pre_last.drop_duplicates('name').reset_index(drop=True)

# 本次名单期间除去上次转化的，还剩下的（纯本次新增的VIP供应商）
real7 = pd.read_csv('dataAnalyse/vip7.csv')
real7 = real7[~real7['name'].isin(pre_last['name'])].reset_index(drop=True)

# 后一次名单下发期间新增的VIP供应商
real8 = pd.read_csv('dataAnalyse/vip8.csv')

# 本次和后一次的预测名单
pre7 = pd.read_csv('dataAnalyse/7/total.csv')
pre8 = pd.read_csv('dataAnalyse/8/total.csv')


sql = 'select zjc_supplier.name,zjc_supplier.province,zjc_supplier.regchecktime,zjc_supplier.createtime from zjc_supplier INNER JOIN zjc_supplier_param on zjc_supplier.id=zjc_supplier_param.supplier_id where zjc_supplier.state=2;'
# all_supplier:所有供应商，包括训练用的正负样本
all_supplier = readDataFromMysql(sql)
all_supplier.columns = ['name', 'province', 'regchecktime', 'createtime']
all_supplier = all_supplier.drop_duplicates(keep=False, subset=['name'])
# 最后两列是注册和审核时间，用于计算已经成为平台用户的时长
time_all = computeTime(all_supplier.iloc[:, -2:])
time_all.columns = ['years']
all_supplier = pd.concat([all_supplier.iloc[:, :-2], time_all], axis=1)
all_supplier = all_supplier.fillna(0)


# 本次名单下发期间新增VIP供应商中查到的、未查到的供应商（一般是state问题）
find = all_supplier[all_supplier['name'].isin(
    real7['name'])].reset_index(drop=True)
unfind = real7[~real7['name'].isin(
    all_supplier['name'])].reset_index(drop=True)
# 查到的供应商中新老供应商个数
find.query('years>0.25')
find.query('years<=0.25')

# 本次名单期间预测中的
this_term = find[find['name'].isin(
    pre7['name'])].reset_index(drop=True)
this_term.query('years>0.25')
this_term.query('years<=0.25')

# 下次名单下发期间开通VIP落在本次预测名单中的
find_real8 = all_supplier[all_supplier['name'].isin(
    real8['name'])].reset_index(drop=True)
next_term = find_real8[find_real8['name'].isin(
    pre8['name'])].reset_index(drop=True)
next_term.query('years>0.25')
next_term.query('years<=0.25')
