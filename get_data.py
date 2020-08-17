import pandas as pd
import numpy as np
import pymysql
from datetime import datetime
from utils import readDataFromMysql


def computeTime(rc_time):
    time_init = datetime(2015, 4, 8)
    endTime = datetime.today()

    rc_time.columns = ['regchecktime', 'createtime']
    rc_time.loc[rc_time['regchecktime'] ==
                '0000-00-00 00:00:00', 'regchecktime'] = time_init
    rc_time.loc[rc_time['createtime'] ==
                '0000-00-00 00:00:00', 'createtime'] = time_init

    time = pd.DataFrame(rc_time.max(axis=1))
    days = (endTime-time).apply(lambda x: x[0].days, axis=1)
    years = np.array(days)/365

    return pd.DataFrame(years)


# zjc_supplier和zjc_supplier_param中的维度
sql = 'select zjc_supplier.id,zjc_supplier.name,zjc_supplier.province,zjc_supplier_param.attach_count,zjc_supplier.type,zjc_supplier.fund,zjc_supplier_param.service_level_average,zjc_supplier_param.instock_service_average,zjc_supplier_param.instock_product_average,zjc_supplier_param.instock_deliverspeed_average,zjc_supplier_param.iswin_count,zjc_supplier_param.price_level_average,zjc_supplier_param.tender_count,zjc_supplier_param.semester_tender_count,zjc_supplier_param.bidconcern_count,zjc_supplier_param.semester_bidconcern_count,zjc_supplier_param.login_days,zjc_supplier_param.semester_login_days,zjc_supplier_param.integrity_count,zjc_supplier_param.contract_rate,zjc_supplier_param.instock_honesty_average,zjc_supplier.regchecktime,zjc_supplier.createtime from zjc_supplier INNER JOIN zjc_supplier_param on zjc_supplier.id=zjc_supplier_param.supplier_id where zjc_supplier.state=2;'
# sql = 'select zjc_supplier.name,zjc_supplier.province,zjc_supplier_param.attach_count,zjc_supplier.type,zjc_supplier.fund,zjc_supplier_param.service_level_average,zjc_supplier_param.instock_service_average,zjc_supplier_param.instock_product_average,zjc_supplier_param.instock_deliverspeed_average,zjc_supplier_param.iswin_count,zjc_supplier_param.price_level_average,zjc_supplier_param.tender_count,zjc_supplier_param.semester_tender_count,zjc_supplier_param.bidconcern_count,zjc_supplier_param.semester_bidconcern_count,zjc_supplier_param.login_days,zjc_supplier_param.semester_login_days,zjc_supplier_param.integrity_count,zjc_supplier_param.contract_rate,zjc_supplier_param.instock_honesty_average,zjc_supplier.regchecktime,zjc_supplier.createtime from zjc_supplier, zjc_supplier_param where zjc_supplier.id=zjc_supplier_param.supplier_id and zjc_supplier.state=2;'
# all_supplier:所有供应商，包括训练用的正负样本
all_supplier = readDataFromMysql(sql)
all_supplier.columns = ['company_id', 'name', 'province', 'attach_count', 'type', 'fund', 'service_average', 'instock_service', 'instock_product', 'instock_deliverspeed', 'win', 'price',
                        'tender', 'se_tender', 'bidconcern', 'se_bidconcern', 'login', 'se_login', 'integrity', 'contract_rate', 'instock_honesty', 'regchecktime', 'createtime']

# recommend.zjc_intrec_supplier_operator_log中的VIP埋点数据
vip_spot_sql = 'SELECT zjc_user.company_id,zjc_supplier.`name`,recommend.zjc_intrec_supplier_operator_log.operator_type as operator_type,COUNT(recommend.zjc_intrec_supplier_operator_log.operator_type) as operate_num from zjc_user,zjc_supplier,recommend.zjc_intrec_supplier_operator_log WHERE zjc_supplier.id=zjc_user.company_id and zjc_user.id=recommend.zjc_intrec_supplier_operator_log.user_id GROUP BY zjc_user.company_id,recommend.zjc_intrec_supplier_operator_log.operator_type'
vip_spot = readDataFromMysql(vip_spot_sql)
vip_spot.columns = ['company_id', 'name', 'operator_type', 'operate_num']

# VIP埋点数据转化为8个不同埋点维度和总共操作数维度
all_supplier_columns = all_supplier.columns.to_list()
vip_spot_columns = ['type1', 'type2', 'type3', 'type4',
                    'type5', 'type6', 'type7', 'type8', 'operate_total']

all_supplier_columns[-2:-2] = vip_spot_columns
all_supplier = all_supplier.reindex(columns=all_supplier_columns)

vip_spot_in_all = vip_spot[vip_spot['company_id'].isin(
    all_supplier['company_id'])]
# 这部分供应商均为state=3即审核不通过的供应商，审核不通过也可以开通VIP
vip_spot_not_in_all = vip_spot[~vip_spot['company_id'].isin(
    all_supplier['company_id'])]

# 将company_id作为索引值，方便索引
all_supplier.set_index(['company_id'], inplace=True)

for row in vip_spot_in_all.itertuples():
    operator_type = getattr(row, 'operator_type')
    operator_type = 'type'+str(operator_type)
    operate_num = getattr(row, 'operate_num')
    company_id = getattr(row, 'company_id')
    all_supplier.loc[company_id, operator_type] = operate_num

# 计算total
all_supplier['operate_total'] = all_supplier.loc[:,
                                                 'type1':"type8"].sum(axis=1)

# 恢复索引
all_supplier = all_supplier.reset_index(drop=False)
all_supplier.fillna(value=0, inplace=True)


# 计算注册时长
time = computeTime(all_supplier.iloc[:, -2:])
time.columns = ['years']
all_supplier = pd.concat([all_supplier.iloc[:, 0:-2], time], axis=1)


# 读取正负样本名单
pay = pd.read_csv("./data/vip_1.csv")
pay = pay.drop_duplicates()
refuse = pd.read_csv("./data/vip_0.csv")
refuse = refuse.drop_duplicates()

# 去除refuse中后来又充值的供应商
refuse = refuse.append(pay)
refuse = refuse.append(pay)
refuse = refuse.drop_duplicates(keep=False, subset=['name'])

# 没有查到的供应商
sample_list = pd.concat([pay, refuse]).reset_index(drop=True)
unfound_sample = sample_list[~sample_list['name'].isin(
    all_supplier['name'])].reset_index(drop=True)

# 查到的供应商
data_pay = all_supplier[all_supplier['name'].isin(
    pay['name'])].reset_index(drop=True)
data_refuse = all_supplier[all_supplier['name'].isin(
    refuse['name'])].reset_index(drop=True)
sample_data = pd.concat([data_pay, data_refuse]).reset_index(drop=True)

# 添加标签
sample_data.insert(2, 'label', 1)
sample_data.iloc[len(pay):, 2] = 0

# 总样本中的新供应商（包含正负样本）
vip_new_supplier = sample_data.query('years<0.25')
vip_new_supplier.to_csv("./dataDump/vip_new_supplier.csv", index=False)

# 总样本中的老供应商（包含正负样本）
vip_data = sample_data.query('years>=0.25')
vip_data.to_csv('./dataDump/vip_data.csv', index=False)


# --------------- 所有要预测的供应商数据 ------------------
# 去除正负样本
all_supplier = all_supplier[~all_supplier['name'].isin(
    sample_data['name'])].reset_index(drop=True)
# 去除战略供应商
strategy = pd.read_csv('./data/strategy.csv')
all_supplier = all_supplier[~all_supplier['name'].isin(
    strategy['name'])].reset_index(drop=True)
# 除去因各种其他原因不开通的供应商
other = pd.read_csv('./data/other.csv')
all_supplier = all_supplier[~all_supplier['name'].isin(
    other['name'])].reset_index(drop=True)

all_supplier.to_csv("./dataDump/pre_data.csv", index=False)
