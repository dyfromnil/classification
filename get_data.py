import pandas as pd
import numpy as np
import pymysql
from datetime import datetime


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


unfound = []


def getdata(data_pd, label):
    db = pymysql.connect('localhost', 'root', 'Qq56138669', 'zjc')
    cursor = db.cursor()
    # data_list = []
    for row in data_pd.itertuples():
        name = getattr(row, 'name')
        sql_order = "select zjc_supplier.name,zjc_supplier_param.attach_count,zjc_supplier.type,zjc_supplier.fund,zjc_supplier_param.service_level_average,zjc_supplier_param.instock_service_average,zjc_supplier_param.instock_product_average,zjc_supplier_param.instock_deliverspeed_average,zjc_supplier_param.iswin_count,zjc_supplier_param.price_level_average,zjc_supplier_param.tender_count,zjc_supplier_param.semester_tender_count,zjc_supplier_param.bidconcern_count,zjc_supplier_param.semester_bidconcern_count,zjc_supplier_param.login_days,zjc_supplier_param.semester_login_days,zjc_supplier_param.integrity_count,zjc_supplier_param.contract_rate,zjc_supplier_param.instock_honesty_average,zjc_supplier.regchecktime,zjc_supplier.createtime from zjc_supplier,zjc_supplier_param where zjc_supplier.id=zjc_supplier_param.supplier_id and zjc_supplier.state=2 and zjc_supplier.name='"+name+"';"

        try:
            cursor.execute(sql_order)
            data = list(cursor.fetchone())
            data.insert(1, label)
            yield tuple(data)
            # data_list.append(data)
        except:
            data = ()
            if label == 1:
                print("正样本，未查到："+name)
            if label == 0:
                print("负样本，未查到："+name)
            unfound.append((name, label))
    db.close()
    # return pd.DataFrame(data_list)
    pd.read_sql


# 读取正负样本名单
data_pay = pd.read_csv("./data/vip_1.csv")
data_pay = data_pay.drop_duplicates()
data_refuse = pd.read_csv("./data/vip_0.csv")
data_refuse = data_refuse.drop_duplicates()

# 查询正负样本相关数据
data_pay = pd.DataFrame([data for data in getdata(data_pay, 1)])
data_refuse = pd.DataFrame([data for data in getdata(data_refuse, 0)])

# 计算注册时长
time = computeTime(data_pay.iloc[:, -2:])
data_pay = pd.concat([data_pay.iloc[:, 0:-2], time], axis=1)
time = computeTime(data_refuse.iloc[:, -2:])
data_refuse = pd.concat([data_refuse.iloc[:, 0:-2], time], axis=1)

# 改列名
data_pay.columns = ['name', 'label', 'attach_count', 'type', 'fund', 'service_average', 'instock_service', 'instock_product', 'instock_deliverspeed', 'win', 'price',
                    'tender', 'se_tender', 'bidconcern', 'se_bidconcern', 'login', 'se_login', 'integrity', 'contract_rate', 'instock_honesty', 'years']
data_refuse.columns = ['name', 'label', 'attach_count', 'type', 'fund', 'service_average', 'instock_service', 'instock_product', 'instock_deliverspeed', 'win', 'price',
                       'tender', 'se_tender', 'bidconcern', 'se_bidconcern', 'login', 'se_login', 'integrity', 'contract_rate', 'instock_honesty', 'years']

# 去除data_refuse中后来又充值的供应商
data_refuse = data_refuse.append(data_pay)
data_refuse = data_refuse.append(data_pay)
data_refuse = data_refuse.drop_duplicates(keep=False, subset=['name'])

# data 整理好的所有的正负样本
data = pd.concat([data_pay, data_refuse])

# 开通了VIP的years<0.25的新供应商
vip_new_supplier = data.query('years<0.25 and label ==1')
vip_new_supplier.to_csv("./dataDump/vip_new_supplier.csv", index=False)

# 没有查到的供应商
unfound = pd.DataFrame(unfound, columns=['name', 'label'])
unfound.to_csv('./dataDump/unfound.csv', index=False)

# 所有的老供应商(包含正负样本)
vip_data = data.query('years>=0.25')
vip_data.to_csv('./dataDump/vip_data.csv', index=False)
