
new = all_supplier.query(
    'days<=0.0192 or (days>=0.0685 and days<=0.0959) or (days>=0.1507 and days<=0.1781)').reset_index(drop=True)

name_province = new[['name', 'province']]

pre_data_new = new.iloc[:, 2:-1]
pre_data_new = pre_data_new.astype('float32')
pre_data_new = np.array(pre_data_new)

pre_data_new_DMatrix = xgb.DMatrix(pre_data_new)
probability_new_new = bst.predict(
    pre_data_new_DMatrix, ntree_limit=bst.best_ntree_limit)
probability_new = pd.DataFrame(probability_new)
probability_new.columns = ['probability_new']
pre_result = pd.concat([name_province, probability_new], axis=1)

pre_positive = pre_result[pre_result['probability_new'] > 0.5]
pre_positive_sort = pre_positive.sort_values(
    'probability_new', ascending=False).reset_index(drop=True)

pre_positive_sort[:50].to_csv('./new_pre_positive_sort.csv', index=False)
