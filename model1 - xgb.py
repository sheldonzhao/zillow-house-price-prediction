import pandas as pd
import utils
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import zipfile
import operator
from sys import maxsize

# 1. split training set ans test set
train_df = pd.read_csv(utils.file_merge_data)
test_set = train_df.copy()
train_set = train_df[train_df['logerror'].notnull()]

# 2. filter float and int features
table_type = train_df.dtypes.reset_index()
table_type.columns = ["feature", "Type"]
# print(table_type.groupby("Type").aggregate('count'))
float_feature_set = table_type[table_type['Type'] == 'float64']['feature']
print('the number of float features is %d', len(float_feature_set))
print(float_feature_set)

# 3. preprocess
feature_set = list(float_feature_set.copy())
for c in train_set.dtypes[train_set.dtypes == object].index.values:
    train_set[c] = (train_set[c] == True)
    test_set[c] = (test_set[c] == True)
    feature_set.append(c)
    print(c)

train_set = train_set[feature_set]
test_set = test_set[feature_set]
y_label = train_set.pop('logerror')
test_set.pop('logerror')
del_feat = ['transactiondate', 'propertyzoningdesc', 'propertycountylandusecode']
train_set.drop(del_feat, axis=1, inplace=True)
test_set.drop(del_feat, axis=1, inplace=True)
print('the number of features set is %d', len(train_set.columns))

'''
# drop useless features
del_feat = []
for i in train_set.columns:
    length = train_set[i].size
    nan_length = train_set[train_set[i].isnull()].size // len(train_set.columns)
    if nan_length / float(length) >= 0.95:
        del_feat.append(i)
train_set.drop(del_feat, axis=1, inplace=True)
test_set.drop(del_feat, axis=1, inplace=True)
print('delete features : %s' % del_feat)
print('the number of float features is %d', len(train_set.columns))
'''

# 4. xgb
X_train, X_test, y_train, y_test = train_test_split(train_set, y_label, test_size=0.2, random_state=0)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
test_set = xgb.DMatrix(test_set)
param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3,
         'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
         'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1}
num_round = 45
watchList = [(dtest, 'eval'), (dtrain, 'train')]
plst = list(param.items()) + [('eval_metric', 'logloss')]
bst = xgb.train(plst, dtrain, num_round, watchList)
y = bst.predict(test_set)
res = pd.DataFrame(
    {'201610': y, '201611': y, '201612': y, '201710': y, '201711': y,
     '201712': y})
res = pd.concat([train_df['parcelid'], res], axis=1)
print('test')
# use original value instead of predicted value
res['logerror2'] = train_df['logerror']
res['logerror2'] = res['logerror2'].fillna(-maxsize)
for index in range(len(res['parcelid'])):
    value = res.ix[index]['logerror2']
    if value != -maxsize:
        res[['201610', '201611', '201612', '201710', '201711', '201712']] = value
res = res[['parcelid', '201610', '201611', '201612', '201710', '201711', '201712']]

res.to_csv(utils.file_output, index=False)

f = zipfile.ZipFile('output.zip', 'w', zipfile.ZIP_DEFLATED)
f.write('output.csv')

'''
# feature importance
feature_score = bst.get_fscore()
feature_score = sorted(feature_score.items(), key=operator.itemgetter(1))
print(feature_score)
df = pd.DataFrame(feature_score, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(16, 10))
plt.show()

'''
